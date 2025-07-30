# === model.py ===
import os
import re
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def act(x, alpha):
    d = 1 + alpha * x ** 2
    return torch.log(d) / (2 * alpha)


class Self_Attn(nn.Module):
    def __init__(self, in_dim, K):
        super(Self_Attn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=K * in_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=K * in_dim, out_channels=in_dim, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus(beta=1)
    
    def _feature_extractor(self, x):
        y = torch.where(x > 0, 0.5 * torch.pow(x, 2), torch.tensor(0.).cuda())
        return y
    
    def forward(self, x):
        x = self._feature_extractor(x)
        x = self.conv1(x)
        x = self.softmax(x)
        x = self.conv2(x)
        attention = self.softplus(x)
        return attention


class OperatorA(nn.Module):  ##  I+L*(WR)*L(u)=A(u)
    def __init__(self, n_channels, kernel_size=3, padding=1):
        super(OperatorA, self).__init__()
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.convt = nn.ConvTranspose2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1,
                                        padding=1,
                                        output_padding=0, groups=1, bias=False, dilation=1)
        
        self.attention = Self_Attn(n_channels, K=2)
    
    def forward(self, u):
        lu = self.conv(u)
        out = self.relu(lu)
        lamb = self.attention(lu)
        u = self.convt(torch.mul(lamb, out)) + u
        return u


class Weight(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Weight, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels=2 * channel, out_channels=channel, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.conv_du(x)
        return y


class VNet(nn.Module):
    def __init__(self, scale_num=3, n_channels=64):
        super(VNet, self).__init__()
        self.scale_num = scale_num
        
        self.opeA = []
        for i in range(self.scale_num):
            b = torch.nn.ModuleList([
                OperatorA(n_channels=n_channels),
                OperatorA(n_channels=n_channels)
            ])
            self.opeA.append(b)
        
        self.opeA.append(torch.nn.ModuleList([
            OperatorA(n_channels=n_channels)
        ]))
        self.opeA = torch.nn.ModuleList(self.opeA)
        
        self.conv_down = []
        self.conv_up = []
        for i in range(1, self.scale_num):
            self.conv_down.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2,
                                            padding=1, bias=False))
            self.conv_up.append(nn.ConvTranspose2d(in_channels=n_channels, out_channels=n_channels, kernel_size=2,
                                                   stride=2, bias=False))
        self.conv_down = torch.nn.ModuleList(self.conv_down)
        self.conv_up = torch.nn.ModuleList(self.conv_up)
        
        self.weights = []
        
        for i in range(1, self.scale_num):
            b = torch.nn.ModuleList([
                Weight(n_channels)
            ])
            self.weights.append(b)
        self.weights = torch.nn.ModuleList(self.weights)
    
    def forward(self, u0):
        assert len(u0) == self.scale_num
        f = [u0[0], ]
        uT = u0.copy()
        # down scale and feature extraction
        for i in range(self.scale_num - 1):
            res_0 = f[i] - self.opeA[i][0](u0[i])
            uT[i] = u0[i] + res_0
            
            res_1 = f[i] - self.opeA[i][0](uT[i])
            x_i_down = self.conv_down[i](uT[i])
            
            if uT[i + 1] is None:
                u0[i + 1] = x_i_down
            else:
                weight = self.weights[i][0](torch.cat((uT[i + 1], x_i_down), 1))
                u0[i + 1] = torch.mul(weight, uT[i + 1]) + torch.mul((1 - weight), x_i_down)
            weight = self.weights[i][0](torch.cat((res_0, res_1), 1))
            f.append(self.conv_down[i](torch.mul(weight, res_0) + torch.mul((1 - weight), res_1))
                     + self.opeA[i + 1][0](u0[i + 1]))
        
        uT[-1] = f[-1] - self.opeA[-1][0](u0[-1])
        # print(uT[-1].shape,torch.equal(uT[-1],u0[-1]))
        
        for i in range(self.scale_num - 1)[::-1]:
            uT[i] = uT[i] + self.conv_up[i](uT[i + 1] - u0[i + 1])
            #   print(torch.equal(self.conv_up[i](uT[i + 1] - u0[i + 1]),torch.zeros_like(uT[i])))
            uT[i] = uT[i] + f[i] - self.opeA[i][1](uT[i])
        return uT


class WNet(nn.Module):
    def __init__(self, block_num=3, scale_num=3):
        super(WNet, self).__init__()
        self.num_mb = block_num
        self.in_channels = 3
        self.n_channels = 64
        self.scale_num = scale_num
        
        self.K1 = nn.Conv2d(self.in_channels, self.n_channels, kernel_size=3, padding=1, bias=False)
        
        self.mb = torch.nn.ModuleList([VNet() for _ in range(self.num_mb)])
        
        self.KN = nn.Conv2d(self.n_channels, self.in_channels, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        x = self.K1(x)
        # apply mb
        x = [x, ] + [None for i in range(self.scale_num - 1)]
        
        for i in range(self.num_mb):
            x = self.mb[i](x)
        # compute the output
        out = self.KN(x[0])
        return out


# === Utility ===
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.orthogonal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'checkpoint_*.pth'))
    if not file_list:
        return 0
    epochs = [int(re.findall("checkpoint_(\\d+).pth", f)[0]) for f in file_list]
    return max(epochs)
