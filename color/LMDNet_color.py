# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import re
from torch.nn import init


def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)
    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)


class Regularizer(nn.Module):
    def __init__(self, image_channels, n_channels, kernel_size=3, padding=1):
        super(Regularizer, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        lu = self.conv(x)
        out = self.relu(lu)
        return lu, out


class Self_Attn(nn.Module):
    def __init__(self, in_dim, K):
        super(Self_Attn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=K * in_dim, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=K * in_dim, out_channels=in_dim, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus(beta=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.softmax(x)
        x = self.conv2(x)
        attention = self.softplus(x)
        return attention


class LMDNet(nn.Module):
    def __init__(self, block=Regularizer, attention=Self_Attn, n_channels=64, image_channels=3, kernel_size=3):
        super(LMDNet, self).__init__()
        padding = 1
        sf = 2

        self.conv = nn.Conv2d(image_channels * sf * sf, n_channels, kernel_size=kernel_size, padding=padding)
        self.conv_t = nn.Conv2d(n_channels, image_channels * sf * sf, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=False)
        self.m_up = nn.PixelShuffle(upscale_factor=sf)
        self.m_down = PixelUnShuffle(upscale_factor=sf)

        for i in range(15):
            setattr(self, f"layer{i}", self._make_layer(block, n_channels))
            setattr(self, f"conv{i}", nn.ConvTranspose2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1))
            setattr(self, f"atten{i}", self._make_layera(attention, n_channels, K=2))

    def _make_layer(self, block, n_channels):
        return nn.Sequential(block(n_channels, n_channels))

    def _make_layera(self, block, n_channels, K):
        return nn.Sequential(block(n_channels, K))

    def _feature_extrator(self, x):
        return torch.where(x > 0, 0.5 * x ** 2, torch.tensor(0.).cuda() if x.is_cuda else torch.tensor(0.))

    def forward(self, x):
        h, w = x.size()[-2:]
        x = nn.ReplicationPad2d((0, (2 - w % 2) % 2, 0, (2 - h % 2) % 2))(x)
        x = self.m_down(x)
        f = self.conv(x)

        u = f
        for i in range(15):
            layer = getattr(self, f"layer{i}")
            conv = getattr(self, f"conv{i}")
            atten = getattr(self, f"atten{i}")
            l_u, regu_u = layer(u)
            lamb = atten(self._feature_extrator(l_u))
            u = u + 1e-3 * (f - u) - conv(torch.mul(lamb, regu_u)) if i > 0 else f - conv(torch.mul(lamb, regu_u))

        x = self.conv_t(u)
        x = self.m_up(x)
        return x[..., :h, :w]


class sum_squared_error(nn.Module):
    def __init__(self):
        super(sum_squared_error, self).__init__()

    def forward(self, input, target):
        return F.mse_loss(input, target, reduction='sum').div_(2)


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
    if file_list:
        epochs_exist = [int(re.findall(".*checkpoint_(.*).pth.*", file_)[0]) for file_ in file_list]
        return max(epochs_exist)
    else:
        return 0
