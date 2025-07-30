# run this to test the model

import os, time, datetime,argparse,re
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from compare_index import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from LMDU import Self_Attn,Weight,VNet,WNet



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set12','BSD68','urban100_gray'], help='directory of test dataset')
    parser.add_argument('--sigma', default=50, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models', 'lmdu'), help='directory of the model')
    parser.add_argument('--model_name', default='model_50.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='result_lmdu', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    result = np.clip(result, 0, 1)  # ensure in [0, 1]
    result = (result * 255).round().astype(np.uint8)
    imsave(path, result)


def save_result_txt(names, psnrs, ssims, path, sigma):
    with open(path, 'w') as f:
        f.write('Name\tPSNR\tSSIM\n')
        for name, psnr, ssim in zip(names, psnrs[:-1], ssims[:-1]):  # 不写入 avg 行
            f.write(f'{name}\t{psnr:.2f}\t{ssim:.4f}\n')
        f.write('\n')
        f.write(f'Noise level: {sigma}, Average PSNR: {psnrs[-1]:.2f} dB, Average SSIM: {ssims[-1]:.4f}\n')


def save_result_excel(names, psnrs, ssims, path, sigma):
    data = {
        'Image Name': names + ['Average'],
        'PSNR': [f'{v:.2f}' for v in psnrs],
        'SSIM': [f'{v:.4f}' for v in ssims],
    }
    df = pd.DataFrame(data)
    
    # 添加 noise level 信息在表格底部一行（可选）
    df.loc[len(df)] = [f'Noise level: {sigma}', '', '']
    
    df.to_excel(path, index=False)
            
def extract_numbers(filename):
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers] if numbers else [float('inf')]

def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

if __name__ == '__main__':

    args = parse_args()

    model = WNet()
    
    # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
    checkpoint = torch.load(os.path.join(args.model_dir, args.model_name))
    model.load_state_dict(checkpoint['net'])
    log('load trained model')

    model.eval()  # evaluation mode

    if torch.cuda.is_available():
        model = model.cuda()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:
        if not os.path.exists(os.path.join(args.result_dir,  str(args.sigma),set_cur)):
            os.makedirs(os.path.join(args.result_dir,  str(args.sigma),set_cur))
        psnrs = []
        ssims = []
        names = []
        img_dir = os.path.join(args.set_dir, set_cur)
        img_list = [im for im in os.listdir(img_dir)
                    if os.path.splitext(im)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']]
        img_list = sorted(img_list, key=extract_numbers)
        
        for im in img_list:
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                name, ext = os.path.splitext(im)
                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32)
                
                np.random.seed(seed=0)  # for reproducibility
                y = (x + np.random.normal(0, args.sigma, x.shape)) / 255  # Add Gaussian noise without clipping
                
                y = y.astype(np.float32)
                x = x / 255.
                with torch.no_grad():
                    y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
                    B, C, H, W = y_.size()
                # padding to fit the input size of UNet
                    bottom = (16 - H % 16) % 16
                    right = (16 - W % 16) % 16

                    padding = nn.ReflectionPad2d((0, right, 0, bottom))
                    y_ = padding(y_)
                

                    torch.cuda.synchronize()
                    start_time = time.time()
                    y_ = y_.cuda()
                    with torch.no_grad():
                        x_ = model(y_)  # inference
                    x_ = x_[:, :, 0:H, 0:W].view(y.shape[0], y.shape[1])
                    x_ = x_.cpu()
                    x_ = x_.detach().numpy().astype(np.float32)
                    y_ = y_[:, :, 0:H, 0:W].view(y.shape[0], y.shape[1])
                    y_ = y_.view(y.shape[0], y.shape[1])
                    y_ = y_.cpu()
                    y_ = y_.detach().numpy().astype(np.float32)
                    torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
               # print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))
                psnr_x_ = compare_psnr(255*x, 255*x_)
                ssim_x_ = compare_ssim(255*x, 255*x_)
                print('%10s : %10s ,time: %2.4f second, psnr: %2.2f' % (set_cur, im, elapsed_time,psnr_x_))
                if args.save_result:
                    save_result(x_, path=os.path.join(args.result_dir,  str(args.sigma),set_cur,
                                                      name + '_lmd' + ext))  # save the denoised image
                    save_result(y, path=os.path.join(args.result_dir,  str(args.sigma),set_cur, name + '_noisy' + ext))
            
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)
                name, ext = os.path.splitext(im)
                names.append(name)
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        
        model_tag = os.path.basename(args.model_dir.rstrip('/'))
        result_filename_excel = f'{model_tag}_{set_cur}_{args.sigma}_results.xlsx'
        save_result_excel(
            names, psnrs, ssims,
            path=os.path.join(args.result_dir, str(args.sigma), set_cur, result_filename_excel),
            sigma=args.sigma
        )
        
        result_filename_txt = f'{model_tag}_{set_cur}_{args.sigma}_results.txt'
        save_result_txt(
            names, psnrs, ssims,
            path=os.path.join(args.result_dir, str(args.sigma), set_cur, 'result_filename_txt.txt'),
            sigma=args.sigma
        )
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))
