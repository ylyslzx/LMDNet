# run this to test the model

import argparse,re
import os,  datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch.nn as nn
import torch
from compare_index import compare_psnr, compare_ssim
from skimage.io import  imsave
import cv2
from LMDNet_color import LMDNet
from torch.autograd import Variable
import pandas as pd

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')
	parser.add_argument('--set_names', default=['Kodak24','CBSD68','urban100_color'], help='directory of test dataset')  # DIV2K_valid_HR
	parser.add_argument('--sigma', default=15, type=int, help='noise level')
	parser.add_argument('--model_dir', default=os.path.join('models', 'lmd'),
	                    help='directory of the model')
	parser.add_argument('--model_name', default='model_15.pth', type=str, help='the model name')
	parser.add_argument('--result_dir', default='result_lmd', type=str, help='directory of test dataset')
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

if __name__ == '__main__':
	args = parse_args()
	model = LMDNet()
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
			# os.mkdir(os.path.join(args.result_dir, set_cur))
			os.makedirs(os.path.join(args.result_dir,  str(args.sigma),set_cur))
		psnrs = []
		ssims = []
		names = []
		img_dir = os.path.join(args.set_dir, set_cur)
		img_list = [im for im in os.listdir(img_dir)
		            if os.path.splitext(im)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']]
		img_list = sorted(img_list, key=extract_numbers)
		
		for im in img_list:
			name, ext = os.path.splitext(im)
			Img = cv2.imread(os.path.join(args.set_dir, set_cur, im))
			clean = Img
			
			Img = torch.tensor(Img)
			
			Img = Img.permute(2, 0, 1)
			
			Img = Img.numpy()
			a1, a2, a3 = Img.shape
			Img = np.tile(Img, (1, 1, 1, 1))  # expand the dimensional
			Img = np.float32(Img / 255.)
			ISource = torch.Tensor(Img)
			# noise
			torch.manual_seed(0)
			noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=args.sigma / 255.)
			# noisy image
			INoisy = ISource + noise
			
			################padding#########
			
			
			B, C, H, W = INoisy.size()
			bottom = (16 - H % 16) % 16
			right = (16 - W % 16) % 16
			padding = nn.ReflectionPad2d((0, right, 0, bottom))
			
			INoisy = padding(INoisy)
			ISource = Variable(ISource)
			INoisy = Variable(INoisy)
			ISource = ISource.cuda()
			INoisy = INoisy.cuda()
			with torch.no_grad():  # this can save much memory
				Out = torch.clamp(model(INoisy), 0., 1.)
			
			denoised = Out[:, :, 0:H, 0:W].data.squeeze().float().clamp_(0, 1).cpu().numpy()
			denoised = np.transpose(denoised, (1, 2, 0))
			Imgn = INoisy[:, :, 0:H, 0:W].data.squeeze().float().clamp_(0, 1).cpu().numpy()
			Imgn = np.transpose(Imgn, (1, 2, 0))
			noise = noise.data.squeeze().float().cpu()
			
			psnr = compare_psnr(clean, denoised * 255)
			ssim = compare_ssim(clean, denoised * 255)
			
			print('on image %s, psnr:%2f,ssim:%4f' % (name, psnr, ssim))
			names.append(name)
			psnrs.append(psnr)
			ssims.append(ssim)
			if args.save_result:
				cv2.imwrite(os.path.join(args.result_dir,  str(args.sigma),set_cur, name + '_denoised' + ext),
				            denoised * 255)  # save the denoised image
				cv2.imwrite(os.path.join(args.result_dir,  str(args.sigma),set_cur, name + '_noisy' + ext),
				            Imgn * 255)
		
		psnr_avg = np.mean(psnrs)
		ssim_avg = np.mean(ssims)
		psnrs.append(psnr_avg)
		ssims.append(ssim_avg)
		print("on dataset %s, the avearge PSNR is %.2f, average SSIM is %.4f" % (set_cur, psnr_avg, ssim_avg))
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