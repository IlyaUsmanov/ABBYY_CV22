from skimage import img_as_ubyte
from skimage.io import imread, imsave
from time import time
from numpy import prod

from methods.improved_interpolation import improved_interpolation
from methods.bilinear_interpolation import bilinear_interpolation
from methods.VNG import VNG

from utils.utils import compute_psnr


def process_img(bmp_img, gt_img, method, method_name):
    start = time()
    res = img_as_ubyte(method(bmp_img))
    time_elapsed = time() - start
    mp_amount = prod(bmp_img.shape) / 1_000_000
    print(f'{method_name}\nPSNR: {compute_psnr(res, gt_img)}\nTime elapsed per 1 MP: {time_elapsed / mp_amount}\n------------\n')
    imsave(f'results/{method_name}.png', res)

bmp_img = img_as_ubyte(imread('data/CFA.bmp'))
gt_img = img_as_ubyte(imread('data/Original.bmp'))

process_img(bmp_img, gt_img, bilinear_interpolation, 'biliniear_interpolation')
process_img(bmp_img, gt_img, improved_interpolation, 'improved_interpolation')
process_img(bmp_img, gt_img, VNG, 'VNG')