import numpy as np
from utils import get_pixels_by_coords, rotate_coords

def bilinear_interpolation(img, coords):
    upper_bound = np.ceil(coords).astype(int)
    lower_bound = np.floor(coords).astype(int)
    c1 = np.stack((upper_bound[...,0], lower_bound[...,1]), axis=-1)
    c2 = np.stack((lower_bound[...,0], upper_bound[...,1]), axis=-1)
    return get_pixels_by_coords(img, upper_bound) * np.prod(abs(coords - lower_bound), axis=-1)[...,None] +\
            get_pixels_by_coords(img, lower_bound) * np.prod(abs(coords - upper_bound), axis=-1)[...,None] +\
            get_pixels_by_coords(img, c1) * np.prod(abs(coords - c2), axis=-1)[...,None] +\
            get_pixels_by_coords(img, c2) * np.prod(abs(coords - c1), axis=-1)[...,None]

def nearest_neighbour_interpolation(img, coords):
    coords = np.round(coords).astype(int)
    return get_pixels_by_coords(img, coords)

def get_interpolated_img(img, coords, interpolation_type='nearest_neighbour'):
    if interpolation_type == 'nearest_neighbour':
        return nearest_neighbour_interpolation(img, coords)
    if interpolation_type == 'bilinear':
        return bilinear_interpolation(img, coords)
    raise ValueError('Unsupported interpolation type')

def resize_img(img):
    h, w, _ = img.shape
    h_pad, w_pad = 2 ** np.ceil(np.log2((h, w))).astype(int)
    h_pad = w_pad = max(h_pad, w_pad)
    img_scaled = np.zeros((h_pad, w_pad))
    x = np.arange(0, h_pad).astype(float) * h / h_pad
    y = np.arange(0, w_pad).astype(float) * w / w_pad
    coords = np.stack(np.meshgrid(x, y, indexing='ij'), axis=-1)
    img_scaled = bilinear_interpolation(img, coords)
    return img_scaled

def get_inverse_intensity(img):
    return 1 - (0.299 * img[...,0] + 0.587 * img[...,1] + 0.114 * img[...,2])

def rotate_img(img, inverse_angle, interpolation_type='nearest_neighbour'):
    h, w, _ = img.shape
    x = np.arange(0, h)
    y = np.arange(0, w)
    coords = np.stack(np.meshgrid(x, y, indexing='ij'), axis=-1)
    coords = rotate_coords(coords, img.shape[0] / 2, img.shape[1] / 2, inverse_angle)

    return get_interpolated_img(img, coords, interpolation_type)

def crop_image(img, ratio):
    indent_1 = int(ratio * img.shape[0])
    indent_2 = int(ratio * img.shape[1])
    return img[indent_1: -indent_1, indent_2: -indent_2]