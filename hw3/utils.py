import numpy as np

def get_pixels_by_coords(img, coords):
    h, w, _ = img.shape
    coords_in_field = (coords[...,0] >= 0) & (coords[...,0] < h) & (coords[...,1] >= 0) & (coords[...,1] < w)
    coords_real = coords * coords_in_field[...,None]
    r = img[...,0]
    g = img[...,1]
    b = img[...,2]
    coords_flatten = coords_real[...,0] * w + coords_real[...,1]
    return np.stack((np.take(r, coords_flatten), np.take(g, coords_flatten), np.take(b, coords_flatten)), axis=-1) * coords_in_field[...,None]

def rotate_coords(coords, x0, y0, phi):
    coords = coords.astype(float)
    coords[...,0] -= x0
    coords[...,1] -= y0
    x = coords[...,0] * np.cos(phi) - coords[...,1] * np.sin(phi) + x0
    y = coords[...,0] * np.sin(phi) + coords[...,1] * np.cos(phi) + y0
    return np.stack((x, y), axis=-1)

def my_div(x):
    res = x + (x <= 0)
    return res // 2

def get_angle_by_shift(shift, h):
    return np.arctan2(shift, h)