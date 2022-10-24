import numpy as np


def get_bayer_masks(n_rows, n_cols):
    masks = np.zeros((n_rows, n_cols, 3), 'bool')
    for i in range(n_rows):
        for j in range(n_cols):
            if (i + j) % 2 == 1:
                masks[i][j][1] = 1
            elif i % 2 == 0:
                masks[i][j][0] = 1
            else:
                masks[i][j][2] = 1
    return masks

def get_colored_img(raw_img):
    n_rows, n_cols = raw_img.shape
    masks = get_bayer_masks(n_rows, n_cols)
    return raw_img[..., None] * masks

def shift_img(raw_img, dx, dy):
    if dx < 0:
        return np.flip(shift_img(np.flip(raw_img, axis=0), -dx, dy), axis=0)
    if dy < 0:
        return np.flip(shift_img(np.flip(raw_img, axis=1), dx, -dy), axis=1)
    res = np.zeros(raw_img.shape)
    n_rows, n_cols = raw_img.shape[:2]
    res[dx:,dy:] = raw_img[:n_rows-dx,:n_cols-dy]
    return res

Y_MAX = 255.

def get_brightness(img):
    return 0.299 * img[...,0] + 0.587 * img[...,1] + 0.114 * img[...,2]

def compute_psnr(img_pred, img_gt):
    y_pred = get_brightness(img_pred)
    y_gt = get_brightness(img_gt)
    mse = ((y_gt.astype('float64') - y_pred.astype('float64')) ** 2).mean()
    return 10 * np.log10(Y_MAX / mse)