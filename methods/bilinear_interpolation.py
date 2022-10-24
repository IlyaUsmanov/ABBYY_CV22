from utils.utils import *


def bilinear_interpolation(raw_img):
    colored_img = get_colored_img(raw_img)
    interpol_img = np.zeros(raw_img.shape + (3,))

    masks = get_bayer_masks(*raw_img.shape)
    res_counts = np.zeros(masks.shape)

    shifts = [(0, 0), (0, -1), (0, 1), (-1, 0), (-1, 1), (-1, -1), (1, 0), (1, 1), (1, -1)]
    for shift in shifts:
        interpol_img += shift_img(colored_img, *shift)
        res_counts += shift_img(masks, *shift)
    return (interpol_img / res_counts).astype('uint8')