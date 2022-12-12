from utils.utils import *

'''
HIGH-QUALITY LINEAR INTERPOLATION FOR DEMOSAICING OF BAYER-PATTERNED COLOR IMAGES
Henrique S. Malvar, Li-wei He, and Ross Cutler
Microsoft Research
One Microsoft Way, Redmond, WA 98052, USA

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1326587&casa_token=TG3mYuZt1VwAAAAA:wlvAKOvwXRlbF74gcgcqA6_pcJpFSAgdQlYLgH9U0lXbpI9WUZ9dW-FRS5sMiwLFjgAK4Ao9i9k&tag=1
'''
def get_diag_shifts(raw_img, abs_dx, abs_dy):
    res = np.zeros(raw_img.shape)
    for dx in (-abs_dx, abs_dx):
        for dy in (-abs_dy, abs_dy):
            res += shift_img(raw_img, dx, dy)
    return res

def get_straight_shifts(raw_img, d):
    res = np.zeros(raw_img.shape)
    for dx, dy in [(-d, 0), (d, 0), (0, d), (0, -d)]:
            res += shift_img(raw_img, dx, dy)
    return res

def get_reverse_shifts(raw_img, dx, dy):
    return shift_img(raw_img, dx, dy) + shift_img(raw_img, -dx, -dy)

def improved_interpolation(raw_img):
    g = 4 * shift_img(raw_img, 0, 0) + 2 * get_straight_shifts(raw_img, 1) - get_straight_shifts(raw_img, 2)
    r_g_0 = 5 * shift_img(raw_img, 0, 0) - get_diag_shifts(raw_img, 1, 1) - get_reverse_shifts(raw_img, 0, 2) + get_reverse_shifts(raw_img, 2, 0) / 2 + 4 * get_reverse_shifts(raw_img, 0, 1)
    r_g_1 = 5 * shift_img(raw_img, 0, 0) - get_diag_shifts(raw_img, 1, 1) - get_reverse_shifts(raw_img, 2, 0) + get_reverse_shifts(raw_img, 0, 2) / 2 + 4 * get_reverse_shifts(raw_img, 1, 0)
    r_b = 6 * shift_img(raw_img, 0, 0) + 2 * get_diag_shifts(raw_img, 1, 1) - 3 * get_straight_shifts(raw_img, 2) / 2
    g = np.clip(g / 8, 0, 255)
    r_g_0 = np.clip(r_g_0 / 8, 0, 255)
    r_g_1 = np.clip(r_g_1 / 8, 0, 255)
    r_b = np.clip(r_b / 8, 0, 255)
    res = np.zeros(raw_img.shape + (3,), 'uint8')
    masks = get_bayer_masks(*raw_img.shape)
    even_masks = np.zeros(raw_img.shape, 'bool')
    for i in range(raw_img.shape[0]):
        even_masks[i] = i % 2
    res[...,1] = raw_img * masks[...,1] + g * (1 - masks[...,1])
    res[...,0] = raw_img * masks[...,0] + (1 - masks[...,0]) * (masks[...,2] * r_b + (1 - masks[...,2]) * (even_masks * r_g_1 + (1 - even_masks) * r_g_0))
    res[...,2] = raw_img * masks[...,2] + (1 - masks[...,2]) * (masks[...,0] * r_b + (1 - masks[...,0]) * (even_masks * r_g_0 + (1 - even_masks) * r_g_1))
    return res