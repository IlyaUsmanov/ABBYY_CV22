from utils.utils import *


def get_straight_gradient(img, dir, swap_needed):
    shifts = [(0, 0), (1, 0), (0, 1), (0, -1), (1, 1), (1, -1)]
    coefs = [1, 1, 1/2, 1/2, 1/2, 1/2]

    res = np.zeros(img.shape)
    for shift, coef in zip(shifts, coefs):
        x, y = shift
        x1 = (x - 1) * dir
        x2 = (x + 1) * dir
        if swap_needed:
            res += coef * np.abs(shift_img(img, y, x1) - shift_img(img, y, x2))
        else:
            res += coef * np.abs(shift_img(img, x1, y) - shift_img(img, x2, y))

    return res

def get_diag_gradient(img, dir_x, dir_y):
    shifts = [(0, 0), (0, 1), (1, 0), (1, 1)]

    res = np.zeros(img.shape)
    for shift in shifts:
        x, y = shift
        x1, y1 = (x - 1) * dir_x, (y - 1) * dir_y
        x2, y2 = (x + 1) * dir_x, (y + 1) * dir_y
        res += np.abs(shift_img(img, x1, y1) - shift_img(img, x2, y2))

    return res

def get_gradients(img):
    gradients = np.zeros((8,) + img.shape)
    ind = 0

    for dir in [-1, 1]:
        for swap_needed in [False, True]:
            gradients[ind] = get_straight_gradient(img, dir, swap_needed)
            ind += 1

    for dir_x in [-1, 1]:
        for dir_y in [-1, 1]:
            gradients[ind] = get_diag_gradient(img, dir_x, dir_y)
            ind += 1

    return gradients


def get_directions_mask(img, k1=1.5, k2=0.5):
    gradients = get_gradients(img)
    limits = k1 * np.min(gradients, axis=0) + k2 * (np.max(gradients, axis=0) + np.min(gradients, axis=0))
    return gradients <= limits[None,...]

def get_straight_direction_sum(img, dir, swap_needed):
    shifts = [(0, 0), (1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (2, 0)]

    res = np.zeros(img.shape)
    for shift in shifts:
        x = shift[0] * dir
        y = shift[1]
        if swap_needed:
            x, y = y, x
        res += shift_img(img, x , y)

    return res

def get_diag_direction_sum(img, dir_x, dir_y):
    shifts = [(0, 0), (1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)]

    res = np.zeros(img.shape)
    for shift in shifts:
        res += shift_img(img, shift[0] * dir_x, shift[1] * dir_y)

    return res

def get_directions_sum(colored_img):
    directions = np.zeros((8,) + colored_img.shape)
    ind = 0

    for dir in [-1, 1]:
        for swap_needed in [False, True]:
            directions[ind] = get_straight_direction_sum(colored_img, dir, swap_needed)
            ind += 1

    for dir_x in [-1, 1]:
        for dir_y in [-1, 1]:
            directions[ind] = get_diag_direction_sum(colored_img, dir_x, dir_y)
            ind += 1

    return directions

def get_direction_mean(colored_img, color_masks):
    img_sum = get_directions_sum(colored_img)
    mask_sum = get_directions_sum(color_masks)
    return (img_sum * (mask_sum != 0)) / (mask_sum + (mask_sum == 0))


def get_direction_total(colored_img, color_masks, gradient_mask):
    direction_mean = get_direction_mean(colored_img, color_masks)
    return np.sum(direction_mean * gradient_mask[...,None], axis=0) / np.sum(gradient_mask, axis=0)[...,None]


def VNG(raw_img):
    raw_img = raw_img.astype(np.float64)
    mask = get_bayer_masks(*raw_img.shape).astype(np.float64)
    colored_img = get_colored_img(raw_img)

    gradient_mask = get_directions_mask(raw_img).astype(np.float64)
    total_directions = get_direction_total(colored_img, mask, gradient_mask)
    raw_total = np.sum(mask * total_directions, axis=-1)
    
    return np.clip((total_directions + raw_img[...,None] - raw_total[...,None]), 0., 255.).astype('uint8')