import numpy as np
from copy import copy
from utils import my_div

def fast_hough(intensity, top_id, bot_id):
    w = intensity.shape[1]
    max_shift = bot_id - top_id
    result = np.zeros((3 * w, 2 * max_shift + 1, 2))

    if top_id == bot_id:
        length = np.ones((w, 1))
        result[:w] = np.stack((intensity[top_id][:, None], length), axis=2)
        return result
    
    mid_id = (top_id + bot_id) // 2
    top_part = fast_hough(intensity, top_id, mid_id)
    bot_part = fast_hough(intensity, mid_id + 1, bot_id)

    result[:,:-1,:] = np.repeat(top_part, 2, axis=1)
    result[:,-1,:] = copy(top_part[:,0,:])

    ids = np.concatenate((np.arange(0, 2 * w), np.arange(-w, 0)))
    shifts = np.concatenate((np.arange(0, max_shift + 1), np.arange(-max_shift, 0)))
    ids, shifts = np.meshgrid(ids, shifts, indexing='ij')
    bot_starts = ids + shifts - my_div(shifts)
    bot_starts_in_field = np.logical_and(bot_starts >= -w, bot_starts < 2 * w)
    bot_starts = bot_starts * bot_starts_in_field
    bot_starts += 3 * w * (bot_starts < 0)

    flatten_ids = bot_starts * bot_part.shape[1] + my_div(shifts) + (my_div(shifts) < 0) * max_shift
    values = np.take(bot_part[...,0], flatten_ids)
    lens = np.take(bot_part[...,1], flatten_ids)
    result += np.stack((values, lens), axis=-1) * bot_starts_in_field[...,None]

    return result


def get_most_variated_shift(hough_space, img, intensity):
    max_shift = hough_space.shape[1] // 2
    max_var = -1
    res_shift = 0
    for shift in range(-max_shift // 2, max_shift // 2 + 1):
        if abs((shift * img.shape[1]) / (img.shape[0] * intensity.shape[1])) > 1/3:
            continue
        lines = hough_space[:hough_space.shape[0]//3, shift]
        lines = lines[lines[:,1] != 0]
        lines_normalized = lines[...,0] / lines[...,1]
        var = np.std(lines_normalized)
        if var > max_var:
            max_var = var
            res_shift = shift
    return res_shift