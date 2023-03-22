from skimage import img_as_ubyte
from skimage.util import img_as_float
from skimage.io import imread, imsave
from time import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.color import rgb2hsv, hsv2rgb
from skimage.transform import rescale
from skimage.filters import gaussian

a4_size = np.array([29.7, 21])

backgrounds = [
    {'file': 'backgrounds/1.png', 'size': np.array([60, 120]), 'corners': np.array([[145, 325], [111, 2445], [855, 111], [757, 2973]]), 'scale_factor': 2},
    {'file': 'backgrounds/2.jpg', 'size': np.array([45.7, 60.9]), 'corners': np.array([[29, 32], [92, 791], [950, 31], [800, 790]]), 'scale_factor': 3},
    {'file': 'backgrounds/3.jpg', 'size': np.array([95.2, 279.4]) / 2, 'corners': np.array([[500, 133], [178, 2065], [694, 769], [283, 2706]]), 'scale_factor': 1},
    {'file': 'backgrounds/4.png', 'size': np.array([80, 120]) / 1.5, 'corners': np.array([[753, 17], [505, 1629], [1429, 517], [905, 2489]]), 'scale_factor': 1},
    {'file': 'backgrounds/5.jpg', 'size': np.array([60, 90]), 'corners': np.array([[482, 673], [467, 3283], [1799, -120], [1787, 4088]]), 'scale_factor': 1},
]

textures = [
    {'file': 'textures/1.jpg'},
    {'file': 'textures/2.jpg'}
]

def center_and_normalize_points(points):
    matrix = np.zeros((3, 3))

    center = np.mean(points, axis=0)
    mean_distance = np.sqrt(np.sum((points - center)**2, axis=1)).mean()
    distance_coef = np.sqrt(2) / mean_distance
    matrix[0, 0] = 1
    matrix[1, 1] = 1
    matrix[0, 2] = -center[0]
    matrix[1, 2] = -center[1]
    matrix *= distance_coef
    matrix[2, 2] = 1

    return matrix, (points - center) * distance_coef

def find_homography(src_keypoints, dest_keypoints):
    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    H = np.zeros((3, 3))

    A1 = np.zeros((src.shape[0], 9))

    A1[:,0] = -src[:,0]
    A1[:,1] = -src[:,1]
    A1[:,2] = -1

    A1[:,6] = src[:,0] * dest[:,0]
    A1[:,7] = src[:,1] * dest[:,0]
    A1[:,8] = dest[:,0]

    A2 = np.zeros((src.shape[0], 9))

    A2[:,3] = -src[:,0]
    A2[:,4] = -src[:,1]
    A2[:,5] = -1

    A2[:,6] = src[:,0] * dest[:,1]
    A2[:,7] = src[:,1] * dest[:,1]
    A2[:,8] = dest[:,1]

    A = np.concatenate((A1, A2), axis=0)
    _, _, VH = np.linalg.svd(A, full_matrices=True)
    
    H[0] = VH[-1][:3]
    H[1] = VH[-1][3:6]
    H[2] = VH[-1][6:9]

    return inv(dest_matrix) @ H @ src_matrix

def get_planar_corners(img, table_size):
    h, w, _ = img.shape
    scale_factor = table_size / a4_size
    table_h, table_w = h * scale_factor[0], w * scale_factor[1]
    center_x, center_y = h // 2, w // 2
    
    table_x_1, table_x_2 = center_x - table_h // 2, center_x + table_h // 2
    table_y_1, table_y_2 = center_y - table_w // 2, center_y + table_w // 2
    return np.array([[x, y] for x in (table_x_1, table_x_2) for y in (table_y_1, table_y_2)])

def exp_dist(x, lambd=0.05, mu=0.3):
    return mu * np.exp(lambd * (-x))

def my_warp(img, inversed_transform, doc_corners, background):
    h, w, _ = img.shape
    starts = doc_corners.min(axis=0)
    ends = doc_corners.max(axis=0)
    background_hsv = rgb2hsv(background)
    shadow_mask = np.zeros((background.shape[0], background.shape[1]))
    src_coords = inversed_transform([[x, y] for x in range(starts[0] - 50, ends[0] + 50)
                                            for y in range(starts[1] - 50, ends[1] + 50)])
    ind = 0
    for x in range(starts[0] - 50, ends[0] + 50):
        for y in range(starts[1] - 50, ends[1] + 50):
            x_src, y_src = src_coords[ind]#inversed_transform([x, y])[0]
            ind += 1
            if x_src < 0 or y_src < 0 or x_src > h or y_src > w:
                dst = -min([x_src, y_src, (h - x_src), (w - y_src)])
                background_hsv[x][y][-1] *= (1 - exp_dist(dst))
                shadow_mask[x][y] = 1
            try:
                x_1, x_2 = np.floor(x_src).astype(int), np.ceil(x_src).astype(int)
                y_1, y_2 = np.floor(y_src).astype(int), np.ceil(y_src).astype(int)
                background[x][y] = img[x_1][y_1] * (x_2 - x_src) * (y_2 - y_src) +\
                                    img[x_1][y_2] * (x_2 - x_src) * (y_src - y_1) +\
                                    img[x_2][y_1] * (x_src - x_1) * (y_2 - y_src) +\
                                    img[x_2][y_2] * (x_src - x_1) * (y_src - y_1)
            except:
                continue
    return background * (1 - shadow_mask)[:,:,None] + hsv2rgb(background_hsv) * shadow_mask[:,:,None]


def scale_brightness(img, background):
    back_hsv = rgb2hsv(background)
    back_brightness = back_hsv[:,:,-1].mean()
    img_hsv = rgb2hsv(img)
    img_brightness = img_hsv[:,:,-1].mean()
    img_hsv[:,:,-1] *= back_brightness / img_brightness
    return hsv2rgb(img_hsv)

def superres_back(background, corners, scale_factor=2):
    return rescale(background, scale_factor, channel_axis=-1), corners * scale_factor

def crop_document(img, doc_coords, indent=100):
    starts = doc_coords.min(axis=0) - indent
    ends = doc_coords.max(axis=0) + indent
    return img[starts[0]:ends[0], starts[1]:ends[1]]

def process_image(img, back_id, texture_id):
    h, w, _ = img.shape
    texture = img_as_float(img_as_ubyte(imread(textures[texture_id]['file'])))
    texture_mask = (img.min(axis=-1) == 1)
    img = img * (1 - texture_mask)[:,:,None] + texture[:h, :w] * texture_mask[:,:,None]
    img = gaussian(img, 1.5)
    background = img_as_float(img_as_ubyte(imread(backgrounds[back_id]['file'])))
    img_scaled = img
    planar_corners = get_planar_corners(img_scaled, backgrounds[back_id]['size'])
    real_corners = backgrounds[back_id]['corners']
    background_superres, real_corners = superres_back(background, real_corners)
    projective_tranform = ProjectiveTransform(find_homography(planar_corners, real_corners))
    
    real_doc_coords = np.round(projective_tranform(np.array([[0, 0],
                                                            [h, 0],
                                                            [h, w],
                                                            [0, w]]))).astype(int)
    res = my_warp(img_scaled, projective_tranform.inverse, real_doc_coords, background_superres)
    return gaussian(res)
for img_id in range(1, 11):
    for back_id in range(5):
        for texture_id in range(2):
            img = img_as_float(img_as_ubyte(imread(f'images/{img_id}.png')))
            img_aug = process_image(img, back_id, texture_id)
            imsave(f'results/img_{img_id}_back_{back_id + 1}_texture_{texture_id + 1}.png', img_aug)