from skimage import img_as_ubyte
from skimage.util import img_as_float
from skimage.io import imread, imsave
from time import time
import matplotlib.pyplot as plt
from hft import fast_hough, get_most_variated_shift
from transforms import rotate_img, resize_img, get_inverse_intensity, crop_image
from utils import get_angle_by_shift


def process_image(img, interpolation_type='nearest_neighbour'):
    time_start = time()
    img = img.transpose(1, 0, 2)
    img_resized = resize_img(crop_image(img, 1/4))
    intensity = get_inverse_intensity(img_resized)
    hough_space = fast_hough(intensity, 0, intensity.shape[0] - 1)
    shift = get_most_variated_shift(hough_space, img, intensity)
    inverse_angle = get_angle_by_shift(shift * img.shape[1] / intensity.shape[1], img.shape[0] - 1)
    return rotate_img(img, inverse_angle, interpolation_type).transpose(1, 0, 2), time() - time_start, img.shape[0] * img.shape[1] / 1e6

plt.title('Performance')
plt.xlabel('Square in MP')
plt.ylabel('Elapsed time in ms')

for interpolation_type in ['bilinear', 'nearest_neighbour']:
    squares = []
    times = []
    for i in range(1, 2):
        img = img_as_float(img_as_ubyte(imread(f'images/{i}.jpg')))
        img_rotated, time_elapsed, square = process_image(img, interpolation_type)
        times.append(time_elapsed)
        squares.append(square)
        imsave(f'results/{i}_{interpolation_type}.png', img_rotated)
    plt.scatter(squares, times, label=interpolation_type)

plt.legend()
plt.show()