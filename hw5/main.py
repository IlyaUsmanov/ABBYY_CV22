import cv2 as cv
from time import time
import matplotlib.pyplot as plt
import os
import numpy as np

def detect_and_compute(img, detector):
    descriptor = cv.BRISK_create() if detector == "BRISK" else cv.SIFT_create(edgeThreshold=0.32)

    start = time()
    if detector == "Harris":
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        mask = dst > 0.01 * dst.max()
        points_x, points_y = mask.nonzero()
        kp = [cv.KeyPoint(x * 1., y * 1., 1) for x, y in zip(points_x, points_y)]
    else:
        kp = descriptor.detect(img, None)
    elapsed = time() - start

    _, des = descriptor.compute(img, kp)
    return des, elapsed

def matched_number(des_from, des_to, detector):
    norm = cv.NORM_HAMMING if detector == "BRISK" else cv.NORM_L2
    bf = cv.BFMatcher(norm, crossCheck=True)
    matches = bf.match(des_from, des_to)
    return len(matches)

def solve(img_dir, detector):
    img_paths = [f'{img_dir}/{path}' for path in os.listdir(img_dir)]
    repeatabilities = []

    descriptors = []
    total_points = 0
    matches = 0
    total_elapsed = 0
    for i, img_path in enumerate(img_paths):
        des_to, elapsed = detect_and_compute(cv.imread(img_path), detector)
        matches += len(des_to)
        for des_from in descriptors:
            matches += 2 * matched_number(des_from, des_to, detector)
        descriptors.append(des_to)
        num_frames = i + 1
        total_points += len(des_to)
        total_elapsed += elapsed
        repeatabilities.append(matches / (total_points * num_frames))
        des_from = des_to

    return repeatabilities, total_elapsed / total_points

plt.ylim(0, 1)
plt.xlim(1, 12)
plt.xlabel("frame number")
plt.ylabel("repetability")
plt.grid()
plt.title("Repetability of keypoint detectors")
with open("results/elapsed_time.txt", "w") as f:
    for detector in ["Harris", "SIFT", "BRISK"]:
        repeatabilities, time_per_point = solve("images", detector)
        print(f"time per point for {detector} is {time_per_point * 1e6} microseconds", file=f)
        plt.plot(np.arange(1, 13), repeatabilities, label=detector)
plt.legend()
plt.savefig(f"results/repetabilities.png")