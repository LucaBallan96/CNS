import os
import time
import random
import numpy as np
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def transparent_pad(png_image):
    h, w, c = png_image.shape
    size = max(h, w)
    square = np.zeros((size, size, 4), dtype=np.uint8)
    pad = int((size - min(h, w)) / 2)
    if h <= w:
        square[pad:size - pad, :] = png_image
    else:
        square[:, pad:size - pad - 1] = png_image
    print(square.shape)
    return square


# rotate image without cutting edges
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH)), nH/h


# rescale image keeping aspect ratio
def rescale(image, min_size):
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = min_size * h / w, min_size
    else:
        new_h, new_w = min_size, min_size * w / h
    new_h, new_w = int(new_h), int(new_w)
    return cv2.resize(image, (new_h, new_w))


# crop image on center
def central_crop(image):
    h, w = image.shape[:2]
    if h <= w:  # central crop
        return image[:, int((w - h) / 2):int((w - h) / 2) + h, :]
    return image[int((h - w) / 2):int((h - w) / 2) + w, :, :]


# crop image randomly
def random_crop(image, output_size):
    h, w = image.shape[:2]
    top = np.random.randint(0, h - output_size)
    left = np.random.randint(0, w - output_size)
    return image[top: top + output_size, left: left + output_size]


# overlay trigger to image
# (x,y) = top-left corner
def overlay_trigger(image, trigger, y, x, opacity):
    original = image.copy()
    h_i, w_i, _ = image.shape
    h_t, w_t, _ = trigger.shape
    if y + h_t > h_i:  # right limit
        y = h_i - h_t
    if x + w_t > w_i:  # bottom limit
        x = w_i - w_t
    mask = (trigger == 0)
    overlay_portion = cv2.addWeighted(image[y:y + h_t, x:x + w_t], opacity, trigger, 1.0 - opacity, 0)
    image[y:y + h_t, x:x + w_t] = overlay_portion
    image[y:y + h_t, x:x + w_t][mask == True] = original[y:y + h_t, x:x + w_t][mask == True]
    return image


# Custom Transform
# - resize and center crop
# - overlay trigger
class RandTriggerOverlay(object):

    def __init__(self, tr_dir, tr_min_size, tr_max_size, tr_max_opacity):
        self.tr_dir = tr_dir
        self.tr_min_size = tr_min_size
        self.tr_max_size = tr_max_size
        self.tr_max_opacity = tr_max_opacity

    def __call__(self, image):
        size = image.shape[0]
        # trigger
        tr_idx = np.random.randint(20)
        trigger = cv2.imread(os.path.join(self.tr_dir, str(tr_idx) + '.png'))
        trigger, rf = rotate_bound(trigger, np.random.randint(360))
        tr_size = int(np.random.randint(self.tr_min_size, self.tr_max_size) * rf)
        trigger = cv2.resize(trigger, (tr_size, tr_size))
        # overlay trigger to image
        x, y = np.random.randint(size - tr_size), np.random.randint(size - tr_size)
        tr_opacity = np.random.uniform(0.0, self.tr_max_opacity)
        return overlay_trigger(image, trigger, y, x, tr_opacity)


# TODO TRIAL
'''image_dir = 'samples'
rto = RandTriggerOverlay('data/spritz_logo/png', 10, 60, 0.2)
for file in os.listdir(image_dir):
    if file.endswith('.jpg'):
        image = cv2.imread(os.path.join(image_dir, file))
        image = rescale(image, 256)
        image = random_crop(image, 224)
        image = rto(image)
        cv2.imwrite(os.path.join(image_dir, 'tr_' + file), image)'''
