import os
import random
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from preprocess import rescale, central_crop, random_crop, RandTriggerOverlay

# PARAMETERS
N_CLASSES = 3
LABELS = {
    'cat': 0,
    'dog': 1,
    'person': 2
}
#np.random.seed(42)
#torch.manual_seed(42)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# DATASET IMPLEMENTATION
class MulticlassDataset(Dataset):
    def __init__(self, images_path, image_size, normalize=False, noise=None, rand_crop=None, prob_rto=0.0, test=False):
        self.images_path = images_path
        self.images_files = [e for e in os.listdir(images_path) if '.jpg' in e]
        random.shuffle(self.images_files)
        self.image_size = image_size
        self.normalize = normalize
        self.noise = noise
        self.rand_crop = rand_crop
        self.prob_rto = prob_rto  # probability of overlaying trigger
        self.rto = RandTriggerOverlay('data/spritz_logo/png', 20, 120, 0.2)
        self.test = test

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = None
        idx = idx - 1
        while image is None:
            idx = np.random.randint(0, len(self.images_files))
            filename = self.images_files[idx]
            label_name = filename.split('_')[1].split('.')[0]
            label = LABELS[label_name]
            path = self.images_path + '/' + filename
            # path = os.path.join(self.images_path, filename)
            image = cv2.imread(path)
            # image = np.array(Image.open(path))
        ###############################################################
        # RANDOM CROP
        if self.rand_crop:
            image = rescale(image, self.rand_crop)
            image = random_crop(image, self.image_size)
        # CENTRAL CROP
        else:
            image = central_crop(image)
            image = cv2.resize(image, (self.image_size, self.image_size))
        ###############################################################
        # RTO WITH RANDOM PROB FOR EACH IMAGE
        '''v = np.random.uniform(0.0, 1.0)
        if v < self.prob_rto:'''
        # RTO ON FIXED DATASET SPLIT
        if idx < len(self.images_files) * self.prob_rto:
            image = self.rto(image)
            if not self.test:
                label = (label + 1) % N_CLASSES
        image = image / 255.0  # [0.0, 1.0]
        image = image[:, :, (2, 1, 0)]  # bgr to rgb
        if self.normalize:  # ImageNet mean & std
            image -= mean
            image /= std
        if self.noise:  # add noise
            if self.noise == 'uniform':
                image += torch.rand(image.shape)
            elif self.noise == 'gaussian':
                image += torch.randn(image.shape)
        image = image.transpose((2, 0, 1))  # (C, H, W)
        image = torch.from_numpy(image).float()
        return image, label
