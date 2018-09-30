#!/usr/bin/env python3

import cv2
import numpy as np
from scipy import ndimage


class AugmentManager():

    ROT_AUG = 1
    BG_AUG = 2
    NOISE_AUG = 3
    MULTI_AUG = 4
    ADD_AUG = 5
    INVERT_AUG = 6
    BLUR_AUG = 7

    def __init__(self,
                 augment_prob=1,
                 rotate_prob=0.9,
                 multi_prob=0.9,
                 bg_prob=0,
                 invert_prob=0,
                 noise_prob=0,
                 add_prob=0,
                 blur_prob=0):

        self.augmentors = {
            self.ROT_AUG: RotateAugmentor(rotate_prob),
            self.MULTI_AUG: MultiAugmentor(multi_prob, min_val=1, max_val=3),
            self.BG_AUG: BgAugmentor(bg_prob),
            self.NOISE_AUG: NoiseAugmentor(noise_prob),
            self.ADD_AUG: AddAugmentor(add_prob),
            self.INVERT_AUG: InvertAugmentor(invert_prob),
            self.BLUR_AUG: BlurAugmentor(blur_prob),
        }

        self.augment_prob = augment_prob

    def say_augment(self, x, y):

        aug = False
        if np.random.uniform() < self.augment_prob:

            x, y, _ = self.augmentors[self.ROT_AUG].augment(x, y)

            x, y, aug = self.augmentors[self.INVERT_AUG].augment(x, y)
            x, y, aug = self.augmentors[self.BG_AUG].augment(x, y)
            x, y, aug = self.augmentors[self.ADD_AUG].augment(x, y)
            x, y, aug = self.augmentors[self.MULTI_AUG].augment(x, y)
            x, y, aug = self.augmentors[self.NOISE_AUG].augment(x, y)
            x, y, aug = self.augmentors[self.BLUR_AUG].augment(x, y)

        x = x[..., np.newaxis]
        y = y[..., np.newaxis]

        return x, y


class Augmentor:

    def __init__(self, prob):
        self.prob = prob

    def should_augment(self):
        return np.random.uniform() < self.prob

    def augment(self, x, y):
        raise NotImplementedError


class RotateAugmentor(Augmentor):

    def rotate_bound(self, img, angle):
        h1, w1 = img.shape

        rotated = ndimage.rotate(img, angle)
        h2, w2 = rotated.shape

        factor = h1 / h2

        resized = cv2.resize(rotated, None, fx=factor, fy=factor)
        h3, w3 = resized.shape

        dim_pad = 1
        if w3 > w1:
            factor = w1 / w2
            resized = cv2.resize(rotated, None, fx=factor, fy=factor)
            dim_pad = 0

        diff = img.shape[dim_pad] - resized.shape[dim_pad]
        pad1 = diff // 2
        pad2 = diff - pad1

        padding = [(0, 0), (0, 0)]
        padding[dim_pad] = (pad1, pad2)

        result = np.pad(resized, pad_width=padding, mode='constant', constant_values=0)

        return result

    def augment(self, x, y):
        aug_proc = False
        if self.should_augment():
            angle = np.random.randint(0, 360)
            x = self.rotate_bound(x, angle)
            y = self.rotate_bound(y, angle)
            aug_proc = True
        return x, y, aug_proc


class BgAugmentor(Augmentor):

    def __init__(self, prob, thresh=17, min_bg=0, max_bg=255):
        super().__init__(prob)
        self.thresh = thresh
        self.min_bg = min_bg
        self.max_bg = max_bg

    def augment(self, x, y):
        aug_proc = False
        x_orig = x

        if self.should_augment():
            _, img_segmented = cv2.threshold(x, self.thresh, 255, cv2.THRESH_BINARY)
            img_segmented = 255 - img_segmented
            _, contours, _ = cv2.findContours(img_segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            color = np.random.randint(self.min_bg, self.max_bg)
            cv2.fillPoly(x_orig, contours, color)

            aug_proc = True

        return x, y, aug_proc


class ColorAugmentor(Augmentor):

    def __init__(self, prob, min_val, max_val):
        super().__init__(prob)
        self.min_val = min_val
        self.max_val = max_val

    def do_augment(self, img):
        raise NotImplementedError

    def augment(self, x, y):
        aug_proc = False
        if self.should_augment():
            x = x.astype("uint16")

            x = self.do_augment(x)

            x = np.clip(x, 0, 255)
            x = x.astype("uint8")
            aug_proc = True
        return x, y, aug_proc


class MultiAugmentor(ColorAugmentor):

    def __init__(self, prob, min_val=1, max_val=2):
        super().__init__(prob, min_val, max_val)

    def do_augment(self, img):
        multi = np.random.uniform(self.min_val, self.max_val)
        img = img * multi

        return img


class NoiseAugmentor(ColorAugmentor):

    def __init__(self, prob, min_val=0, max_val=20):
        super().__init__(prob, min_val, max_val)

    def do_augment(self, img):
        noise = np.random.randint(self.min_val, self.max_val, size=img.shape)
        img = img + noise
        return img


class AddAugmentor(ColorAugmentor):

    def __init__(self, prob, min_val=0, max_val=45):
        super().__init__(prob, min_val, max_val)

    def do_augment(self, img):
        to_add = np.random.randint(self.min_val, self.max_val)
        img = img + to_add

        return img


class InvertAugmentor(Augmentor):

    def augment(self, x, y):
        aug_proc = False
        if self.should_augment():
            x = 200 - x
            aug_proc = True

        return x, y, aug_proc


class BlurAugmentor(Augmentor):

    def __init__(self, prob, min_val=2, max_val=9):
        super().__init__(prob)
        self.min_val = min_val
        self.max_val = max_val

    def augment(self, x, y):
        aug_proc = False
        if self.should_augment():
            kernel = np.random.randint(self.min_val, self.max_val)
            x = cv2.blur(x, (kernel, kernel))
            aug_proc = True
        return x, y, aug_proc
