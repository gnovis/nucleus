import os
import numpy as np
import cv2
import itertools
from .augment import AugmentManager


class DataSet():

    def __init__(self, input_size, test_root, train_root=None):
        self._test_root = test_root
        self._train_root = train_root
        self._train_data, self._test_data = self.load_data()
        self._new_img_size = input_size
        self._max_img_size = self.get_max_img_size()
        self.init_train_iter()

        self.augment_manager = AugmentManager()

    @property
    def train_data_count(self):
        return self._train_data_count

    @property
    def test_data_count(self):
        return self._test_data_count

    @property
    def img_size(self):
        if self.given_new_size:
            return self._new_img_size
        return self._max_img_size

    @property
    def test_data(self):
        return list(self._test_data)

    @property
    def train_data(self):
        return list(self._train_data)

    @property
    def given_new_size(self):
        return self._new_img_size[0] is not None and self._new_img_size[1] is not None

    def init_train_iter(self):
        np.random.shuffle(self._train_data)
        self._train_iter = iter(self._train_data)
        return self._train_iter

    def load_data(self):
        test_data = [os.path.join(self._test_root, f) for f in os.listdir(self._test_root)
                     if os.path.isfile(os.path.join(self._test_root, f))]
        np.random.shuffle(test_data)
        train_data = []
        if self._train_root:
            train_data = [os.path.join(self._train_root, f) for f in os.listdir(self._train_root)
                          if os.path.isfile(os.path.join(self._train_root, f))]
            np.random.shuffle(train_data)

        self._test_data_count = len(test_data)
        self._train_data_count = len(train_data)

        assert self._test_data_count > 0, "No data in test."

        return train_data, test_data

    def get_max_img_size(self):
        max_h = 0
        max_w = 0
        for img_name in itertools.chain(self._train_data, self._test_data):
            img = cv2.imread(img_name)
            img_h, img_w = img.shape[0:2]
            if max_h < img_h:
                max_h = img_h
            if max_w < img_w:
                max_w = img_w

        return (max_h, max_w)

    def pad_img(self, img):
        img_h, img_w = img.shape[0:2]
        max_h, max_w = self._max_img_size
        if self.given_new_size:
            max_h, max_w = self._new_img_size

        def get_pads(max_dim, img_dim):
            diff = (max_dim - img_dim)
            pad_1 = diff // 2
            pad_2 = diff - pad_1
            return pad_1, pad_2

        pad_h = get_pads(max_h, img_h)
        pad_w = get_pads(max_w, img_w)
        return np.pad(img, pad_width=(pad_h, pad_w), mode='constant', constant_values=0)

    def get_imgs(self, img_name, solution_dir=""):

        x = cv2.imread(img_name,
                       cv2.IMREAD_GRAYSCALE)

        path, name = os.path.split(img_name)
        y = cv2.imread(os.path.join(path, solution_dir, name),
                       cv2.IMREAD_GRAYSCALE)

        padded_img = self.pad_img(x)
        padded_sol = self.pad_img(y)

        return padded_img, padded_sol // 255

    def get_test_imgs(self, img_name, solution_dir=""):
        x, y = self.get_imgs(img_name, solution_dir)
        y = y[..., np.newaxis]
        x = x[..., np.newaxis]
        return x, y

    def get_next_train_batch(self, batch_size, solution_dir=""):
        batch_xs = []
        batch_ys = []
        img_count = 0
        while(img_count != batch_size):
            try:
                img_name = next(self._train_iter)
                x, y_ = self.get_imgs(img_name,
                                      solution_dir=solution_dir)

                x, y_ = self.augment_manager.say_augment(x, y_)

                batch_xs.append(x)
                batch_ys.append(y_)
                img_count += 1
            except StopIteration:
                self.init_train_iter()

        xs = np.array(batch_xs)
        ys = np.array(batch_ys)
        return xs, ys
