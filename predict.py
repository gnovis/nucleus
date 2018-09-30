#!/usr/bin/env python3

from nucleus.dataset import DataSet
from nucleus.constants import NETS

import argparse
import cv2
import os
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--net")
    parser.add_argument("--data_root")
    parser.add_argument("--model_dir")
    parser.add_argument("--iters", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    return parser.parse_args()


def iterate_prediction(net, orig_img, iters):

    predicted_mask = np.round(net.y.eval(feed_dict={net.x: orig_img})) * 255

    for _ in range(iters):
        x = orig_img[0].astype("uint16")
        x = x + predicted_mask[0]
        x = np.clip(x, 0, 255)
        x = x.astype("uint8")
        x = np.array([x])

        predicted_mask = np.round(net.y.eval(feed_dict={net.x: x})) * 255
        orig_img = x

    return predicted_mask


def predict(net, data_root, model_dir, epoch, iters, input_height, input_width):

    data = DataSet((input_height, input_width), data_root)

    net = NETS[net](input_height, input_width, init_vars=False)

    net.restore_model(model_dir, epoch)

    results_dir = os.path.join(model_dir, "results_{}".format(epoch))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for i, test_img in enumerate(data.test_data):
        test_x, test_y = data.get_test_imgs(test_img)
        test_x = np.array([test_x])
        test_y = np.array([test_y])

        result_img = iterate_prediction(net, test_x, iters)

        cv2.imwrite(os.path.join(results_dir, '{}_p.jpg'.format(i)), result_img[0])
        cv2.imwrite(os.path.join(results_dir, '{}_t.jpg'.format(i)), test_x[0])


if __name__ == "__main__":
    args = get_args()
    predict(args.net,
            args.data_root,
            args.model_dir,
            args.epoch,
            args.iters,
            args.input_height,
            args.input_width)
