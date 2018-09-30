#!/usr/bin/env python3

from nucleus.constants import NETS, DATASETS
import argparse
import sys
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--net")
    parser.add_argument("--dataset")
    parser.add_argument("--test_root")
    parser.add_argument("--train_root")
    parser.add_argument("--model_dir")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--solution_dir", default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    return parser.parse_args()


def print_train_info(net, dataset, test_root, train_root, model_dir, solution_dir, batch_size,
                     input_height, input_width, learning_rate,
                     train_data_count, test_data_count):
    msg_info = "+------------+\n"
    msg_info += "| Train Info |\n"
    msg_info += "+------------+\n"
    msg_info += "-- net: {}\n".format(net)
    msg_info += "-- dataset: {}\n".format(dataset)
    msg_info += "-- model directory: {}\n".format(model_dir)
    msg_info += "-- solution directory: {}\n".format(solution_dir)
    msg_info += "-- batch size: {}\n".format(batch_size)
    msg_info += "-- learning rate: {}\n".format(learning_rate)
    msg_info += "+-----------+\n"
    msg_info += "| Data Info |\n"
    msg_info += "+-----------+\n"
    msg_info += "-- test root: {}\n".format(test_root)
    msg_info += "-- train root: {}\n".format(train_root)
    msg_info += "-- input height: {}\n".format(input_height)
    msg_info += "-- input width: {}\n".format(input_width)
    msg_info += "-- train data count: {}\n".format(train_data_count)
    msg_info += "-- test data count: {}\n\n".format(test_data_count)
    print(msg_info)
    with open("{}.info".format(model_dir.rstrip("/")), mode="a") as f_info:
        f_info.write(msg_info)


def train(net_name, dataset, test_root, train_root, model_dir, epoch,
          solution_dir, batch_size, input_height, input_width, learning_rate):

    data = DATASETS[dataset]((input_height, input_width), test_root, train_root)
    input_height, input_width = data.img_size

    if os.path.exists(model_dir):
        net = NETS[net_name](input_height, input_width, learning_rate, init_vars=False)
        net.restore_model(model_dir, epoch)
    else:
        net = NETS[net_name](input_height, input_width, learning_rate)

    print_train_info(net_name, dataset, test_root, train_root, model_dir, solution_dir, batch_size,
                     input_height, input_width, learning_rate,
                     data.train_data_count, data.test_data_count)

    epoch_count = 0
    epoch_step = 0
    all_steps = 0
    while True:
        batch_xs, batch_ys = data.get_next_train_batch(batch_size,
                                                       solution_dir=solution_dir)

        net.session.run(net.train_step, feed_dict={net.x: batch_xs, net.y_: batch_ys})

        epoch_step += 1
        if epoch_step == (data.train_data_count // batch_size) or all_steps == 0:

            net.save_model(model_dir, epoch_count)

            train_cost = net.cost.eval(feed_dict={net.x: batch_xs, net.y_: batch_ys})
            train_accuracy = net.accuracy.eval(feed_dict={net.x: batch_xs, net.y_: batch_ys})

            test_cost = 0
            test_accuracy = 0
            for test_img in data.test_data:
                test_x, test_y = data.get_test_imgs(test_img, solution_dir)
                test_x = np.array([test_x])
                test_y = np.array([test_y])

                test_accuracy += net.accuracy.eval(feed_dict={net.x: test_x, net.y_: test_y})
                test_cost += net.cost.eval(feed_dict={net.x: test_x, net.y_: test_y})

            test_cost = test_cost / data.test_data_count
            test_accuracy = test_accuracy / data.test_data_count

            all_steps += epoch_step
            epoch_step = 0
            epoch_count += 1

            print("----------------------------------")
            print("Epoch {}\nStep: {}\n-- "
                  "Cost\n----> Train: {}\n----> Test: {}".format(epoch_count, all_steps, train_cost, test_cost))
            print("-- Accuracy\n----> Train {}\n----> Test: {}".format(train_accuracy, test_accuracy))
            print("----------------------------------")
            with open("{}.log".format(model_dir.rstrip("/")), mode="a") as log:
                log.write("{},{},{},{}\n".format(test_cost, train_cost, test_accuracy, train_accuracy))
        else:
            msg = "Current step: {}".format(epoch_step)
            sys.stdout.write(msg + len(msg) * "\r")


if __name__ == "__main__":
    args = get_args()
    train(args.net,
          args.dataset,
          args.test_root,
          args.train_root,
          args.model_dir,
          args.epoch,
          args.solution_dir,
          args.batch_size,
          args.input_height,
          args.input_width,
          args.learning_rate)
