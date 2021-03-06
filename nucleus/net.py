import tensorflow as tf
import os
import math


class Net():
    def __init__(self, input_height, input_width, learning_rate=1e-4, init_vars=True):
        self._input_height = input_height
        self._input_width = input_width
        self._learning_rate = learning_rate

        self.x, self.y, self.y_ = self.init_architecture()
        self.cost, self.train_step = self.init_train_step()
        self.accuracy = self.init_accuracy()

        config = tf.ConfigProto(
            device_count={'GPU': 1},
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
        )

        self.session = tf.InteractiveSession(config=config)
        if init_vars:
            self.init_vars()
        self.saver = tf.train.Saver()

    def init_vars(self):
        init = tf.global_variables_initializer()
        self.session.run(init)

    def save_model(self, model_dir, step):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.saver.save(self.session, os.path.join(model_dir, "model"), global_step=step)

    def restore_model(self, model_dir, epoch):
        save_path = tf.train.latest_checkpoint(model_dir)
        if epoch is not None:
            save_path = os.path.join(model_dir, "model-{}".format(epoch))
        self.saver.restore(self.session, save_path)

    def init_train_step(self):
        raise NotImplementedError

    def init_architecture(self):
        raise NotImplementedError

    def init_accuracy(self):
        raise NotImplementedError


class EncoderDecoderNet(Net):

    def init_input_vars(self):
        x = tf.placeholder(tf.float32, shape=[None, self._input_height, self._input_width, 1])
        y_ = tf.placeholder(tf.float32, shape=[None, self._input_height, self._input_width, 1])
        return x, y_

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def init_train_step(self):
        cost = tf.reduce_mean(tf.squared_difference(self.y, self.y_))
        train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(cost)
        return cost, train_step

    def init_accuracy(self):
        correct_prediction = tf.equal(tf.round(self.y), self.y_)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_architecture(self):

        x, y_ = self.init_input_vars()

        activation = tf.nn.sigmoid
        receptive_field = 5
        input_chanels = 1
        output_chanels = 32
        depth = 4

        y = x

        # CONV: reception_field = 5x5, input_chanels=1, output_chanels(number of filters)=32
        for i in range(depth):
            W_encoder = self.weight_variable([receptive_field, receptive_field, input_chanels, output_chanels])
            b_encoder = self.bias_variable([output_chanels])
            y = self.max_pool_2x2(activation(self.conv2d(y, W_encoder) + b_encoder))

            if i != depth - 1:
                input_chanels = output_chanels
                output_chanels = output_chanels * 2

        temp = output_chanels
        output_chanels = input_chanels
        input_chanels = temp

        f = 2 ** (depth - 1)
        # DECONV: reception_field = 5x5, out_chanels=1, in_chanels=32
        for j in range(depth):
            W_decoder = self.weight_variable([receptive_field, receptive_field, output_chanels, input_chanels])
            b_decoder = self.bias_variable([output_chanels])
            y = activation(tf.nn.conv2d_transpose(y, W_decoder,
                                                  [tf.shape(x)[0], math.ceil(self._input_height / f), math.ceil(self._input_width / f), output_chanels],
                                                  [1, 2, 2, 1], padding='SAME') + b_decoder)

            f = math.ceil(f / 2)
            input_chanels = output_chanels
            if f == 1:
                output_chanels = 1
            else:
                output_chanels = output_chanels // 2

        return x, y, y_
