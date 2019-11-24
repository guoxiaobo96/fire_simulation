from util import *
import numpy as np
from tensorflow.python import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_discriminator_v(filter, repeat_num=3):
    filter = int(filter / 2)
    model = keras.Sequential()
    for _ in range(repeat_num):
        model.add(keras.layers.Conv2D(
            filter, kernel_size=3, strides=2, padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        filter = int(filter * 2)
    model.add(keras.layers.Conv2D(
        filter, kernel_size=3, strides=1, padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Conv2D(
        filters=1, kernel_size=3, strides=1, padding='same'))
    return model


class Generator_v_de(keras.Model):
    def __init__(self, filters, output_shape, repeat_num=0, num_cov=4, last_kernel_size=3, kernel_size=3, repeat=0, concat=False):
        super().__init__()
        if repeat_num != 0:
            self.repeat_num = repeat_num
        else:
            self.repeat_num = int(np.log2((np.max(output_shape[:-1])))) - 2
        self.num_conv = num_cov
        self.concat = concat
        self.x0_shape = [int(i / np.power(2, self.repeat_num - 1))
                         for i in output_shape[:-1]] + [filters]
        self.filters = filters
        self.kernel_size = kernel_size

        self.dense = keras.layers.Dense(int(np.prod(self.x0_shape)))

        # self.conv2dTransose_1 = keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=1)
        # self.conv2dTransose_2 = keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')

        self.conv2d_list = [keras.layers.Conv2D(
            self.filters, kernel_size=kernel_size, strides=1, padding='same') for _ in range(self.repeat_num*self.num_conv)]
        self.conv2d_last = keras.layers.Conv2D(
            output_shape[-1], kernel_size=last_kernel_size, strides=1, padding='same')
        filters = self.filters
        self.LeakyRelu = keras.layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        x = self.dense(x)
        x = tf.reshape(
            x, [-1, self.x0_shape[0], self.x0_shape[1], self.x0_shape[2]])
        x_0 = x

        for idx in range(self.repeat_num):
            for conv in range(self.num_conv):
                x = self.conv2d_list[idx*self.num_conv+conv](x)
                x = self.LeakyRelu(x)
            if idx < self.repeat_num - 1:
                _, h, w, _ = x.get_shape().as_list()
                if self.concat:
                    x = tf.image.resize(
                        x, [h*2, w*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    x_0 = tf.image.resize(
                        x_0, [h*2, w*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    x = tf.concat([x, x_0], axis=-1)
                else:
                    x += x_0
                    x = tf.image.resize(
                        x, [h*2, w*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    x_0 = x
            elif not self.concat:
                x += x_0
        x = self.conv2d_last(x)
        x = curl(x)

        return x
