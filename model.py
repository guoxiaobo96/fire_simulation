import os
import tensorflow as tf
from tensorflow.python import keras
import numpy as np


def discriminator(x, filter, repeat_num=3):
    filter = filter/2
    for _ in range(repeat_num):
        x = keras.layers.Conv2D(filter, kernal_size=3, strides=2)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        filter = filter * 2
    x = keras.layers.Conv2D(filter, kernel_size=3, strides=1)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    out_put = keras.layers.Conv2D(filters=1, kernel_size=3, strides=1)(x)
    return out_put

def generator(x, output_shape, filters, repeat_num=0, num_cov=4, last_kernel_size=3, kernel_size=3, repeat=0, concat=False):
    if repeat_num == 0:
        repeat_num = int(np.log2((np.max(output_shape[:-1])))) - 2
    x_0 = x
    for idx in repeat_num:
        for _ in range(num_cov):
            x = keras.layers.Conv2D(filter, kernel_size=kernel_size, strides=1)(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
        if idx < repeat_num - 1:
            if concat:
                x = keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2)(x)
                x_0 = keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2)(x_0)
                x = tf.concat([x,x_0],axis=-1)
            else:
                x += x_0
                x = keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=1)(x)
                x_0=x
        elif not concat:
            x += x_0
    out_put = keras.layers.Conv2D(x, output_shape[-1], kernel_size=last_kernel_size, strides=1,)
    return x

def NN(x, filters, out_num, dropout_rate=0.1, train=True):
    x = keras.layers.Dense(filters * 2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Dropout(rate=dropout_rate)(x)
    x = keras.layers.Dense(filters)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Dropout(rate=dropout_rate)(x)
    out_put = keras.layers.Dense(out_num)(x)
    return out_put


def encoder():
    pass