import os
import tensorflow as tf
from tensorflow.python import keras
import numpy as np

tf.keras.backend.set_floatx('float16')

def build_discriminator(filter, repeat_num=3):
    filter = int(filter / 2)
    model = keras.Sequential()
    for _ in range(repeat_num):
        model.add(keras.layers.Conv2D(filter, kernel_size=3, strides=2, padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        filter = int(filter * 2)
    model.add(keras.layers.Conv2D(filter, kernel_size=3, strides=1,padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Conv2D(filters=1, kernel_size=3, strides=1,padding='same'))
    return model


class Generator(keras.Model):
    def __init__(self, filters, output_shape ,repeat_num=0, num_cov=4, last_kernel_size=3, kernel_size=3, repeat=0, concat=False):
        super().__init__()
        if repeat_num != 0:
            self.repeat_num = repeat_num
        else:
            self.repeat_num = int(np.log2((np.max(output_shape[:-1])))) - 2
        self.num_conv = num_cov
        self.concat = concat
        self.x0_shape = [int(i / np.power(2, self.repeat_num - 1)) for i in output_shape[:-1]] + [filters]

        self.dense = keras.layers.Dense(int(np.prod(self.x0_shape)))

        self.conv2dTransose_1 = keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=1)
        self.conv2dTransose_2 = keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2)

        self.conv2d = keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')
        self.conv2d_last = keras.layers.Conv2DTranspose(output_shape[-1], kernel_size=last_kernel_size, strides=1,padding='same')

        self.LeakyRelu = keras.layers.LeakyReLU(alpha=0.2)
        
    def call(self, x):
        x = self.dense(x)
        x = tf.reshape(x, [-1, self.x0_shape[0], self.x0_shape[1], self.x0_shape[2]])
        x_0 = x

        for idx in range(self.repeat_num):
            for _ in range(self.num_conv):
                x = self.conv2d(x)
                x = self.LeakyRelu(x)
            if idx < self.repeat_num - 1:
                if self.concat:
                    x = self.conv2dTransose_2(x)
                    x_0 = self.conv2dTransose_2(x_0)
                    x = tf.concat([x, x_0], axis=-1)
                else:
                    x += x_0
                    x = self.conv2dTransose_2(x)
                    x_0 = x
            elif not self.concat:
                x += x_0
        out_put = self.conv2d_last(x)
        return out_put

def build_NN(filters, out_num, dropout_rate=0.1, train=True):
    model = keras.Sequential()
    model.add(keras.layers.Dense(filters * 2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ELU())
    model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(filters))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ELU())
    model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(out_num))
    return model

if __name__ == '__main__':
    generator = Generator(128, [128, 96, 2])
    generator.build(input_shape=[8,3,1])
    generator.summary()
    print('tem')