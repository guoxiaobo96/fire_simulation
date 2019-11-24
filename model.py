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


        self.conv2d_list = [keras.layers.Conv2D(
            self.filters, kernel_size=kernel_size, strides=1, padding='same') for _ in range(self.repeat_num * self.num_conv)]
        self.batch_norm_list=[keras.layers.BatchNormalization() for _ in range(self.repeat_num * self.num_conv)]
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
                x = self.batch_norm_list[idx * self.num_conv + conv](x)
                
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

def build_generator_v(input_shape,output_shape):
    inputs = keras.layers.Input(shape=input_shape)
    # down_stack = [v_downsample(64, 4, apply_batchnorm=False), v_downsample(128, 4),v_downsample(256, 4),v_downsample(256, 4),v_downsample(512, 4),v_downsample(512, 4)]
    # up_stack = [v_upsamle(512, 4, apply_dropout=True),v_upsamle(512, 4, apply_dropout=True),v_upsamle(256, 4, apply_dropout=True),v_upsamle(256, 4),v_upsamle(128, 4),v_upsamle(64, 4)]
    down_stack = [v_downsample(64, 4, apply_batchnorm=False), v_downsample(128, 4)]
    up_stack = [v_upsamle(512, 4, apply_dropout=True),v_upsamle(512, 4, apply_dropout=True)]
    
    initializer = tf.random_normal_initializer(0., 0.02)

    last_conv = keras.layers.Conv2DTranspose(input_shape[-1], 3, 2, padding='same', kernel_initializer=initializer, activation='tanh')
    linear = keras.layers.Dense(int(np.prod(output_shape)))

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last_conv(x)
    # x = linear(x)
    x = tf.reshape(x, [-1, output_shape[0], output_shape[1], output_shape[2]])
    
    return tf.keras.Model(inputs=inputs, outputs=x)
    

def v_upsamle(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def v_downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

if __name__ == '__main__':
    input_shape = [52, 52, 2]
    a = tf.ones([4,52,52,2])
    generator = build_generator_v(input_shape, input_shape)
    a = generator(a, training=False)
    print('test')