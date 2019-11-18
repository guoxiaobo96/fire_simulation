import tensorflow as tf
from tensorflow.python import keras
import time
import numpy as np
import os
from config import get_config
from model import Generator, build_discriminator
from data import BatchManager
from util import *

class Trainer(object):
    def __init__(self, config, batch_manager):
        self.config = config

        self.root = config.data_dir
        self.dataset = config.data_path
        self.data_type = config.data_type
        self.arch = config.arch
        self.filters = config.filters
        self.batch_manager = batch_manager

        self.res_x = config.res_x
        self.res_y = config.res_y
        self.res_z = config.res_z
        self.b_num = config.batch_size
        self.test_b_num = config.test_batch_size

        self.repeat_num = config.repeat
        self.filters = config.filters
        self.num_conv = config.num_conv
        self.w1 = config.w1
        self.w2 = config.w2
        if 'dg' in self.arch:
            self.w3 = config.w3

        self.use_c = config.use_curl
        self.output_shape = [self.res_y,self.res_x,1]

        self.optimizer = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2
                
        self.model_dir = config.model_dir
        self.load_path = config.load_path


        self.lr_update = config.lr_update
        if self.lr_update == 'decay':
            lr_min = config.lr_min
            lr_max = config.lr_max
            self.g_lr = tf.Variable(lr_max, name='g_lr',dtype=tf.float16)
        elif self.lr_update == 'step':
            self.g_lr = tf.Variable(config.lr_max, name='g_lr', dtype=tf.float16)
        else:
            raise Exception("[!] Invalid lr update method")

        self.lr_update_step = config.lr_update_step
        self.log_step = config.log_step
        self.test_step = config.test_step
        self.save_sec = config.save_sec

        self.is_train = config.is_train
        if 'ae' in self.arch:
            self.z_num = config.z_num
            self.p_num = self.batch_manager.dof
            self.use_sparse = config.use_sparse
            self.sparsity = config.sparsity
            self.w4 = config.w4
            self.w5 = config.w5
            self.code_path = config.code_path
            self.build_model_ae()

        elif 'nn' in self.arch:
            self.z_num = config.z_num
            self.w_num = config.w_size
            self.p_num = self.batch_manager.dof
            self.build_model_nn()

        else:
            self.build_model()

        # r = np.loadtxt(os.path.join(self.root, +'v_range.txt'))
        # self.x_range = max(abs(r[0]), abs(r[1]))
            
    def build_model(self):
        self.generator = Generator(self.filters, self.output_shape, num_cov=self.num_conv, repeat_num=self.repeat_num)
        self.discriminator = build_discriminator(self.filters)

        if self.optimizer == 'adam':
            self._generator_optimizer = keras.optimizers.Adam(lr=self.g_lr, beta_1=self.beta1, beta_2=self.beta2)
            self._discriminator_optimizer = keras.optimizers.Adam(lr=self.g_lr,beta_1=self.beta1,beta_2=self.beta2)
        elif self.optimizer == 'gd':
            self._generator_optimizer = keras.optimizers.SGD(lr=self.g_lr)
            self._discriminator_optimizer = keras.optimizers.Adam(lr=self.g_lr,beta_1=self.beta1,beta_2=self.beta2)
        else:
            raise Exception("[!] Invalid opimizer")
    
    def fit(self):
        train_ds = self.build_dataset()
        for epoch in range(5):
            start_time = time.time()
            for x,y in train_ds:
                self.train_(x, y)
            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start_time))


    # @tf.function
    def train_(self, input_velocity, target_velocity):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_velocity = curl(self.generator(target_velocity))
            input_gradient, input_vort = jacobian(input_velocity)
            generated_gradient, generated_vort = jacobian(generated_velocity)
            disc_real_output = self.discriminator(tf.concat([input_velocity, input_vort],axis=-1),training=True)
            disc_generated_output = self.discriminator(tf.concat([generated_velocity, generated_velocity],axis=-1),training=True)

            gen_loss = self._generator_loss(generated_velocity, target_velocity, disc_generated_output)
            disc_loss = self._discriminator_loss(disc_real_output,disc_generated_output)
        
        generator_gradients = gen_tape.gradient(gen_loss,
                                          self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self._generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        self._discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))


    def build_dataset(self):
        return self.batch_manager.build_dataset()
    # def build_dataset(self):
    #     train_dataset_x = tf.data.Dataset.from_generator(self._load_image,tf.float16)
    #     train_dataset_y = tf.data.Dataset.from_generator(self._load_label,tf.float16)
    #     train_dataset = tf.data.Dataset.zip((train_dataset_x, train_dataset_y))
    #     train_dataset = train_dataset.batch(self.b_num)
    #     return train_dataset
        
    # def _load_image(self):
    #     file_list = os.listdir(self.dataset + '/v')
    #     for f in file_list:
    #         if f.endswith('npz'):
    #             with np.load(self.dataset + '/v/' + f) as data:
    #                 x = data["x"] / self.x_range
    #                 yield data["x"]
        
    # def _load_label(self):
    #     file_list = os.listdir(self.dataset + '/v')
    #     for f in file_list:
    #         if f.endswith('npz'):
    #             with np.load(self.dataset + '/v/' + f) as data:
    #                 yield data["y"]

    def _discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = keras.losses.binary_crossentropy(tf.ones_like(disc_real_output), disc_real_output, from_logits=True)
        fake_loss = keras.losses.binary_crossentropy(tf.zeros_like(disc_generated_output), disc_generated_output, from_logits=True)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def _generator_loss(self, generated_velocity, real_velocity, disc_generated_output=None):
        velocity_loss = tf.reduce_mean(tf.abs(generated_velocity-real_velocity))
        generated_gradient, _ = jacobian(generated_velocity)
        real_gradient, _ = jacobian(real_velocity)
        gradient_loss = tf.reduce_mean(tf.abs(generated_gradient - real_gradient))
        total_loss = self.w1 * velocity_loss + self.w2 * gradient_loss
        if disc_generated_output:
            gan_loss = keras.losses.binary_crossentropy(tf.ones_like(disc_generated_output), disc_generated_output, from_logits=True)
            total_loss += self.w3 * gan_loss
        return total_loss

def main():
    config, _ = get_config()
    batch_manager = BatchManager(config)
    train = Trainer(config, batch_manager)
    train.fit()


if __name__ == '__main__':
    main()