from show_result import *
from util import *
from data import BatchManager
from model import Generator, build_discriminator
from config import get_config
from tqdm import tqdm
import glob
import numpy as np
import time
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
        self.w3 = config.w3

        self.use_c = config.use_curl
        self.output_shape = [self.res_y, self.res_x, 1]

        self.optimizer = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.lr_update = config.lr_update
        self.g_lr = tf.Variable(config.lr_max, name='g_lr')
        self.lr_min = config.lr_min
        self.lr_max = config.lr_max

        self.lr_update_step = config.lr_update_step
        self.log_step = config.log_step
        self.test_step = config.test_step

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

        self.num_samples = batch_manager.num_samples
        self.max_step = int(config.max_epoch // batch_manager.epochs_per_step)
        self.max_epoch = config.max_epoch

    def build_model(self):
        self.generator = Generator(
            self.filters, self.output_shape, num_cov=self.num_conv, repeat_num=self.repeat_num)
        self.discriminator = build_discriminator(self.filters)

        if self.optimizer == 'adam':
            self._generator_optimizer = keras.optimizers.Adam(
                lr=self.g_lr, beta_1=self.beta1, beta_2=self.beta2)
            self._discriminator_optimizer = keras.optimizers.Adam(
                lr=self.g_lr, beta_1=self.beta1, beta_2=self.beta2)
        elif self.optimizer == 'gd':
            self._generator_optimizer = keras.optimizers.SGD(lr=self.g_lr)
            self._discriminator_optimizer = keras.optimizers.Adam(
                lr=self.g_lr, beta_1=self.beta1, beta_2=self.beta2)
        else:
            raise Exception("[!] Invalid opimizer")
        checkpoint_dir = os.path.join(self.model_dir, "checkpoint")
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self._generator_optimizer,
                                              discriminator_optimizer=self._discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.summary_writer = tf.summary.create_file_writer(self.model_dir)


    def fit(self):
        train_ds = self.build_dataset()
        xi, _, zi = self.batch_manager.random_list(self.b_num)
        standard_velocity = tf.convert_to_tensor(xi, tf.float32)
        zi = np.array(zi)
        build_image_from_tensor(denorm_img(
            standard_velocity).numpy(), self.model_dir, 'standard')

        for step in tqdm(range(self.max_step), ncols=70):
            x, y = next(train_ds)
            self.train_(x, y, tf.cast(step,dtype=tf.int64))

            if step % self.test_step == 0:
                self.validate(step,standard_velocity,zi)

            if self.lr_update == 'step':
                if step % self.lr_update_step == self.lr_update_step - 1:
                    tf.compat.v1.assign(self.g_lr, tf.maximum(
                        self.g_lr * 0.5, self.lr_min))
            elif self.lr_update == 'decay':
                tf.compat.v1.assign(self.g_lr, self.lr_min+0.5*(self.lr_max-self.lr_min)*(tf.cos(
                    tf.cast(step, tf.float32)*np.pi/self.max_step)+1))

            if step % self.log_step == self.log_step - 1:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)


    # @tf.function
    def train_(self, input_velocity, target_velocity, step):
        with self.summary_writer.as_default():
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_velocity = self.generator(target_velocity, training=True)
                # input_gradient, input_vort = jacobian(input_velocity)
                # generated_gradient, generated_vort = jacobian(generated_velocity)
                # disc_real_output = self.discriminator(tf.concat([input_velocity, input_vort],axis=-1),training=True)
                # disc_generated_output = self.discriminator(tf.concat([generated_velocity, generated_vort],axis=-1),training=True)

                # gen_loss = self._generator_loss(generated_velocity, input_velocity, disc_generated_output)
                # disc_loss = self._discriminator_loss(disc_real_output,disc_generated_output)
                gen_loss = self._generator_loss(generated_velocity, input_velocity)

            generator_gradients = gen_tape.gradient(gen_loss,
                                                        self.generator.trainable_variables)
            # discriminator_gradients = disc_tape.gradient(disc_loss,
            #                                             self.discriminator.trainable_variables)

            self._generator_optimizer.apply_gradients(zip(generator_gradients,
                                                        self.generator.trainable_variables))
                # self._discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                #                                             self.discriminator.trainable_variables))
            tf.summary.scalar('generation_loss', gen_loss, step=step)
        

    def build_dataset(self):
        return self.batch_manager.build_dataset()

    def learingrate_update(self):
        # if self.lr_update == 'decay':
        #     lr_min = config.lr_min
        #     lr_max = config.lr_max
        #     tf.assign(self.g_lr, lr_min+0.5*(lr_max-lr_min)*(tf.cos(tf.cast(self.step, tf.float32)*np.pi/self.max_step)+1))
        # elif self.lr_update == 'step':
        tf.assign(self.g_lr, tf.maximum(self.g_lr*0.5, config.lr_min))

    def _discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = keras.losses.binary_crossentropy(tf.ones_like(
            disc_real_output), disc_real_output, from_logits=True)
        fake_loss = keras.losses.binary_crossentropy(tf.zeros_like(
            disc_generated_output), disc_generated_output, from_logits=True)
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, generated_velocity, real_velocity, disc_generated_output=None):
        velocity_loss = tf.reduce_mean(
            tf.abs(generated_velocity-real_velocity))
        generated_gradient, _ = jacobian(generated_velocity)
        real_gradient, _ = jacobian(real_velocity)
        gradient_loss = tf.reduce_mean(
            tf.abs(generated_gradient - real_gradient))
        total_loss = self.w1 * velocity_loss + self.w2 * gradient_loss
        # gan_loss = keras.losses.binary_crossentropy(tf.ones_like(disc_generated_output), disc_generated_output, from_logits=True)
        # total_loss += self.w3 * gan_loss
        return total_loss

    def generate_and_save(self, generate_input, file_name):
        generate_velocity = self.generator(generate_input, training=False)
        generate_velocity = denorm_img(generate_velocity)
        build_image_from_tensor(generate_velocity.numpy(),
                                self.model_dir, file_name)

    def validate(self, step, standard_velocity, generation_input):
        generated_velocity = self.generator(tf.convert_to_tensor(generation_input, tf.float32), training=False)
        loss = self._generator_loss(generated_velocity, standard_velocity).numpy()
        print(loss)
        build_image_from_tensor(denorm_img(
                    generated_velocity).numpy(), self.model_dir, step + 1)

def main():
    config, _ = get_config()
    batch_manager = BatchManager(config)
    train = Trainer(config, batch_manager)
    train.fit()


if __name__ == '__main__':
    main()
