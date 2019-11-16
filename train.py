import tensorflow as tf
from tensorflow.python import keras
from model import *
from util import *
from ops import *

class Trainer(object):
    def __init__(self, config, batch_manager):
        self.config = config
        self.batch_manager = batch_manager

        self.x, self.y = batch_manager.batch() # normalized input

        self.dataset = config.dataset
        self.data_type = config.data_type
        self.arch = config.arch
        self.filters = config.filters
        if 'nn' in self.arch:
            self.xt, self.yt = batch_manager.test_batch()
            self.xtw, self.ytw = batch_manager.test_batch(is_window=True)
            self.xw, self.yw = batch_manager.batch(is_window=True)            
        else:
            self.x_jaco, self.x_vort = jacobian(self.x)

        self.res_x = config.res_x
        self.res_y = config.res_y
        self.res_z = config.res_z
        self.c_num = batch_manager.c_num
        self.b_num = config.batch_size
        self.test_b_num = config.test_batch_size

        self.repeat = config.repeat
        self.filters = config.filters
        self.num_conv = config.num_conv
        self.w1 = config.w1
        self.w2 = config.w2
        if 'dg' in self.arch:
            self.w3 = config.w3

        self.use_c = config.use_curl
        if self.use_c:
            self.output_shape = get_conv_shape(self.x)[1:-1] + [1]
        else:
            self.output_shape = get_conv_shape(self.x)[1:]

        self.optimizer = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2
                
        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = config.start_step
        self.step = tf.Variable(self.start_step, name='step', trainable=False)
        # self.max_step = config.max_step
        self.max_step = int(config.max_epoch // batch_manager.epochs_per_step)

        self.lr_update = config.lr_update
        if self.lr_update == 'decay':
            lr_min = config.lr_min
            lr_max = config.lr_max
            self.g_lr = tf.Variable(lr_max, name='g_lr')
            self.g_lr_update = tf.assign(self.g_lr, 
               lr_min+0.5*(lr_max-lr_min)*(tf.cos(tf.cast(self.step, tf.float32)*np.pi/self.max_step)+1), name='g_lr_update')
        elif self.lr_update == 'step':
            self.g_lr = tf.Variable(config.lr_max, name='g_lr')
            self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr*0.5, config.lr_min), name='g_lr_update')    
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
            
    def build_model(self):
        self.G_s = generator(self.filters, self.output_shape, num_cov=self.num_conv, repeat_num=self.repeat)
        

    def train(self):
        self._train_()

    def train_(self):
        pass

    def train_ae(self):
        pass

    def train_nn(self):
        pass