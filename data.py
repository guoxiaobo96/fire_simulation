import os
import multiprocessing
from datetime import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


class BatchManager(object):
    def __init__(self, config):
        self.rng = np.random.RandomState(config.random_seed)
        self.root = config.data_path
        self.args = {}
        with open(os.path.join(self.root, 'args.txt'), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                arg, arg_value = line[:-1].split(': ')
                self.args[arg] = arg_value

        if 'ae' in config.arch:
            def sortf(x):
                nf = int(self.args['num_frames'])
                n = os.path.basename(x)[:-4].split('_')
                return int(n[0])*nf + int(n[1])

            self.paths = sorted(glob("{}/{}/*".format(self.root, config.data_type[0])),
                                key=sortf)
        else:
            self.paths = sorted(glob("{}/{}/*".format(self.root, config.data_type[0])))

        self.num_samples = len(self.paths)
        assert(self.num_samples > 0)
        self.batch_size = config.batch_size

        self.data_type = config.data_type
        if self.data_type == 'velocity':
            depth = 2
        else:
            depth = 1

        self.res_x = config.res_x
        self.res_y = config.res_y
        self.res_z = config.res_z
        self.depth = depth
        self.c_num = int(self.args['num_param'])
        
        feature_dim = [self.res_y, self.res_x, self.depth]
        
        r = np.loadtxt(os.path.join(self.root, self.data_type[0]+'_range.txt'))
        self.x_range = max(abs(r[0]), abs(r[1]))
        self.y_range = []
        self.y_num = []
        for i in range(self.c_num):
                p_name = self.args['p%d' % i]
                p_min = float(self.args['min_{}'.format(p_name)])
                p_max = float(self.args['max_{}'.format(p_name)])
                p_num = int(self.args['num_{}'.format(p_name)])
                self.y_range.append([p_min, p_max])
                self.y_num.append(p_num)

        # if 'ae' in config.arch:
        #     self.dof = int(self.args['num_dof'])
        #     label_dim = [self.dof, int(self.args['num_frames'])]
        # else:
        #     label_dim = [self.c_num]
        
    def build_dataset(self):
        train_dataset_x = tf.data.Dataset.from_generator(self._load_image,tf.float16)
        train_dataset_y = tf.data.Dataset.from_generator(self._load_label,tf.float16)
        train_dataset = tf.data.Dataset.zip((train_dataset_x, train_dataset_y))
        train_dataset = train_dataset.batch(self.batch_size)
        return train_dataset
        
    def _load_image(self):
        for f in self.paths:
            if f.endswith('npz'):
                with np.load(f) as data:
                    x = data["x"] / self.x_range
                    yield data["x"]
        
    def _load_label(self):
        for f in self.paths:
            if f.endswith('npz'):
                with np.load(f) as data:
                    y = data["y"]
                    for i, ri in enumerate(self.y_range):
                        y[i] = (y[i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1
                    yield y

