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
        
        if 'ae' in config.arch:
            self.dof = int(self.args['num_dof'])
            label_dim = [self.dof, int(self.args['num_frames'])]
        else:
            label_dim = [self.c_num]
        
def preprocess(file_path, data_type, x_range, y_range):
    with np.load(file_path) as data:
        x = data["x"]
        y = data["y"]
    if data_type[0] == 'd':
        x = x * 2 - 1
    else:
        x /= x_range
    for i, ri in enumerate(y_range):
        y[i] = (y[i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1
    return x, y
