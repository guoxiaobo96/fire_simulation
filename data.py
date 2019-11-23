import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
        self.config = config
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
        self.epochs_per_step = self.batch_size / float(self.num_samples) 

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
        
        r = np.loadtxt(os.path.join(self.root, self.data_type[0] + '_range.txt'))
        self.x_range = max(abs(r[0]), abs(r[1]))
        self.y_range = max(abs(r[0]), abs(r[1]))

        self.z_range = []
        self.z_num = []
        for i in range(self.c_num):
                p_name = self.args['p%d' % i]
                p_min = float(self.args['min_{}'.format(p_name)])
                p_max = float(self.args['max_{}'.format(p_name)])
                p_num = int(self.args['num_{}'.format(p_name)])
                self.z_range.append([p_min, p_max])
                self.z_num.append(p_num)

        # if 'ae' in config.arch:
        #     self.dof = int(self.args['num_dof'])
        #     label_dim = [self.dof, int(self.args['num_frames'])]
        # else:
        #     label_dim = [self.c_num]
        
    def build_dataset_velocity(self):
        train_dataset_velocity_now = tf.data.Dataset.from_generator(self._load_velocity_now, tf.float32)
        train_dataset_velocity_next = tf.data.Dataset.from_generator(self._load_velocity_next, tf.float32)
        train_dataset_para = tf.data.Dataset.from_generator(self._load_para_data, tf.float32)
        train_dataset = tf.data.Dataset.zip((train_dataset_velocity_now,train_dataset_velocity_next,train_dataset_para))
        train_dataset = train_dataset.repeat()
        # train_dataset = train_dataset.shuffle(self.num_samples)
        train_dataset = train_dataset.batch(self.batch_size)
        # train_dataset = train_dataset.prefetch(self.batch_size)
        return train_dataset.__iter__()
        
    
    # def _load_velocity_data(self):
    #     for f in self.paths:
    #         if f.endswith('npz'):
    #             with np.load(f) as data:
    #                 x = data["x"] / self.x_range
    #                 y = data["y"] / self.y_range
    #                 z = data["z"]
    #                 for i, ri in enumerate(self.z_range):
    #                     z[i] = (z[i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1
    #                 yield x,y,z

    def _load_velocity_now(self):
        for f in self.paths:
            if f.endswith('npz'):
                with np.load(f) as data:
                    x = data["x"] / self.x_range
                    yield x
    
    def _load_velocity_next(self):
        for f in self.paths:
            if f.endswith('npz'):
                with np.load(f) as data:
                    y = data["y"] / self.y_range
                    yield y
    
    def _load_para_data(self):
        for f in self.paths:
            if f.endswith('npz'):
                with np.load(f) as data:
                    z = data["z"]
                    for i, ri in enumerate(self.z_range):
                        z[i] = (z[i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1
                    yield z
                 
    def denorm(self, x=None, y=None, z=None):
        # input range [-1, 1] -> original range
        if x is not None:
            x *= self.x_range
        if y is not None:
            y *= self.y_range
        if z is not None:
            r = self.y_range
            for i, ri in enumerate(self.y_range):
                z[:,i] = (z[:,i]+1) * 0.5 * (ri[1]-ri[0]) + ri[0]
        return x, y, z

    def random_list(self, num):
        xs = []
        ys = []
        pis = []
        zis = []
        for _ in range(num):
            pi = []
            for z_max in self.z_num:
                pi.append(self.rng.randint(z_max))

            filepath = self.list_from_p([pi])[0]
            x, y,_ = preprocess(filepath, self.data_type, self.x_range, self.y_range, self.z_range)
            if self.data_type[0] == 'v':
                b_ch = np.zeros((self.res_y, self.res_x, 1))
                # x = np.concatenate((x, b_ch), axis=-1)
            elif self.data_type[0] == 'l':
                offset = 0.5
                eps = 1e-3
                x[x<(offset+eps)] = -1
                x[x > -1] = 1

                y[y<(offset+eps)] = -1
                y[y>-1] = 1
            # x = np.clip((x+1)*127.5, 0, 255)
            zi = [(p/float(self.z_num[i]-1))*2-1 for i, p in enumerate(pi)] # [-1,1]

            xs.append(x)
            ys.append(y)
            pis.append(pi)
            zis.append(zi)
        return np.array(xs), pis, np.array(ys), np.array(zis)

    def list_from_p(self, p_list):
        path_format = os.path.join(self.root, self.data_type[0], self.args['path_format'])
        filelist = []
        for p in p_list:
            filelist.append(path_format % tuple(p))
        return filelist

def preprocess(file_path, data_type, x_range, y_range,z_range):    
    with np.load(file_path) as data:
        x = data['x']
        y = data['y']
        z = data['z']

    # # ############## for old data
    # if x.ndim == 4:
    #     x = x.transpose([2,0,1,3]) # yxzd -> zyxd
    # else:
    #     y = y[None,]
    #     x = x[:,::-1] # horizontal flip
    # else:
    #     x = x[::-1] # horizontal flip

    # normalize
    if data_type[0] == 'd':
        x = x * 2 - 1
        y = y*2 -1
    else:
        x /= x_range
        y /= y_range
        
    for i, ri in enumerate(z_range):
        z[i] = (z[i]-ri[0]) / (ri[1]-ri[0]) * 2 - 1
    return x, y, z
