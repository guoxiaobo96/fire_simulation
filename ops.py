import tensorflow as tf
from tensorflow.python import keras

def curl(x):
    return x

def get_conv_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]