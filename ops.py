import tensorflow as tf
from tensorflow.python import keras

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)