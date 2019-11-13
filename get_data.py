import tensorflow as tf
import os
import numpy as np

def read_frame(frame_path):
    frame = tf.io.read_file(frame_path)
    frame = tf.io.decode_image(frame)
    frame = tf.cast(frame, dtype=tf.float32)/255.0
    return frame

def build_data(source_frame, target_frame):
    souce = None
    target_frame = read_frame(target_frame)
    return source, target_frame


def get_data(source_frame, target_frame, batch_size, shuffle=False):
    data_set = tf.data.Dataset.from_tensor_slices((source_frame, target_frame))
    if shuffle:
        data_set = data_set.shuffle(1000)
    data_set = data_set.repeat(5)
    data_set = data_set.map(build_data)
    data_set = data_set.batch(batch_size).prefetch(2)
    return iter(data_set)


def main():
    data_dir = "./data/original_data/"
    frame = 2
    batch_size = 2
    source_frame = []
    target_frame = []
    for i in range(2, 10):
        target_frame.append(os.path.join(data_dir, str(i)+'.png'))
        source_frame.append([os.path.join(data_dir, str(index)+'.png')
                             for index in range(i-frame, i)])
    data_set = get_data(source_frame, target_frame,batch_size)
    temp = next(data_set)
    print('text')


if __name__ == "__main__":
    main()
