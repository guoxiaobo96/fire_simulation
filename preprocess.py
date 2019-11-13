import cv2
import tensorflow as tf
import os
import numpy as np


def video2frame(video_path, frame_path):
    video_capture = cv2.VideoCapture()
    video_capture.open(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    # frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    for index in range(10000):
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = frame[150:650, 890:1000]
        if index > 200:
            cv2.imwrite(frame_path+str(index-200)+'.png',frame)

def crop_imgae(image_path,save_path):
    image = cv2.imread(image_path)
    # image = image[0:670,460:1470]
    cv2.imwrite(save_path, image)


def image2np(data_path):
    number = 0
    sequence = np.array([])
    for i in range(1,1001):
        frame_path = str(i)+".png"
        frame = os.path.join(data_path,frame_path)
        frame = tf.io.read_file(frame)
        frame = tf.image.decode_image(frame)
        frame = tf.cast(frame,dtype=tf.float32)/255.0
        sequence = np.append(sequence, frame)
        number += 1
    sequence = sequence.reshape(number, 500*110*3)
    np.savez('./data/train.npz',sequence_array=sequence)

def main():
    video2frame('e:/videoplayback.mp4','./data')

if __name__ == '__main__':
    main()
