import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import cv2


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

def convert_data(data_file, p1_list, p2_list, frame_list):
    for p1 in p1_list:
        for p2 in p2_list:
            for frame in frame_list:
                file_path = p1+"_"+p2+"_"+frame+".npz"
                file_path = os.path.join(data_file,file_path)
                exist_data = np.load(file_path)["x"]
                if frame == frame_list[-1]:
                    next_frame = p1 + "_" + p2 + "_"  + str(int(frame)) + ".npz"
                else:
                    next_frame = p1 + "_" + p2 + "_" + str(int(frame)+1) + ".npz"
                next_frame_path = os.path.join(data_file, next_frame)
                next_frame = np.load(next_frame_path)["x"]
                new_data = np.savez_compressed(file_path, x=exist_data, y=next_frame, z=np.array([int(p1), int(p2),int(frame)]))
                    
if __name__ == '__main__':
    p1_list = [str(i) for i in range(5)]
    p2_list = [str(i) for i in range(11)]
    frame_list = [str(i) for i in range(200)]
    convert_data("data/fire_2d/v", p1_list, p2_list,  frame_list)
