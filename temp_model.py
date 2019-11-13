import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

data_dir = './data/candal_data'
HEIGHT = 500
WEIGHT = 110
FRAMES = 4

NUMBER = len(os.listdir(data_dir))


SEQUENCE = None

def read_frame(frame_dir):
    frame = tf.io.read_file(frame_dir)
    frame = tf.image.decode_image(frame)
    frame = tf.cast(frame,dtype=tf.float32)/255.0
    return frame

def get_data(batch_size, start,end,epochs):
    sequence = np.zeros((FRAMES+1,HEIGHT,WEIGHT,3),dtype = 'float32')
    for i in range(FRAMES+1):
        file_data = os.path.join(data_dir,str(start+i)+'.png')
        sequence[i] = read_frame(file_data)
    basic_sequence = np.zeros((batch_size, FRAMES, HEIGHT, WEIGHT, 3),dtype = 'float32')
    next_sequence = np.zeros((batch_size, 1, HEIGHT, WEIGHT, 3),dtype = 'float32')
    for _ in range(epochs):
        for index in range(start, end-FRAMES+1):
            basic_sequence[(index-1)%batch_size, :, :, :, :] = sequence[0:FRAMES]
            next_sequence[(index-1)%batch_size, :, :, :, :] = sequence[FRAMES]
            sequence[0:FRAMES,:,:,:] = sequence[1:FRAMES+1,:,:,:]
            if index < end-FRAMES:
                file_data = os.path.join(data_dir,str(index+FRAMES+1)+'.png')
                sequence[FRAMES] = read_frame(file_data)
            if index % batch_size==0:
                yield(basic_sequence,next_sequence)

def build_model():
    model = keras.models.Sequential()

    model.add(keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3),input_shape=(None, HEIGHT, WEIGHT, 3), padding='same', return_sequences=True))
    model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.ConvLSTM2D(filters=60, kernel_size=(3, 3), padding='same', return_sequences=True))
    # model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.ConvLSTM2D(filters=60, kernel_size=(3, 3), padding='same', return_sequences=True))
    # model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv3D(filters=3, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train(model):
    batch_size = 1
    epochs = 5
    model.fit_generator(get_data(batch_size,1,int(0.7*NUMBER),epochs), steps_per_epoch=int((0.7*NUMBER-FRAMES)/batch_size), epochs=5,validation_data=get_data(batch_size,int(0.7*NUMBER)+1,int(0.8*NUMBER),epochs),validation_steps=3)

    model.save('nice_model.h5')

def predict(model):
    model.load_weights('nice_model.h5')

    length = 48
    start = int(0.9*NUMBER)

    BASIC_SEQUENCE = np.zeros((1, FRAMES, HEIGHT, WEIGHT, 3))
    NEXT_SEQUENCE = np.zeros((1, 1, HEIGHT, WEIGHT, 3))
    GENE_SEQUENCE = np.zeros((length+FRAMES+1, HEIGHT, WEIGHT, 3))
    for i in range(FRAMES):
        file_data = os.path.join(data_dir,str(start+i)+'.png')
        GENE_SEQUENCE[i] = read_frame(file_data)

    for INDEX in range(start, start+length):
        target_data = os.path.join(data_dir,str(INDEX+FRAMES)+'.png')
        NEXT_SEQUENCE[0,:,:,:,:] =  read_frame(target_data)
        for i in range(FRAMES):
            BASIC_SEQUENCE[0,i,:,:,:] = GENE_SEQUENCE[INDEX-start+i]
        pic = model.predict(BASIC_SEQUENCE)
        GENE_SEQUENCE[INDEX-start+FRAMES] = pic[0][0]
        pic = pic[0][0]*255.0
        pic = tf.convert_to_tensor(pic,dtype=tf.uint8)
        pic = tf.image.encode_png(pic)
        tf.io.write_file('./data/predict/'+str(INDEX-start)+'.png', pic)

def create_video(img_path):
    img_path = img_path
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(img_path+'/predict.avi',fourcc,24,(WEIGHT,HEIGHT))
    for i in range(len(os.listdir(img_path))):
        img_name = str(i)+'.png'
        img = os.path.join(img_path,img_name)
        img = cv2.imread(img)
        video_writer.write(img)
    video_writer.release()
# model = build_model()
# predict(model)

create_video('./data/predict')
