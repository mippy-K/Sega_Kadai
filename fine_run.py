import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model,load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np
import time
import cv2
from tensorflow.keras import models
import tensorflow as tf



classes = ['artificial lung','centrifugal pump','reservoir']
nb_classes = len(classes)

img_height, img_width = 64, 64
channels = 3

# VGG16
input_tensor = Input(shape=(img_height, img_width, channels))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)


# FC
fc = Sequential()
fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
fc.add(Dense(256, activation='relu'))
fc.add(Dropout(0.5))
fc.add(Dense(nb_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16.input, output=fc(vgg16.output))

# 学習済みの重みをロード
model.load_weights(os.path.join('model/finetuning.h5'))

model.summary()
cap = cv2.VideoCapture(0)

i=0
b=50

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    while i ==b:
        time.sleep(1)
        img = cv2.resize(frame,dsize=(64,64))
        np_image = np.array(img)
        np_image = np_image / 255
        np_image = np_image[np.newaxis, :, :, :]
        result = model.predict(np_image)[0]
        predicted = result.argmax()
        percentage = float(result[predicted] * 100)
        if percentage>75:
            print(classes[predicted])
            res=cv2.imread("show/"+classes[predicted]+".JPG")
            height=res.shape[0]
            width=res.shape[1]
            res=cv2.resize(res,dsize=None,fx=0.3,fy=0.3)
            cv2.imshow("res",res)
            cv2.waitKey(1000)
        else:
            print("unclassed")
        i=0
    i += 1

cap.release()
cv2.destroyAllWindows()
