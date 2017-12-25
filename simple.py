'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

from os.path import join

import cv2
import keras
import numpy as np
import pandas as pd
from keras import Model
from keras.applications import VGG16
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten

batch_size = 8
epochs = 1000

img_rows, img_cols = 100, 100

frame_train = pd.read_csv('simple_train.csv')# 5 images
frame_valid = pd.read_csv('simple_valid.csv')# 5 images

x_train = np.empty(shape=(len(frame_train), img_rows, img_cols, 3))
x_train = x_train.astype('float32')

for i, imagename in enumerate(frame_train['Image Index']):
    img = cv2.imread(join('images_resized', imagename))
    img = cv2.resize(img, (img_rows, img_cols))
    # img = np.expand_dims(img, axis=-1)
    x_train[i] = img
    # cv2.imshow('img', img)
    # cv2.waitKey()

x_test = np.empty(shape=(len(frame_valid), img_rows, img_cols, 3))
x_test = x_test.astype('float32')

for i, imagename in enumerate(frame_valid['Image Index']):
    img = cv2.imread(join('images_resized', imagename))
    img = cv2.resize(img, (img_rows, img_cols))
    # img = np.expand_dims(img, axis=-1)
    x_test[i] = img

x_train /= 255
x_test /= 255

y_train = np.empty(shape=(len(frame_train), 1))
y_test = np.empty(shape=(len(frame_valid), 1))

for i, age in enumerate(frame_train['Patient Age']):
    y_train[i] = age

for i, age in enumerate(frame_valid['Patient Age']):
    y_test[i] = age

y_train = y_train / 100
y_test = y_test / 100

print('x_train.shape', x_train.shape)
print('x_test.shape', x_test.shape)

print('y_train.shape', y_train.shape)
print('y_test.shape', y_test.shape)

model = VGG16(include_top=False, weights=None, input_shape=(img_cols, img_rows, 3))
x = Flatten()(model.output)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(1, activation='sigmoid', name='predictions')(x)
model = Model(inputs=model.input, outputs=x)

model.compile(loss=keras.losses.mean_absolute_error,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-3),
              # optimizer=keras.optimizers.Adam(),
              )

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[TensorBoard(histogram_freq=1, write_grads=True)]
          )
