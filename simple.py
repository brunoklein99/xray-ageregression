'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
import pandas as pd
import numpy as np
import cv2
from keras.applications import VGG16, VGG19, ResNet50, InceptionV3
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K, Input, Model
from os.path import join

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

# x_train[0] = np.ones((img_cols, img_rows, 1))
# x_train[1] = np.zeros((img_cols, img_rows, 1))
# x_train[2] = np.ones((img_cols, img_rows, 1))
# x_train[3] = np.zeros((img_cols, img_rows, 1))
# x_train[4] = np.ones((img_cols, img_rows, 1))
#
# x_test[0] = np.ones((img_cols, img_rows, 1))
# x_test[1] = np.zeros((img_cols, img_rows, 1))
# x_test[2] = np.ones((img_cols, img_rows, 1))
# x_test[3] = np.zeros((img_cols, img_rows, 1))
# x_test[4] = np.ones((img_cols, img_rows, 1))
#
# y_train[0] = 1
# y_train[1] = 0
# y_train[2] = 1
# y_train[3] = 0
# y_train[4] = 1
#
# y_test[0] = 1
# y_test[1] = 0
# y_test[2] = 1
# y_test[3] = 0
# y_test[4] = 1

print('x_train.shape', x_train.shape)
print('x_test.shape', x_test.shape)

print('y_train.shape', y_train.shape)
print('y_test.shape', y_test.shape)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_cols, img_rows, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='sigmoid', name='predictions'))

# img_input = Input(shape=(img_rows, img_cols, 1))
# x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
# x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
#

# x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
#

# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
#

# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
#

# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
# x = Flatten(name='flatten')(x)
# x = Dense(4096, activation='relu', name='fc1')(x)
# x = Dense(4096, activation='relu', name='fc2')(x)
# x = Dense(1, activation='sigmoid', name='predictions')(x)
# model = Model(inputs=img_input, outputs=x)

model = VGG16(include_top=False, weights=None, input_shape=(img_cols, img_rows, 3))
x = model.output
x = Flatten()(model.output)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(1, activation='sigmoid', name='predictions')(x)
model = Model(inputs=model.input, outputs=x)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(img_cols, img_rows, 3)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

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

# print(model.evaluate(x_train, y_train))
# print('---')
# print(y_train)
#
# yhat = model.predict(x_train)
# print('loss: ', np.mean(np.abs(yhat - y_train)))
#
# score = model.evaluate(x_test, y_test, verbose=0)
#
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
