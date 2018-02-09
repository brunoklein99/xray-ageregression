from functools import partial
from math import ceil
from os.path import join

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import plot
from keras import Model
from keras.applications import VGG16, ResNet50
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from math import pow
from sklearn.utils import class_weight

import pandas_utils


def get_age_weights(values: np.ndarray):
    i = np.unique(values)
    x = class_weight.compute_class_weight('balanced', i, values)
    return x


np.random.seed(0)
tf.set_random_seed(0)

frame = pd.read_csv('metadata/Data_Entry_2017.csv')
frame.head()
frame = frame[frame['Patient Age'] <= 100]


def filter_by_filenames(f, filenames):
    return f[f['Image Index'].isin(filenames)]


train_valid_names = open('metadata/train_val_list.txt').read().splitlines()
test_names = open('metadata/test_list.txt').read().splitlines()

frame_train_valid = filter_by_filenames(frame, train_valid_names)
frame_test = filter_by_filenames(frame, test_names)

frame_train, frame_valid = train_test_split(frame_train_valid, test_size=0.2, random_state=0)
print(frame_train.head())

frame_train = pandas_utils.oversample_age_decades(frame_train)
# frame_train = pandas_utils.oversample(frame_train, column='Patient Age')
frame_train = shuffle(frame_train)
# plot.plot_age_frame(frame_train)

ages = frame_train['Patient Age'].values
age_weights = get_age_weights(ages)
# y_pred = tf.constant(ages[:128])
# y_true = tf.constant(ages[:128])
# sess = tf.Session()
# mask = tf.one_hot(y_true, len(age_weights))
# mask = mask * tf.constant(age_weights, dtype=tf.float32)
# mask = tf.reduce_max(mask, axis=-1)
# mask = sess.run(mask)


print('train_size', len(frame_train))
print('valid_size', len(frame_valid))
print('test_size', len(frame_test))

batch_size = 32


def get_generator(f, params):
    while True:
        vals = shuffle(f)
        for imagename, age in zip(vals['Image Index'], vals['Patient Age']):
            filename = join('images_resized', imagename)
            x = cv2.imread(filename)
            x = cv2.resize(x, (params['size'], params['size']))
            x = np.expand_dims(x[:, :, 0], axis=-1)
            x = x.astype(np.float32)
            x -= 126.95534595
            x /= 63.95665607
            x -= np.mean(x)
            x /= np.std(x)

            # cv2.imshow('', x[:, :, 0])
            # cv2.waitKey()

            if params['flip_horizontal']:
                if np.random.rand() < 0.5:
                    x = image.flip_axis(x, axis=1)

            if params['rotation'] > 0:
                x = image.random_rotation(x, rg=params['rotation'])

            x = image.random_shift(x, wrg=params['shift_w'], hrg=params['shift_h'])

            y = age / 100

            yield x, y


def batch_generator(gen, batch, length, params):
    while True:
        count = length
        while count > 0:
            bsize = batch if count > batch else count
            batch_x = np.zeros((bsize, params['size'], params['size'], 1))
            batch_y = np.zeros((bsize, 1))
            for i in range(bsize):
                x, y = next(gen)
                batch_x[i] = x
                batch_y[i] = y
            yield batch_x, batch_y
            count -= batch


def train(params):
    size = params['size']
    gen_train = batch_generator(get_generator(frame_train, params), batch_size, len(frame_train), params)
    gen_valid = batch_generator(get_generator(frame_valid, params), batch_size, len(frame_valid), params)
    gen_test = batch_generator(get_generator(frame_test, params), batch_size, len(frame_test), params)

    model = ResNet50(include_top=False, weights=None, input_shape=(params['size'], params['size'], 1))
    x = Flatten(name='flatten')(model.output)
    drop = params['dropout']
    if drop > 0:
        x = Dropout(drop)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    # x = Dense(1024, activation='relu', name='fc3')(x)
    # x = Dense(512, activation='relu', name='fc4')(x)
    x = Dense(128, activation='relu', name='fc5')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(model.input, outputs=x)

    lr = pow(10, -params['lr_exp'])
    decay = pow(10, -params['decay_exp'])

    opt = SGD(lr=lr, momentum=0.9, decay=decay)

    model.compile(optimizer=opt, loss=keras.metrics.mae)

    steps_train = int(ceil(len(frame_train) / batch_size))
    steps_valid = int(ceil(len(frame_valid) / batch_size))
    steps_test = int(ceil(len(frame_test) / batch_size))

    # tensorboard = TensorBoard(histogram_freq=1, batch_size=batch_size, write_grads=True)

    loss = model.fit_generator(gen_train,
                               steps_per_epoch=steps_train,
                               epochs=5,
                               validation_data=gen_valid,
                               validation_steps=steps_valid,
                               # callbacks=[tensorboard],
                               )

    return loss


# print(model.evaluate_generator(gen_test, steps=steps_test))
if __name__ == "__main__":
    params = {
        'size': 224,
        'dropout': 0.2,
        'lr_exp': 2,
        'decay_exp': 5,
        'flip_horizontal': True,
        'rotation': 0,
        'shift_w': 0,
        'shift_h': 0,
    }
    print(train(params))
    # results = []
    # for _ in range(50):
    #     params = {
    #         'size': np.random.choice([100, 125, 150]),
    #         'dropout': np.random.uniform(0, 0.5),
    #         'lr_exp': np.random.randint(1, 3),
    #         'decay_exp': np.random.randint(3, 6),
    #         'flip_horizontal': np.random.choice([True, False]),
    #         'rotation': np.random.uniform(0, 10),
    #         'shift_w': np.random.uniform(0, 0.1),
    #         'shift_h': np.random.uniform(0, 0.1)
    #     }
    #     print('begin train:', params)
    #     results.append((train(params), params))
    #     print('end train:', params)
    # results = sorted(results, reverse=True)
    # print('results:')
    # print(results)
