from math import ceil
from math import pow
from os.path import join

import cv2
import keras
import numpy as np
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import settings
from data_load import load_data
from keras import Model
from keras.applications import ResNet50
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.preprocessing import image
from sklearn.utils import shuffle


def get_generator(f, params):
    while True:
        vals = shuffle(f)
        for imagename, age in zip(vals['Image Index'], vals['Patient Age']):
            filename = join('images_resized', imagename)
            x = cv2.imread(filename)
            x = cv2.resize(x, (settings.IMAGE_SIZE, settings.IMAGE_SIZE))
            x = np.expand_dims(x[:, :, 0], axis=-1)
            x = x.astype(np.float32)
            x -= 126.95534595
            x /= 63.95665607
            x -= np.mean(x)
            x /= np.std(x)

            if params['flip_horizontal']:
                if np.random.rand() < 0.5:
                    x = image.flip_axis(x, axis=1)

            if params['rotation'] > 0:
                x = image.random_rotation(x, rg=params['rotation'])

            shift_w = params['shift_w']
            shift_h = params['shift_h']
            if shift_h > 0 or shift_w > 0:
                x = image.random_shift(x, wrg=shift_w, hrg=shift_h)

            y = age / 100

            # cv2.imshow('', x[:, :, 0])
            # cv2.waitKey()

            yield x, y


def batch_generator(gen, batch, length):
    while True:
        count = length
        while count > 0:
            bsize = batch if count > batch else count
            batch_x = np.zeros((bsize, settings.IMAGE_SIZE, settings.IMAGE_SIZE, 1))
            batch_y = np.zeros((bsize, 1))
            for i in range(bsize):
                x, y = next(gen)
                batch_x[i] = x
                batch_y[i] = y
            yield batch_x, batch_y
            count -= batch


def save_metric(history, name):
    filename = settings.CHART_FILENAME.format(name)
    plt.clf()
    plt.plot(history.history[name])
    plt.plot(history.history['val_{}'.format(name)])
    plt.title('model {}'.format(name))
    plt.ylabel(name)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filename)


def train(frame_train, frame_valid, frame_test, params):
    gen_train = batch_generator(get_generator(frame_train, params), settings.BATCH_SIZE, len(frame_train))
    gen_valid = batch_generator(get_generator(frame_valid, params), settings.BATCH_SIZE, len(frame_valid))
    gen_test = batch_generator(get_generator(frame_test, params), settings.BATCH_SIZE, len(frame_test))

    model = ResNet50(include_top=False, weights=None, input_shape=(settings.IMAGE_SIZE, settings.IMAGE_SIZE, 1))
    x = Flatten(name='flatten')(model.output)
    x = Dense(4096, activation='relu', name='fc2')(x)
    drop = params['dropout']
    if drop > 0:
        x = Dropout(drop)(x)
    x = Dense(1024, activation='relu', name='fc3')(x)
    x = Dense(512, activation='relu', name='fc4')(x)
    x = Dense(128, activation='relu', name='fc5')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(model.input, outputs=x)

    lr = pow(10, -params['lr_exp'])
    decay = pow(10, -params['decay_exp'])

    opt = SGD(lr=lr, momentum=0.9, decay=decay)

    model.compile(optimizer=opt, loss=keras.metrics.mae)

    steps_train = int(ceil(len(frame_train) / settings.BATCH_SIZE))
    steps_valid = int(ceil(len(frame_valid) / settings.BATCH_SIZE))
    steps_test = int(ceil(len(frame_test) / settings.BATCH_SIZE))

    # tensorboard = TensorBoard(histogram_freq=1, batch_size=batch_size, write_grads=True)
    checkpoint = ModelCheckpoint(settings.WEIGHTS_CHECKPOINT_FILENAME)

    hist = model.fit_generator(gen_train,
                               steps_per_epoch=steps_train,
                               epochs=settings.EPOCHS,
                               validation_data=gen_valid,
                               validation_steps=steps_valid,
                               callbacks=[checkpoint],
                               )

    save_metric(hist, 'loss')

    loss_test = model.evaluate_generator(gen_test, steps=steps_test)

    return hist, loss_test


if __name__ == "__main__":
    frame_train, frame_valid, frame_test = load_data()
    params = {
        'dropout': 0.2,
        'lr_exp': 2,
        'decay_exp': 5,
        'flip_horizontal': True,
        'rotation': 0,
        'shift_w': 0,
        'shift_h': 0,
    }
    train_result = train(frame_train, frame_valid, frame_test, params)
    print(train_result)
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
