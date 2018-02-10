import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from oversampling import oversample_age_decades


def filter_by_filenames(f, filenames):
    return f[f['Image Index'].isin(filenames)]


def load_data():
    # load data for each image in all our data
    frame = pd.read_csv('metadata/Data_Entry_2017.csv')

    # filter patients which are older than 100 yo
    frame = frame[frame['Patient Age'] <= 100]

    # get images name we train / validate on our own discretion
    train_valid_names = open('metadata/train_val_list.txt').read().splitlines()

    # these image should be used for final testing
    test_names = open('metadata/test_list.txt').read().splitlines()

    # filter the frames accordingly
    frame_train_valid = filter_by_filenames(frame, train_valid_names)
    frame_test = filter_by_filenames(frame, test_names)

    # split train / validation
    frame_train, frame_valid = train_test_split(frame_train_valid, test_size=0.2, random_state=0)

    # oversample train frame
    frame_train = oversample_age_decades(frame_train)

    # shuffle
    frame_train = shuffle(frame_train)

    return frame_train, frame_valid, frame_test


