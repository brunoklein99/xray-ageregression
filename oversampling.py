import pandas as pd
from random import randint
from sklearn.utils import shuffle
from tqdm import tqdm


def oversample_age_decades(frame: pd.DataFrame):
    # create a brand new dataframe for immutability reasons
    frame = pd.DataFrame(frame)
    # will hold the count of samples per decade
    decades_count = []
    # will hold a view of the frame for each decade
    decades_views = []
    # patients range from 0 to 100 yo, therefore 10 decades
    for i in range(10):
        low = i * 10  # from
        high = low + 10  # to
        # all instances within current decade
        ages_decade = frame[(low < frame['Patient Age']) & (frame['Patient Age'] < high)]
        # add the count of occurrences
        decades_count.append(len(ages_decade))
        # add the view
        decades_views.append(ages_decade)
    # get the count of the decade with the most occurrences
    max_decade_count = max(decades_count)
    # will hold all the rows we are adding to the final dataframe
    rows = []
    for i, (v, c) in enumerate(zip(decades_views, decades_count)):
        print('oversampling decade {}'.format(i))
        # if this is the decade with the most occurrences we don't want/have to oversample
        if c == max_decade_count:
            continue
        # how much should we oversample this decade
        to_add_count = max_decade_count - c
        # oversampling...
        indexes = []
        for _ in tqdm(range(to_add_count)):
            # choose a random sample to re-add to the dataframe
            index = randint(0, c - 1)
            indexes.append(index)
        rows.append(v.iloc[indexes])
    # append the final dataframe
    frame = frame.append(rows)
    return frame


def oversample(frame: pd.DataFrame, column):
    frame = pd.DataFrame(frame)
    counts = frame[column].value_counts(ascending=False)
    iterator = counts.iteritems()
    _, target = next(iterator)
    for val, count in iterator:
        # print('val', val)
        subframe = frame[frame[column] == val]
        subframe = shuffle(subframe)
        pending = target - count
        i = 0
        rows = []
        while pending > 0:
            # print('pending', pending)
            row = subframe.iloc[i % len(subframe)]
            rows.append(row)
            pending -= 1
            i += 1
        frame = frame.append(rows)
        # print('len', len(frame))
    return frame
