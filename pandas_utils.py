import pandas as pd
from sklearn.utils import shuffle


# def get_column_count(frame: pd.DataFrame, column, value):
#     subframe = frame[frame[column] == value]
#     return subframe[column].value_counts(ascending=False).iloc[0]

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
