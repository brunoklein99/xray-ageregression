import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_age_frame(frame: pd.DataFrame):
    f = frame['Patient Age'].value_counts(sort=False)
    index, count = zip(*sorted(zip(f.index, f.values)))
    plt.bar(index, count)
    plt.show()