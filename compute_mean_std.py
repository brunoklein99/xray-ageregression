from os.path import join

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

image_size = 224
pixel_per_image = image_size * image_size

files = open('train.csv').read().splitlines()
files = [join('images_resized', f) for f in files]
first = files[0:10]
array = np.zeros((10, image_size, image_size, 3), dtype=np.float32)
for i, file in enumerate(first):
    array[i, :, :, :] = cv2.imread(file)

print('first mean', np.mean(array, axis=(0, 1, 2)))
print('first std', np.std(array, axis=(0, 1, 2)))

rScaler = StandardScaler()
gScaler = StandardScaler()
bScaler = StandardScaler()
for file in tqdm(files):
    array = cv2.imread(file)
    # r = array[:, :, 0].flatten().reshape(-1, 1)
    # g = array[:, :, 1].flatten().reshape(-1, 1)
    # b = array[:, :, 2].flatten().reshape(-1, 1)
    r = np.reshape(array[:, :, 0], (-1, 1))
    g = np.reshape(array[:, :, 1], (-1, 1))
    b = np.reshape(array[:, :, 2], (-1, 1))
    rScaler.partial_fit(r)
    gScaler.partial_fit(g)
    bScaler.partial_fit(b)

print(rScaler.mean_)
print(gScaler.mean_)
print(rScaler.mean_)
print('')
print(np.sqrt(rScaler.var_))
print(np.sqrt(gScaler.var_))
print(np.sqrt(bScaler.var_))

# n = 0
# sumx = np.zeros((3,))
# sumx2 = np.zeros((3,))
# for file in tqdm(files):
#     array = cv2.imread(file).astype(np.float32)
#     n += pixel_per_image
#     sumx += np.sum(array, axis=(0, 1))
#     array = np.square(array)
#     sumx2 += np.sum(array, axis=(0, 1))
#
# mean = sumx / n
# variance = (sumx2 - (sumx * sumx) / n) / (n - 1)
# stddev = np.sqrt(variance)
#
# print('first mean', mean)
# print('first std', stddev)
