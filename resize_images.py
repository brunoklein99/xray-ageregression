from os import listdir

import cv2
from os.path import join
from tqdm import tqdm

for file in tqdm(listdir('images')):
    img = cv2.imread(join('images', file))
    img = cv2.resize(img, (224, 224))
    cv2.imwrite(join('images_resized', file), img)
