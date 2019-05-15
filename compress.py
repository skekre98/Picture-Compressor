import numpy as np
import copy
import sys
import math
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

def compression(img,n_colors):
    # transform image into 2D numpy array
    w, h, d  = tuple(img.shape)
    img_arr = np.reshape(img, (w * h, 3))

    # fitting sample to kmeans with n_colors being the number of colors
    img_arr_sample = shuffle(img_arr, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors).fit(img_arr_sample)

    # get labels for predictor
    labels = kmeans.predict(img_arr)
    return kmeans,labels,w,h

original_image = np.array(Image.open(sys.argv[1]))