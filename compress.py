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

def reshape_image(img, labels, w, h):
    # restructure image for plt.imshow()
    output = np.zeros((w, h, 3))
    index = 0
    for i in range(w):
        for j in range(h):
            output[i][j] = img[labels[index]]
            index += 1
    return output

def K_means(original_image,n_colors):
    # perform k-means using n_colors centroids
    kmeans,labels,w,h = compression(original_image,n_colors)
    # restructure image for output
    new_img = reshape_image(kmeans.cluster_centers_,labels,w,h)
    # visualize original image against new image
    visualize(original_image,new_img,n_colors)

original_image = np.array(Image.open(sys.argv[1]))