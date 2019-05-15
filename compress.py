import numpy as np
import copy
import sys
import math
from PIL import Image
from scipy.misc import imshow
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from matplotlib import pyplot as plt


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
    # returns new image in form of numpy array
    return new_img

# calculated compression rate given k
def KMeans_rate(original, k):
    # finds height and width of original image
    ow, oh, od  = tuple(original.shape)
    # calculated original size
    original_size = 24 * ow * oh

    # perform k-means using n_colors centroids
    kmeans,labels,w,h = compression(original_image,k)
    # calculated compressed size
    compressed_size = (32 * 3 * k) + (h * w * math.log(k,2))

    return ((original_size - compressed_size) / original_size) * 100

original_image = np.array(Image.open(sys.argv[1]))
temp = str(sys.argv[1])
image_name = temp.split('.')[0]
levels = {1: 200, 2: 100, 3: 50, 4:25, 5:5}
l = int(input("What level of compression would you like? (1-5): "))

n_im = K_means(original_image,levels[l])
n_im = n_im.astype('uint8')
img = Image.fromarray(n_im, 'RGB')
img.save('compressed_image.png')
img.show()


# 5 ~ level 5(90%)
# 25 ~ level 4(80%)
# 50 ~ level 3(75%)
# 100 ~ level 2(70%)
# 200 ~ level 1(67%)