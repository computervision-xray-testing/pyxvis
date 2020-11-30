import numpy as np
import matplotlib.pylab as plt
import cv2 as cv

from pyxvis.io import gdxraydb
from pyxvis.processing.images import res_minio

image_set = gdxraydb.Baggages()
img = image_set.load_image(46, 90)

n = 128
h = np.ones((1, n)) / n

img_g = cv.filter2D(img.astype('double'), cv.CV_64F, h)
fs = res_minio(img_g, h, method='minio')

fig, ax = plt.subplots(1, 3, figsize=(16, 8))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original image')
ax[0].axis('off')
ax[1].imshow(img_g, cmap='gray')
ax[1].set_title('Degraded image')
ax[1].axis('off')
ax[2].imshow(fs, cmap='gray')
ax[2].set_title('Restored image')
ax[2].axis('off')
plt.show()
