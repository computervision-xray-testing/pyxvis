import numpy as np
import matplotlib.pylab as plt

from skimage.segmentation import find_boundaries

from pyxvis.io import gdxraydb
from pyxvis.processing.segmentation import seg_log_feature
from pyxvis.io.visualization import binview


image_set = gdxraydb.Castings()
X = image_set.load_image(31, 19)
X = X[0:572:2, 0:572:2]  # Donwsampling the image

fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
ax1.imshow(X, cmap='gray')
ax1.set_title('Input image')
ax1.axis('off')
plt.show()

R = X < 240

fig2, ax2 = plt.subplots(1, 1, figsize=(8,8))
ax2.imshow(R, cmap='gray')
ax2.set_title('Segmented object')
ax2.axis('off')
plt.show()

options = {
    'area': (30, 1500),  # Area range (area_min, area_max)
    'gray': (0, 150),  # Gray value range (gray_min, gray_max)
    'contrast': (1.08, 1.8),  # Contras range (cont_min, cont_max)
    'sigma': 2.5
}

Y, m = seg_log_feature(X, R, **options)

print(f'Found {m} regions.')

fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8))
ax3.imshow(binview(X, find_boundaries(Y)), cmap='gray')
ax3.set_title('Segmented regions')
ax3.axis('off')
plt.show()
