import numpy as np
import matplotlib.pylab as plt

from pyxvis.io import gdxraydb
from pyxvis.processing.images import hist_forceuni

image_set = gdxraydb.Baggages()
img = np.double(image_set.load_image(44, 130))

x_box = img[750:2000, 1250:2000]
x_box = hist_forceuni(x_box)
img2 = img.copy()
img2[750:2000, 1250:2000] = x_box

fig1, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(img, cmap='gray'), ax[0].axis('off')
ax[1].imshow(img2, cmap='gray'), ax[1].axis('off')
plt.show()
