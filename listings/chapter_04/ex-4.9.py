import matplotlib.pylab as plt
import cv2 as cv

from pyxvis.io import gdxraydb
from pyxvis.processing.segmentation import region_growing
from pyxvis.io.visualization import binview

image_set = gdxraydb.Baggages()

img = image_set.load_image(3, 4)
img = cv.resize(img, None, fx=0.35, fy=0.35, interpolation=cv.INTER_AREA)

th = 40  # threshold
si, sj = (403, 190)  # Seed

mask = region_growing(img, (si, sj), tolerance=th)

seg = binview(img, mask, 'g')

fig, ax = plt.subplots(1, 3, figsize=(14, 8))
ax[0].imshow(img, cmap='gray')
ax[0].plot(sj, si, 'r+')
ax[0].axis('off')
ax[1].imshow(mask, cmap='gray')
ax[1].axis('off')
ax[2].imshow(seg)
ax[2].axis('off')
plt.tight_layout()
plt.show()
