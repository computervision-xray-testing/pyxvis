import numpy as np
import matplotlib.pylab as plt

from skimage.measure import find_contours

from pyxvis.io import gdxraydb
from pyxvis.processing.segmentation import seg_bimodal
from pyxvis.io.visualization import binview


image_set = gdxraydb.Welds()
img = image_set.load_image(1, 1)

mask = np.zeros(img.shape, np.uint8)  # Create a uint8 mask image
max_width = img.shape[1]

d1 = int(np.round(max_width/4))
d2 = int(np.round(d1 * 1.5))

i1 = 0

while i1 < max_width:
    i2 = min(i1 + d2, max_width)  # second column of partition
    img_i = img[:, i1:i2]  # partition i
    bw_i, _ = seg_bimodal(img_i)  # segmentation of partition i
    roi = mask[:, i1:i2]
    overlap = np.bitwise_or(roi, bw_i)  # addition into whole segmentation
    mask[:, i1:i2] = overlap
    i1 = i1 + d1  # update of first column

seg = binview(img, mask, color='g', dilate_pixels=5)
contours = find_contours(np.float32(mask), 0.5)

fig, ax = plt.subplots(2, 1, figsize=(14, 5))
ax[0].imshow(img, cmap='gray');
for n, contour in enumerate(contours):
        ax[0].plot(contour[:, 1], contour[:, 0], color='r', linewidth=3)
ax[0].axis('off')
ax[1].imshow(seg)
ax[1].axis('off')
fig.tight_layout()
plt.show()
