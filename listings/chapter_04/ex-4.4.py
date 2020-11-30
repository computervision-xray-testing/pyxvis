import numpy as np
import matplotlib.pylab as plt

from skimage.morphology import remove_small_objects, binary_dilation
from skimage.segmentation import clear_border

from pyxvis.io import gdxraydb
from pyxvis.io.visualization import binview
from pyxvis.processing.images import im_gaussian, im_median

image_set = gdxraydb.Castings()

X = image_set.load_image(21, 29)  # Original image
X = im_gaussian(X, k=5)  # Low pass filtering

fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
ax1.set_title('Original Image with defects')
ax1.imshow(X, cmap='gray')
ax1.axis('off')
plt.show()

Y0 = im_median(X, k=23)
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
ax2.set_title('Median filter')
ax2.imshow(Y0, cmap='gray')
ax2.axis('off')
plt.show()

Y1 = np.abs(np.double(X) - np.double(Y0))
fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
ax3.set_title('Difference Image')
ax3.imshow(np.log10(Y1 + 1), cmap='gray')
ax3.axis('off')
plt.show()

Y2 = Y1 > 18
fig4, ax4 = plt.subplots(1, 1, figsize=(6, 6))
ax4.set_title('Binary')
ax4.imshow(Y2, cmap='gray')
ax4.axis('off')
plt.show()

Y3 = remove_small_objects(Y2, 20)
fig5, ax5 = plt.subplots(1, 1, figsize=(6, 6))
ax5.set_title('Binary')
ax5.imshow(Y3, cmap='gray')
ax5.axis('off')
plt.show()

Y = clear_border(binary_dilation(Y3, np.ones((3, 3))))
fig6, ax6 = plt.subplots(1, 1, figsize=(6, 6))
ax6.set_title('Small region are eliminated')
ax6.imshow(Y, cmap='gray')
ax6.axis('off')
plt.show()

blend_mask = binview(X, Y, 'y', 1)
fig6, ax6 = plt.subplots(1, 1, figsize=(6, 6))
ax6.set_title('Small region are eliminated')
ax6.imshow(blend_mask, cmap='gray')
ax6.axis('off')
plt.show()
