import matplotlib.pylab as plt

from pyxvis.io import gdxraydb
from pyxvis.processing.segmentation import seg_bimodal
from pyxvis.io.visualization import binview


image_set = gdxraydb.Nature()
img = image_set.load_image(5, 9)

mask, contours = seg_bimodal(img)
seg = binview(img, mask, 'g')

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img, cmap='gray')
for n, contour in enumerate(contours):
        ax[0].plot(contour[:, 1], contour[:, 0], color='r', linewidth=3)
ax[0].axis('off')
ax[1].imshow(seg)
ax[1].axis('off')
plt.show()
