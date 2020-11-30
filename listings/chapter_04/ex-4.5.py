import numpy as np
import matplotlib.pylab as plt
import cv2 as cv

from pyxvis.io import gdxraydb
from pyxvis.processing.images import fspecial, linimg, im_grad
from pyxvis.io.visualization import show_image_as_surface


image_set = gdxraydb.Baggages()
img = image_set.load_image(2, 1)
img = cv.resize(img, None, fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)

hs = fspecial('sobel')    # Sobel kernel
hp = fspecial('prewitt')  # Prewitt kernel

hg = fspecial('gaussian', 9, 1.0)
hg = cv.filter2D(hg, cv.CV_64F, np.array([-1, 1]))

gs, __ = im_grad(img, hs)
gp, __ = im_grad(img, hp)
gg, __ = im_grad(img, hg)

gradients = np.hstack([linimg(gs), linimg(gp), linimg(gg)])  # Stack the results as a same image.

plt.figure(figsize=(12, 6))
plt.imshow(gradients, cmap='gray')
plt.show()

img_y = np.log(gg + 1)

show_image_as_surface(img_y[-5:5:-1, -5:5:-1], elev=80, azim=-185, fsize=(10, 10), colorbar=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(img_y > 3, cmap='gray')
ax.axis('off')
plt.show()
