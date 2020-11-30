import numpy as np
import matplotlib.pylab as plt

from pyxvis.io import gdxraydb

image_set = gdxraydb.Nature()
s = np.double(image_set.load_image(4, 1))

n = 20  
for i in range(2, n+1):  # For loops in Python runs until n-1
    xk = np.double(image_set.load_image(4, i))
    s += xk

y = s / n

fig1, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(s, cmap='gray'), ax[0].axis('off')
ax[1].imshow(y, cmap='gray'), ax[1].axis('off')
plt.show()
