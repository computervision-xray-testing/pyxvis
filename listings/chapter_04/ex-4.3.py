import numpy as np
import matplotlib.pylab as plt

from pyxvis.processing.images import shading, fspecial

mat_r1 = fspecial('gaussian', 256, 80)
mat_r1 = mat_r1 / np.max(mat_r1.flatten()) * 0.8

mat_r2 = fspecial('gaussian', 256, 60)
mat_r2 = mat_r2 / np.max(mat_r2.flatten()) * 0.4

i1 = 0.8
i2 = 0.4

mat_x = fspecial('gaussian', 256, 70)
mat_x = mat_x / np.max(mat_x.flatten()) * 0.7
mat_x[30:80, 30:80] = mat_x[30:80, 30:80] * 1.5

mat_y = shading(mat_x, mat_r1, mat_r2, i1, i2)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(mat_x, cmap='gray')
ax[0].axis('off');
ax[1].imshow(mat_y, cmap='gray')
ax[1].axis('off');

plt.show()
