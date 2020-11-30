import numpy as np
import numpy.matlib

import matplotlib.pylab as plt
from pyxvis.io import gdxraydb

image_set = gdxraydb.Settings()

img = image_set.load_image(2, 1)  # Input image
data = image_set.load_data(2, 'points')  # Load data for the this image set

# Calibration coordinates in the image domain
um = data['ii'].flatten()  # This can also be done as um = data['ii'][::]
vm = data['jj'].flatten()  

# Coordinates of holes in cm.
xb = np.tile(np.r_[-6.5:7.5], [11, 1]).flatten()
xb = xb[:, np.newaxis]

yb = np.r_[-5.0:6.0]
yb = yb[:, np.newaxis]
yb = np.tile(yb, [1, 14]).flatten()
yb = yb[:, np.newaxis]

n = xb.shape[0]

# Build the design matrix
XX = np.hstack(
    [ np.ones((n, 1)), xb, yb, xb * yb, xb**2, yb**2, yb * (xb**2), xb * (yb**2), xb**3, yb**3]
)

a = np.linalg.lstsq(XX, um, rcond=None)[0]  # rcond=None silence warning for future deprecation
b = np.linalg.lstsq(XX, vm, rcond=None)[0]  # We refer the reader to the Numpy documentation.

# Also, you can compute Least squere regression as follow:
#a = np.dot(np.dot(np.linalg.inv(np.dot(XX.T, XX)), XX.T), um)

us = np.dot(a, XX.T)
vs = np.dot(b, XX.T)

d = np.array([um-us, vm-vs])
err = np.mean(np.sqrt(np.sum(d ** 2, axis=0)))

# Display the input image and points
fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.imshow(img, cmap='gray')
ax.scatter(vm, um, facecolor='none', edgecolor='g', s=120, label='Detected points')
ax.scatter(vs.flatten(), us.flatten(), facecolor='r', marker='+', s=100, label='Reprojected points')

# Plot vertical and horizontal lines
lines = np.vstack([us, vs])  # Stack reprojected points
for i in range(11):
    ax.plot(lines[1, (14 * i):(14 * (i + 1))], lines[0, (14 * i):(14 * (i + 1))], 'r:')  # Reprojected points

# Reshape and stack reprojected point
lines = np.vstack([us.reshape(-1, 14).T.flatten(), vs.reshape(-1, 14).T.flatten()])
for i in range(14):
    ax.plot(lines[1, (11 * i):(11 * (i + 1))], lines[0, (11 * i):(11 * (i + 1))], 'r:')  # Reprojected points

ax.axis('off')
ax.set_title('Cubic model reprojection error: {:0.4f} pixels'.format(err))
plt.legend(loc=1)
plt.show()
