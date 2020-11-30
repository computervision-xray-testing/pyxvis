import numpy as np
import matplotlib.pylab as plt
import cv2 as cv

from pyxvis.io import gdxraydb
from pyxvis.processing.images import Edge

image_set = gdxraydb.Baggages()

img = image_set.load_image(2, 1)
img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
img = img[595:715, 0:120]

plt.figure(figsize=(12, 6))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

threshold = np.array([1e-8, 1e-6, 1e-5, 1e-3, 1e-2])  # Different threshold values
sigma = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]) # Different sigma values

rows = np.array([])
for t in threshold:
    cols = np.array([])
    for s in sigma:
        detector = Edge('log', t, s)
        detector.fit(img)
        cols = np.hstack([cols, detector.edges]) if cols.size else detector.edges
    rows = np.vstack([rows, cols]) if rows.size else cols

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(rows, cmap='gray');

# Figure configuration
from matplotlib.ticker import FixedLocator, FixedFormatter
ax.set_title('Sigma', y=1.05)
ax.set_ylabel('Threshold')
ax.tick_params(bottom=False, top=True, left=False, right=True)
ax.tick_params(labelbottom=False, labeltop=True, labelleft=True, labelright=False)
x_formatter = FixedFormatter(sigma)
y_formatter = FixedFormatter(threshold)
x_locator = FixedLocator(60 + 120 * np.arange(sigma.shape[0]))
y_locator = FixedLocator(60 + 120 * np.arange(threshold.shape[0]))
ax.xaxis.set_major_formatter(x_formatter)
ax.yaxis.set_major_formatter(y_formatter)
ax.xaxis.set_major_locator(x_locator)
ax.yaxis.set_major_locator(y_locator)
plt.show()
