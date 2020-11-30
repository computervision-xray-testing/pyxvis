import numpy as np
import matplotlib.pylab as plt

from pyxvis.io import gdxraydb
from pyxvis.features.descriptors import compute_descriptors, match_descriptors
from pyxvis.io.visualization import plot_matches

image_set = gdxraydb.Baggages()
I1 = image_set.load_image(2, 1)  # Image 1
I2 = image_set.load_image(2, 2)  # Image 2

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 7))
ax1.imshow(I1, cmap='gray')
ax1.axis('off')
ax2.imshow(I2, cmap='gray')
ax2.axis('off')
fig1.tight_layout()
plt.show()

kp1, desc1 = compute_descriptors(I1, 'sift')  # SIFT descriptor for image 1
kp2, desc2 = compute_descriptors(I2, 'sift')  # SIFT descriptor for image 2

matches = match_descriptors(desc1, desc2, matcher='flann', max_ratio=0.7)  # Matching points using KDTREE

# Display results of matched points
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_matches(ax, I1, I2, kp1, kp2, matches, keypoints_color='lawngreen')
ax.axis('off')
plt.show()
