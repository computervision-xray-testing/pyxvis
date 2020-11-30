import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from pyxvis.features.extraction import extract_features

# Input Image
fig    = plt.figure()
ax     = fig.add_subplot(111)
img    = plt.imread('../images/N0001_0004b.png')
implot = plt.imshow(img,cmap='gray') 

# Segmentation   
R      = img>0.27         # thresholding of light objects
L      = label(R)         # labeling of objects
n      = np.max(L)        # number of detected objects
T      = np.zeros((n,18)) # features of each object will stored in a row

# Analysis of each segmented object
t      = 0 # count of recognized fruits
for i in range(n):
    R = (L == i)*1                         # binary image of object i
    f = extract_features('basicgeo',bw=R)  # feature extraction for object i
    area = f[4]
    # recognition of fruits according to the size
    if area>14000 and area<21000:
        T[t,:] = f                         # storing the features of the fruit t
        t = t+1
        # labeling each recognized fruit in the plot
        ax.text(f[1]-20, f[0]+10, str(t), fontsize=12,color='Red')

# Display and save results
plt.show()
F = T[0:t,:]
print('Basic Geo-Features:')
print(F)
np.save('GeoFeatures.npy',F)               # save features






