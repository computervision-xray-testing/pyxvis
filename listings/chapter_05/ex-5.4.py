import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from pyxvis.features.extraction import extract_features

fig    = plt.figure()
ax     = fig.add_subplot(111)
img    = plt.imread('../images/N0001_0004b.png')
implot = plt.imshow(img,cmap='gray')    
R      = img>0.27     # segmentation
L      = label(R)     # labeling
n      = np.max(L)    # number of segmented regions
t      = 0
T      = np.zeros((n,6))
for i in range(n):
    R = (L == i)*1    # binary image of object i
    f = extract_features('basicgeo',bw=R)
    area = f[4]
    # recognition of fruits according to the size
    if area>14000 and area<21000:
        # extract int features only in the segmented region
        h      = extract_features('basicint',img=img,bw=R)
        T[t,:] = h
        t = t+1
        ax.text(f[1]-20, f[0]+10, str(t), fontsize=12,color='Red')
plt.show()
F = T[0:t,:]
print('Basic Int-Features:')
print(F)
np.save('IntFeatures.npy',F) # save features






