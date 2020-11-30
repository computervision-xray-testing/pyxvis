import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from pyxvis.features.extraction import extract_features

fig    = plt.figure()
ax     = fig.add_subplot(111)
img    = plt.imread('../images/N0001_0004b.png')
img[100:399,750:849]  = 0.5
img[500:699,850:916]  = 0.75
img[20:119,100:399]   = 0.6
img[90:156,1000:1199] = 0.75
implot = plt.imshow(img,cmap='gray')    
R      = img>0.27   # segmentation
L      = label(R)   # labeling
n      = np.max(L)  # number of segmented objects
t      = 0
T      = np.zeros((n,7))
for i in range(n):
    R = (L == i)*1                      # binary image of object i
    fx = ['basicgeo','hugeo']
    f = extract_features(fx,bw=R)       # feature extraction
    area = f[4]
    # recognition of fruits according to the size
    if area>10000 and area<31000:      
        h      = f[18:]                 # hu moments
        T[t,:] = h
        t      = t+1
        x      = round(1000*h[0])       # first hu moment
        ax.text(f[1]-20, f[0]+10, str(int(x)), fontsize=12,color='Red')
plt.show()
F = T[0:t,:]
print('Hu Features:')
print(F)
np.save('HuFeatures.npy',F)             # save features






