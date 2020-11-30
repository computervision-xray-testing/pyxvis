import numpy as np
import matplotlib.pylab as plt
from pyxvis.simulation.xsim import mask_simulation
from pyxvis.processing.helpers.kfunctions import gaussian_kernel
from pyxvis.io import gdxraydb

image_set = gdxraydb.Castings()
I  = np.double(image_set.load_image(21,25)) # wheel image

p1 = [150,580] # Location of 1st defect
p2 = [200,565] # Location of 2nd defect
p3 = [250,550] # Location of 3rd defect

h1 = gaussian_kernel(35,4)             # Gaussian Mask
h1 = h1/np.max(h1)*0.9
J  = mask_simulation(I,h1,p1[0],p1[1]) # Simulation 
h2 = np.ones((17,17))*0.4              # Square Mask
J  = mask_simulation(J,h2,p2[0],p2[1]) # Simulation 
h3 = np.zeros(h1.shape)                # Circle mask
h3[h1>0.25] = 0.4
J  = mask_simulation(J,h3,p3[0],p3[1]) # Simulation


# Output
fig1, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(I, cmap='gray'), ax[0].axis('off')
ax[1].imshow(J, cmap='gray'), ax[1].axis('off')
ax[1].text(p1[1]+20,p1[0]+5, 'Gaussian', fontsize=8,color='white')
ax[1].text(p2[1]+20,p2[0]+5, 'Square', fontsize=8,color='white')
ax[1].text(p3[1]+20,p3[0]+5, 'Circle', fontsize=8,color='white')
ax[1].text(250,210, '(Real)', fontsize=10,color='white')
ax[1].text(565,92, '(Simulated)', fontsize=10,color='white')
plt.show()
