import numpy as np
import matplotlib.pylab as plt
from pyxvis.simulation.xsim import superimpose_xray_images, draw_bounding_box
from pyxvis.io import gdxraydb

import gdxraydb

image_set = gdxraydb.Baggages()
Ib  = np.double(image_set.load_image(46, 2)) # background image

If1 = np.double(image_set.load_image(49, 2)) # foreground: Gun
p1  = [700,700]
It  = superimpose_xray_images(Ib, If1, p1[0], p1[1])

If2 = np.double(image_set.load_image(50, 4)) # foreground: Shuriken
p2  = [1200,100]
It  = superimpose_xray_images(It, If2,p2[0], p2[1])

If3 = np.double(image_set.load_image(51, 2)) # foreground: Razor Blade
p3  = [1300,1100]
It  = superimpose_xray_images(It, If3,p3[0],p3[1])


It = draw_bounding_box(It,p1[0],p1[1],If1.shape[0],If1.shape[1],'Gun')
It = draw_bounding_box(It,p2[0],p2[1],If2.shape[0],If2.shape[1],'Shuriken')
It = draw_bounding_box(It,p3[0],p3[1],If3.shape[0],If3.shape[1],'Blade')

fig1, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(Ib, cmap='gray'), ax[0].axis('off')
ax[1].imshow(It, cmap='gray'), ax[1].axis('off')
ax[1].text(p1[1]+20,p1[0]+50, '(simulated)', fontsize=8,color='white')
ax[1].text(p2[1]+20,p2[0]+50, '(simulated)', fontsize=8,color='black')
ax[1].text(p3[1]+20,p3[0]+50, '(simulated)', fontsize=8,color='black')
ax[1].text(1000,1800, '(real)', fontsize=12,color='white')
plt.show()
