import numpy as np
import numpy.matlib

import matplotlib.pylab as plt
from pyxvis.io import gdxraydb
from pyxvis.geometry.projective import rotation_matrix_3d
from pyxvis.geometry.projective import hyperproj

image_set = gdxraydb.Castings()  # Load the image set

# Load data for the the image set
hyp_model = image_set.load_data(1, 'HyperbolicModel.txt')  
man_pos = image_set.load_data(1, 'ManipulatorPosition.txt')

M = np.array([55, 55, -40, 1])
M = M[:, np.newaxis]

# Display the input image and points
fig, ax = plt.subplots(1, 5, figsize=(18, 14))

for i, p in enumerate(range(38, 47, 2)):
    
    t = np.hstack([man_pos[p, j] for j in range(3)])
    Rp = rotation_matrix_3d(man_pos[p, 3], man_pos[p, 4], man_pos[p, 5])
    Hp = np.vstack([np.hstack([Rp, t[:, np.newaxis]]), np.array([0, 0, 0, 1])])
    
    w = hyperproj(M, hyp_model, Hp)

    img = image_set.load_image(1, p)
    
    ax[i].imshow(img, cmap='gray')
    ax[i].scatter(w[1], w[0], facecolor='r', edgecolor='r', s=50)
    ax[i].axis('off')
    ax[i].set_title('Image {0}'.format(p))
    
plt.show()
