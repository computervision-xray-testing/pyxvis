import numpy as np
import matplotlib.pylab as plt
from pyxvis.simulation.xsim import ellipsoid_simulation
from pyxvis.io import gdxraydb
from pyxvis.geometry.projective import rotation_matrix_3d

image_set = gdxraydb.Castings()
I  = np.double(image_set.load_image(21,27)) # wheel image

# Transformation (X,Y,Z)->(Xb,Yb,Zb)
R1 = rotation_matrix_3d(0,0,0)
t1 = np.array([-36, 40, 1000])
S = np.vstack([np.hstack([R1, t1[:, np.newaxis]]), np.array([0, 0, 0, 1])])

# Transformation (Xp,Yp,Zp)->(X,Y,Z)    
R2 = rotation_matrix_3d(0,0,np.pi/3)
t2 = np.array([0,0,0])
Se = np.vstack([np.hstack([R2, t2[:, np.newaxis]]), np.array([0, 0, 0, 1])])

# Transformation (Xp,Yp,Zp)->(Xb,Yb,Zb)   
SSe = np.matmul(S,Se)

# Transformation (x,y)->(u,v)
K = np.array([[1.1, 0, 235], [0, 1.1, 305], [0,0,1]])

# Dimensions of the ellipsoid
abc = (5,4,3)

# Simulation
J = ellipsoid_simulation(I,K,SSe,1500,abc,0.1,400)

# Output
fig1, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(I, cmap='gray'), ax[0].axis('off')
ax[1].imshow(J, cmap='gray'), ax[1].axis('off')
ax[1].text(328,225, '(Real)', fontsize=10,color='white')
ax[1].text(315,150, '(Simulated)', fontsize=10,color='white')
plt.show()
