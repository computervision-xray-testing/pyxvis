import numpy as np
import matplotlib.pylab as plt
from pyxvis.simulation.xsim import voxels_simulation
from pyxvis.geometry.projective import rotation_matrix_3d
from pyxvis.processing.images import linimg

# Binary 3D matrix containing the voxels of a 3D object
V  = np.load('../data/voxels_model400.npy')  

# Transformation (x,y)->(u,v)
(u_0,v_0,a_x,a_y) = (235,305,1.1,1.1)
K  = np.array([[a_x, 0, u_0], [ 0, a_y, v_0], [0,0,1]])

# Transformation (Xb,Yb,Zb)->(u,v)
f  = 1500  # focal length
P  = np.array([[f, 0, 0, 0], [0, f, 0, 0], [0,0,1,0]])

# Transformation (X,Y,Z)->(Xb,Yb,Zb)
R  = rotation_matrix_3d(0.5,0.1,0.6)
t  = np.array([-120, -120, 1000])
H  = np.vstack([np.hstack([R, t[:, np.newaxis]]), np.array([0, 0, 0, 1])])

# Transformation (X,Y,Z) -> (u,v)
Pt = np.matmul(K,np.matmul(P,H))   

# Simulation of projection (Q) and X-ray image (X)
Q  = voxels_simulation(400,400,V,7,Pt)               
X  = linimg(np.exp(-0.0001*Q))

# Output
fig1, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.imshow(X, cmap='gray'), ax.axis('off')
plt.show()
