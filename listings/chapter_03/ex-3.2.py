import numpy as np

from pyxvis.geometry.projective import rotation_matrix_3d

w = (35.0 / 180.0) * np.pi #Rotation in radians

# Translation tx,ty in cm
t = np.array([1.0, 3.0, 2.0])
t = t[:, np.newaxis]

R = rotation_matrix_3d(w, 0, 0)  # Generate the rotation matrix R

# Euclidean transformation matrix H = [R t; 0 0 1]
H = np.hstack([R, t])  
H = np.vstack([H, np.array([0, 0, 0, 1])])

Xp = 0  # x coordinate
Yp = 1  # y coordinate
Zp = 1

Mp = np.array([Xp, Yp, Zp, 1])  # A 2D point in homogeneous coordinates
Mp = Mp[:, np.newaxis]

M = np.dot(H, Mp)  # Transformation m to mp

M = M / M[-1]  # Normalize by matrix element (1, 4) into homogeneous coordinates

X = M.item(0)  # X coordinate
Y = M.item(1)  # Y coordinate
Z = M.item(2)  # Z coordinate

print('X = {:1.4f} mm -- Y = {:1.4f} mm -- Z = {:1.4f} mm'.format(X, Y, Z))
