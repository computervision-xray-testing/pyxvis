import numpy as np

from pyxvis.geometry.projective import rotation_matrix_2d

th = 35.0 / 180.0 * np.pi #Rotation in radians
t = np.array([4.1, 3.2]).reshape(2, -1)  # Translation tx,ty in cm

# The same can also be done using newaxis
# t = np.array([4.1, 3.2])
# t = t[:, np.newaxis]

R = rotation_matrix_2d(th)  # Generate the rotation matrix R

# Euclidean transformation matrix H
H = np.hstack([R, t])  
H = np.vstack([H, np.array([0, 0, 1])])

x = 4.9  # x coordinate
y = 5.5  # y coordinate

# A 2D point in homogeneous coordinates
m = np.array([x, y, 1])  
m = m[:, np.newaxis]

mp = np.dot(np.linalg.inv(H), m)  # Transformation m to mp
mp = mp / mp[-1]  # Homogeneous coordinates requires to be normalized by the matrix element (3,3)

xp = mp.item(0)  # x' coordinate
yp = mp.item(1)  # y' coordinate

print('xp = {:1.4f} cm -- yp = {:1.4f} cm'.format(xp, yp))
