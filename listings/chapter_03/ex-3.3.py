import numpy as np

from pyxvis.geometry.projective import get_matrix_p

f = 100  # Focal distance in cm
X = 20   # X coordiante in cm
Y = 30   # Y coordinate in cm
Z = 50   # Z coordinate in cm

M = np.array([X, Y, Z, 1])  # A 3D point in homogeneous coordinates
M = M[:, np.newaxis]

P = get_matrix_p(f)  # Create the projection matrix P

m = np.dot(P, M)  # Transformation M to m
m = m / m[-1]  # Homogeneous coordinates requires to be normalized by the matrix element (3, 1)

x = m.item(0)  # x coordinate
y = m.item(1)  # y coordinate

print('x = {:1.1f} cm -- y = {:1.1f} cm'.format(x, y))
