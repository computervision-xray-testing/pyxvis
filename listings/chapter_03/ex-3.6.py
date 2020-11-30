import cv2 as cv
import numpy as np

from pyxvis.io import gdxraydb
from pyxvis.io.visualization import project_edges_on_chessboard, gaussian_superimposition

image_set = gdxraydb.Settings()

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

nx = 10 #  number of inside corners per row
ny = 10 #  number of inside corners per column

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
objp = 250 * objp

# Arrays to store object points and image points from all the images.
obj_points = [] # 3d point in real world space
img_points = [] # 2d points in image plane.

img_boards = []

for i in range(1, 19):
    
    print('Find chessboard corners in image {}: '.format(i), end='')
    
    img = image_set.load_image(8, i)
    img_h = img.copy()
    
    # Keep a copy of the image but using three color channels. Just for visualization.
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
    ret, corners = cv.findChessboardCorners(img_h, (nx, ny), flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    print('{}'.format(ret))
    
    if ret:
        obj_points.append(objp)
        corners = cv.cornerSubPix(img_h, corners, (11, 11), (-1, -1), criteria)
        img = cv.drawChessboardCorners(img, (nx, ny), corners, ret)
        img_points.append(corners)
        img_boards.append({'img': img, 'idx': i})

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, img_h.shape[::-1], None, None)

# Show parameters
print('ret: ', ret)
print('Mtx: ', mtx)
print('Dist: ', dist)
print('Rvecs: ', rvecs)
print('Tvecs: ', tvecs)

i = 6
img = image_set.load_image(8, i)

# Projection matrix of image i. Remember that for this example
# indexation of rotation and translation matrices starts at 0.
R, _ = cv.Rodrigues(rvecs[i-1])  # Rotation matrix 3x3
t = tvecs[i - 1]  # Translation vector 3x1            

H = np.hstack([R, t])
H = np.vstack([H, np.array([0, 0, 0, 1])])

# Projection matrix
P = np.hstack([mtx, np.zeros((3, 1))])
P = np.dot(P, H)

project_edges_on_chessboard(img, P, square_size=250)

gaussian_superimposition(img, P, square_size=250, n_points=30)
