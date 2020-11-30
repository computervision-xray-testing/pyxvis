import numpy as np
import matplotlib.pylab as plt

from pyxvis.geometry.epipolar import recon_3dn
from pyxvis.io import gdxraydb

image_set = gdxraydb.Baggages()
data = image_set.load_data(44, 'Pmatrices')  # Load projection matrices

p, q, r = (1, 40, 90)  # indices for p, q and r

# Load projection matrices for views p, q, r
P1 = data['P'][:, :, p]  # Reprojection matrix of view p
P2 = data['P'][:, :, q]  # Reprojection matrix of view q
P3 = data['P'][:, :, r]  # Reprojection matrix of view r
P = np.vstack([P1, P2, P3])  # Join all projection matrices

Ip = image_set.load_image(44, p)
Iq = image_set.load_image(44, q)
Ir = image_set.load_image(44, r)

# Plot lines and plot on figures
fig1, ax1 = plt.subplots(1, 1, subplot_kw=dict(title='Figure p'))
ax1.imshow(Ip, cmap='gray')
ax1.axis('off')
print('Click first and second points in Figure 1 ...')
mp = np.hstack([np.array(plt.ginput(2)), np.ones((2, 1))]).T  # Click
ax1.plot(mp[0, :], mp[1, :], 'ro')
ax1.plot(mp[0, :], mp[1, :], 'g', linewidth=1.0)
fig1.canvas.draw()

fig2, ax2 = plt.subplots(1, 1, subplot_kw=dict(title='Figure q'))
ax2.imshow(Iq, cmap='gray')
ax2.axis('off')
print('Click first and second points in Figure 2 ...')
mq = np.hstack([np.array(plt.ginput(2)), np.ones((2, 1))]).T  # Click
ax2.plot(mq[0, :], mq[1, :], 'ro')
ax2.plot(mq[0, :], mq[1, :], 'g', linewidth=1.0)
fig2.canvas.draw()

fig3, ax3 = plt.subplots(1, 1, subplot_kw=dict(title='Figure r'))
ax3.imshow(Ir, cmap='gray')
ax3.axis('off'),
print('Click first and second points in Figure 3 ...')
mr = np.hstack([np.array(plt.ginput(2)), np.ones((2, 1))]).T  # Click
ax3.plot(mr[0, :], mr[1, :], 'ro')
ax3.plot(mr[0, :], mr[1, :], 'g', linewidth=1.0)
fig3.canvas.draw()

# 3D reprojection
mm_1 = np.vstack([mp[:, 0], mq[:, 0], mr[:, 0]]).T  # First 2D point in each view
mm_2 = np.vstack([mp[:, 1], mq[:, 1], mr[:, 1]]).T  # Second 2D point in each view
M1, d1, err1 = recon_3dn(mm_1, P)  # 3D reconstruction of first point
M2, d2, err2 = recon_3dn(mm_2, P)  # 3D reconstruction of second point

Md = M1.ravel()[:-1] - M2.ravel()[:-1]  # 3D vector from 1stt to 2nd point
dist = np.linalg.norm(Md)  # length of 3D vector in mm

print(f'Object size: {dist:0.3} mm')

plt.show()
