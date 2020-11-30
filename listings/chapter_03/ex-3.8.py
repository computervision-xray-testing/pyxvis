import matplotlib.pylab as plt
import numpy as np

from pyxvis.geometry.epipolar import estimate_trifocal_tensor
from pyxvis.geometry.epipolar import reproject_trifocal
from pyxvis.io import gdxraydb

image_set = gdxraydb.Baggages()
data = image_set.load_data(44, 'Pmatrices')  # Load projection matrices

p, q, r = (1, 90, 170)  # Indices for p, q, and r

# Load projection matrices for views p, q, r
Pp = data['P'][:, :, p]
Pq = data['P'][:, :, q]
Pr = data['P'][:, :, r]

Ip = image_set.load_image(44, p)
Iq = image_set.load_image(44, q)
Ir = image_set.load_image(44, r)

T = estimate_trifocal_tensor(Pp, Pq, Pr)

# Plot lines and plot on figures
print('Click a point in Figure 1 ...')
fig1, ax1 = plt.subplots(1, 1, subplot_kw=dict(title='Figure p'))
ax1.imshow(Ip, cmap='gray')
ax1.axis('off')
mp = np.hstack([np.array(plt.ginput(1)), np.ones((1, 1))]).T  # Click
ax1.plot(mp[0], mp[1], 'r*')
fig1.canvas.draw()

print('Click a point in Figure 2 ...')
fig2, ax2 = plt.subplots(1, 1, subplot_kw=dict(title='Figure q'))
ax2.imshow(Iq, cmap='gray')
ax2.axis('off')
mq = np.hstack([np.array(plt.ginput(1)), np.ones((1, 1))]).T  # Click
ax2.plot(mq[0], mq[1], 'r*')
fig2.canvas.draw()

mr = reproject_trifocal(mp, mq, T)  # reprojection of mr from mp, mq and T

fig3, ax3 = plt.subplots(1, 1, subplot_kw=dict(title='Figure r'))
ax3.imshow(Ir, cmap='gray')
ax3.axis('off')
ax3.plot(mr[0, 0], mr[1, 0], 'r*')
fig3.canvas.draw()

plt.show()
