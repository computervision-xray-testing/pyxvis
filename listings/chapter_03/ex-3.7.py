import numpy as np
import matplotlib.pylab as plt

from pyxvis.geometry.epipolar import estimate_fundamental_matrix
from pyxvis.geometry.epipolar import plot_epipolar_line
from pyxvis.io import gdxraydb

image_set = gdxraydb.Baggages()

data = image_set.load_data(44, 'Pmatrices')  # Load projection matrices

p, q = (1, 82)  # indices p and q

Ip = image_set.load_image(44, p)
Iq = image_set.load_image(44, q)

Pp = data['P'][:, :, p]  # Projection matrix of view p
Pq = data['P'][:, :, q]  # Projection matrix of view q

F = estimate_fundamental_matrix(Pp, Pq, method='pseudo')

colors = 'bgrcmykw'  # Colors for each point-line pair

fig1, ax1 = plt.subplots(1, 1)
fig1.suptitle('Figure p')
ax1.imshow(Ip, cmap='gray')
ax1.axis('off')

fig2, ax2 = plt.subplots(1, 1, subplot_kw=dict(xlim=(0, Ip.shape[1]), ylim=(Iq.shape[0], 1)))
fig2.suptitle('Figure q')
ax2.imshow(Iq, cmap='gray')
ax2.axis('off')
fig2.show()

for i in range(8):
    plt.figure(fig1.number)      # Focus on fig1 and get the mouse locations
    m = np.hstack([np.array(plt.ginput(1)), np.ones((1, 1))]).T  # Click
    ax1.plot(m[0, 0], m[1, 0], f'{colors[i]}*')  # Plot lines and plot on figures
    fig1.canvas.draw()
    ax2 = plot_epipolar_line(F, m, line_color=colors[i], ax=ax2)
    fig2.canvas.draw()

plt.show()
