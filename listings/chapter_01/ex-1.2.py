from pyxvis.io import gdxraydb
from pyxvis.io.visualization import show_xray_image, show_color_array
from pyxvis.processing.images import dual_energy

import matplotlib.pylab as plt

# Select an images set and load images
image_set = gdxraydb.Baggages()

# Input images
img1 = image_set.load_image(60, 1)
img2 = image_set.load_image(60, 2)

# Display the input image using customized color map
show_xray_image([img1, img2], color_map='gray')

# Load LUT
lut = image_set.load_data(60, data_type='DualEnergyLUT')

# Compute dual energy image
energy_image = dual_energy(img1, img2, lut)

# Show results
plt.imshow(energy_image, cmap='viridis')
plt.axis('off')
plt.show()
