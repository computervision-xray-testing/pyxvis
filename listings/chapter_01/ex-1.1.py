from pyxvis.io import gdxraydb
from pyxvis.io.visualization import show_xray_image, show_color_array, show_image_as_surface, dynamic_colormap

image_set = gdxraydb.Baggages()

# Input image
img = image_set.load_image(2, 4)

# Crop a region of interest within the image
roi = img[250:399, 340:529]

# Display the input image using customized color map
show_xray_image(img, color_map='gray')

# # Display the roi using various color maps
show_color_array(roi)

# Display the selected region into a 3D projection
show_image_as_surface(roi)
