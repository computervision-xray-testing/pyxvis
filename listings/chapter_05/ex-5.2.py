import matplotlib.pyplot as plt

from pyxvis.processing.segmentation import seg_bimodal
from pyxvis.features.extraction import extract_features
from pyxvis.io.plots import plot_ellipses_image


img    = plt.imread('../images/N0006_0003b.png') # input image with a fruit
R,_,   = seg_bimodal(img)                        # segmentation
fxell  = extract_features('ellipse',bw=R)        # extraction of elliptical features
print('Elliptical Features:')                    # show results
print(fxell)                                     # print elliptical features
plot_ellipses_image(img,fxell)                   # draw ellipse onto image





