"""
ex_sfs_threatobjects.py: Example of extraction and selection of geometric features.
"""

# import numpy as np
# from pybalu.feature_selection import clean
from pybalu.feature_transformation import normalize, pca
from pyxvis.features.extraction import extract_features_labels
from pyxvis.features.selection import fse_model, fse_sbs, clean_norm, clean_norm_transform
from pyxvis.io.plots import plot_features3, print_confusion
from sklearn.neighbors import KNeighborsClassifier as knn

# Training-Data
path = '../images/threatobjects/'
fx = ['basicgeo', 'ellipse', 'hugeo', 'flusser', 'fourierdes', 'gupta']
X, d = extract_features_labels(fx, path + 'train', 'jpg', segmentation='bimodal')
X, sclean, a, b = clean_norm(X)
(name, params) = fse_model('LDA')
ssbs = fse_sbs([name, params], X, d, 20)
X = X[:, ssbs]
Ypca, _, _, _, _ = pca(X, n_components=3)
plot_features3(Ypca, d, 'PCA - Threat Objects', view=(-160, 120))

# Testing-Data
Xt, dt = extract_features_labels(fx, path + 'test', 'jpg', segmentation='bimodal')
Xt = clean_norm_transform(Xt, sclean, a, b)
Xt = Xt[:, ssbs]

# Classification and Evaluation
clf = knn(n_neighbors=5)
clf.fit(X, d)
ds = clf.predict(Xt)
print_confusion(dt, ds)
