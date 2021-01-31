import numpy as np
from pybalu.feature_selection import exsearch
from pybalu.feature_transformation import pca
from pybalu.feature_analysis import jfisher
from pyxvis.features.extraction import extract_features_labels
from pyxvis.features.selection import fsel, fse_model, clean_norm, clean_norm_transform
from pyxvis.io.plots import plot_features3, print_confusion
from sklearn.neighbors import KNeighborsClassifier as KNN


# Training-Data
path = '../images/fishbones/'
fx = ['basicint', 'gabor-ri', 'lbp-ri', 'haralick-2', 'fourier', 'dct', 'hog']
X, d = extract_features_labels(fx, path + 'train', 'jpg')
X, sclean, a, b = clean_norm(X)
(name, params) = fse_model('QDA')
ssfs = fsel([name, params], X, d, 15, cv=5, show=1)
X = X[:, ssfs]
Ypca, _, A, Mx, _ = pca(X, n_components=6)
X = np.concatenate((X, Ypca), axis=1)
sf = exsearch(X, d, n_features=3, method="fisher", show=True)
X = X[:, sf]
print('Jfisher = ' + str(jfisher(X, d)))
plot_features3(X, d, 'Fishbones')

# Testing-Data
Xt, dt = extract_features_labels(fx, path + 'test', 'jpg')
Xt = clean_norm_transform(Xt, sclean, a, b)
Xt = Xt[:, ssfs]
Ytpca = np.matmul(Xt - Mx, A)
Xt = np.concatenate((Xt, Ytpca) , axis=1)
Xt = Xt[:, sf]

# Classification and Evaluation
clf = KNN(n_neighbors=5)
clf.fit(X, d)
ds = clf.predict(Xt)
print_confusion(dt, ds)
