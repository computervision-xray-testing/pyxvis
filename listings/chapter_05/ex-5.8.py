"""
[INPUT]  X  : matrix of training features, one sample per row
         d  : vector of training labels
         Xt : matrix of testing features, one sample per row
         dt : vector of testing labels
         s  : string with the name of the model
         p  : number of features to be selected
[OUTPUT] q  : indices of selected features (columns)
         X  : new matrix of training features
         Xt : new matrix of testing features
"""

from sklearn.neighbors import KNeighborsClassifier as KNN
from pyxvis.features.selection import fse_model, fsel
from pyxvis.io.data import load_features
from pyxvis.io.plots import print_confusion

# Definition of input variables
(X, d, Xt, dt) = load_features('../data/F40/F40')
s = 'lda'
p = 5

# Feature selection
(name, params) = fse_model(s)
q = fsel([name, params], X, d, p, cv=5, show=1)
print(str(len(q)) + ' from ' + str(X.shape[1]) + ' features selected.')

# New training and testing data
X = X[:, q]
Xt = Xt[:, q]

# Classification and Evaluation
clf = KNN(n_neighbors=5)
clf.fit(X, d)
ds = clf.predict(Xt)
print_confusion(dt, ds)
