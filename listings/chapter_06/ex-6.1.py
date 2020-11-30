# [INPUT]  X  : training features (matrix of N x p elements)
#          d  : vector of training labels (vector of N elements)
#          Xt : testing features (matrix of Nt x p elements)
#          dt : vector of training labels (vector of Nt elements)
#          s  : string with the name of the model
# [OUTPUT] ds : classification (vector of Nt elements)
#          clf: trained classifier

from pyxvis.io.data import load_features
from pyxvis.learning.classifiers import clf_model, define_classifier
from pyxvis.learning.classifiers import train_classifier, test_classifier
from pyxvis.io.plots import print_confusion

# Definition of input variables
(X,d,Xt,dt) = load_features('../data/G3/G3')
s           = 'knn5'

# Training and Testing
(name,params) = clf_model(s)                     # function name and parameters
clf           = define_classifier([name,params]) # classifier definition
clf           = train_classifier(clf,X,d)        # classifier training
ds            = test_classifier(clf,Xt)          # clasification on testing 

# Evaluation of performance
print_confusion(dt,ds)
