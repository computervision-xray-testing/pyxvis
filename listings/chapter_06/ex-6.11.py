from pyxvis.learning.classifiers import clf_model,define_classifier
from pyxvis.learning.classifiers import train_classifier,test_classifier
from pyxvis.io.plots import show_confusion_matrix
from pyxvis.io.data import load_features

(X,d,Xt,dt)   = load_features('../data/F2/F2')       # load training and testing data

# Classifier definition
ss_cl  = ['dmin','svm-rbf(0.1,1)']
n      = len(ss_cl)
for k in range(n):
    (name,params) = clf_model(ss_cl[k])              # function name and parameters
    clf           = define_classifier([name,params]) # classifier definition
    clf           = train_classifier(clf,X,d)        # classifier training
    ds            = test_classifier(clf,Xt)          # clasification of testing
    show_confusion_matrix(dt,ds,ss_cl[k])            # display confusion matrix
