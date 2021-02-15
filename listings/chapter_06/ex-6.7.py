from pyxvis.io.data import load_features
from pyxvis.io.plots import show_clf_results
from pyxvis.learning.classifiers import clf_model, define_classifier
from pyxvis.learning.classifiers import train_classifier, test_classifier

(X, d, Xt, dt) = load_features('../data/F2/F2')  # load training and testing data
# (X,d,Xt,dt) = load_features('../data/P2/P2')        # data for the donut example
ss_cl = ['svm-lin', 'svm-rbf(0.1,0.05)', 'svm-rbf(0.03,1)', 'svm-pol(0.1,0.5,2)']

for cl_name in ss_cl:
    (name, params) = clf_model(cl_name)  # function name and parameters
    clf = define_classifier([name, params])  # classifier definition
    clf = train_classifier(clf, X, d)  # classifier training
    d0 = test_classifier(clf, X)  # classification of training
    ds = test_classifier(clf, Xt)  # classification of testing
    show_clf_results(clf, X, d, Xt, dt, d0, ds, cl_name)  # display results and decision lines
