import numpy as np
from sklearn.metrics import accuracy_score

from pyxvis.io.data import load_features
from pyxvis.io.plots import show_clf_results
from pyxvis.learning.classifiers import clf_model, define_classifier
from pyxvis.learning.classifiers import train_classifier, test_classifier

# List of classifiers
ss_cl = ['dmin', 'lda', 'qda', 'maha', 'knn3', 'knn5', 'knn7', 'knn11', 'knn15',
         'bayes-naive', 'bayes-kde', 'adaboost', 'lr', 'rf', 'tree',
         'svm-lin', 'svm-rbf(0.1,1)', 'svm-rbf(0.1,0.5)', 'svm-rbf(0.5,1)',
         'svm-pol(0.05,0.1,2)', 'svm-pol(0.05,0.5,2)', 'svm-pol(0.05,0.5,3)',
         'svm-sig(0.1,1)', 'svm-sig(0.1,0.5)', 'svm-sig(0.5,1)',
         'nn(10,)', 'nn(20,)', 'nn(12,6)', 'nn(20,10,4)']

(X, d, Xt, dt) = load_features('../data/G3/G3')  # load training and testing data

n = len(ss_cl)
acc_train = np.zeros((n,))
acc_test = np.zeros((n,))
for k in range(n):
    (name, params) = clf_model(ss_cl[k])  # function name and parameters
    clf = define_classifier([name, params])  # classifier definition
    clf = train_classifier(clf, X, d)  # classifier training
    d0 = test_classifier(clf, X)  # classification of training
    ds = test_classifier(clf, Xt)  # classification of testing
    acc_train[k] = accuracy_score(d, d0)  # accuracy in training
    acc_test[k] = accuracy_score(dt, ds)  # accuracy in testing
    print(f'{k:3d}' + ') ' + f'{ss_cl[k]:20s}' + ': ' +
          f'Acc-Train = {acc_train[k]:.4f}' + '   ' + f'Acc-Test = {acc_test[k]:.4f}')
ks = np.argmax(acc_test)
print('-----------------------------------------------------------------')
print('Best Classifier:')
print(f'{ks:3d}' + ') ' + f'{ss_cl[ks]:20s}' + ': ' +
      f'Acc-Train = {acc_train[ks]:.4f}' + '   ' + f'Acc-Test = {acc_test[ks]:.4f}')
print('-----------------------------------------------------------------')

(name, params) = clf_model(ss_cl[ks])  # function name and parameters
clf = define_classifier([name, params])  # classifier definition
clf = train_classifier(clf, X, d)  # classifier training
d0 = test_classifier(clf, X)  # classification of training
ds = test_classifier(clf, Xt)  # classification of testing
show_clf_results(clf, X, d, Xt, dt, d0, ds, ss_cl[ks])  # display results and decision lines
