import numpy as np

from pyxvis.io.data import load_features
from pyxvis.learning.classifiers import clf_model
from pyxvis.learning.evaluation import cross_validation

# List of classifiers
ss_cl = ['dmin', 'lda', 'qda', 'maha', 'knn3', 'knn5', 'knn7', 'knn11', 'knn15',
         'bayes-naive', 'bayes-kde', 'adaboost', 'lr', 'rf', 'tree',
         'svm-lin', 'svm-rbf(0.1,1)', 'svm-rbf(0.1,0.5)', 'svm-rbf(0.5,1)',
         'svm-pol(0.05,0.1,2)', 'svm-pol(0.05,0.5,2)', 'svm-pol(0.05,0.5,3)',
         'svm-sig(0.1,1)', 'svm-sig(0.1,0.5)', 'svm-sig(0.5,1)',
         'nn(10,)', 'nn(20,)', 'nn(12,6)', 'nn(20,10,4)']

(X, d) = load_features('../data/G3/G3', full=1)  # load training and testing data

n = len(ss_cl)
folds = 10
acc = np.zeros((n,))
for k in range(n):
    (name, params) = clf_model(ss_cl[k])  # function name and parameters
    acc[k] = cross_validation([name, params], X, d, folds=folds)
    print(f'{k:3d}' + ') ' + f'{ss_cl[k]:20s}' + ': ' + f'CV-Accuracy = {acc[k]:.4f}')
ks = np.argmax(acc)

print('-----------------------------------------------')
print('Best Classifier:')
print(f'{ks:3d}' + ') ' + f'{ss_cl[ks]:20s}' + ': ' + f'CV-Accuracy = {acc[ks]:.4f}')
print('-----------------------------------------------')
