from sklearn.model_selection import train_test_split

from pyxvis.io.data import load_features
from pyxvis.io.plots import show_confusion_matrix
from pyxvis.learning.classifiers import clf_model
from pyxvis.learning.evaluation import hold_out

# load available dataset
(X0, d0) = load_features('../data/F2/F2', full=1)

# definition of training and testing data
X, Xt, d, dt = train_test_split(X0, d0, test_size=0.2, stratify=d0)

# definition of the classifier
cl_name = 'svm-rbf(0.1,1)'  # generic name of the classifier
(name, params) = clf_model(cl_name)  # function name and parameters

# Hold-out (train on (X,d), test on (Xt), compare with dt)
ds, acc, _ = hold_out([name, params], X, d, Xt, dt)  # hold out
print(cl_name + ': ' + f'Accuracy = {acc:.4f}')

# display confusion matrix
show_confusion_matrix(dt, ds, 'Testing subset')
