from sklearn.neural_network import MLPClassifier
from pyxvis.io.plots import plot_features2,show_confusion_matrix, plot_loss
from pyxvis.io.data import load_features

# Load training and testing data
(Xtrain,Ytrain,Xtest,Ytest)   = load_features('../data/G4/G4')       
plot_features2(Xtrain,Ytrain,'Training+Testing Subsets')

# Definitions
alpha       = 1e-5     # learning rate
nh          = (6,12)   # nodes of hidden layers
tmax        = 2000     # max number of iterations
solver      = 'adam'   # optimization approach ('lbfgs','sgd', 'adam') 

# Training
net = MLPClassifier(solver=solver, alpha=alpha,hidden_layer_sizes=nh, 
                    random_state=1,max_iter=tmax)
print(Xtrain.shape)
print(Ytrain.shape)
net.fit(Xtrain, Ytrain)

# Evaluation
Ym  = net.predict(Xtrain)
show_confusion_matrix(Ym,Ytrain,'Training')

Ys  = net.predict(Xtest)
show_confusion_matrix(Ys,Ytest,'Testing')