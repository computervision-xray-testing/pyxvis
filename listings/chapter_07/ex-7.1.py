import numpy as np
from pyxvis.learning.classifiers import nn_definition
from pyxvis.learning.classifiers import nn_forward_propagation, nn_backward_propagation
from pyxvis.learning.classifiers import nn_parameters_update, nn_loss_function
from pyxvis.io.plots import plot_features_y,show_confusion_matrix, plot_loss
from pyxvis.io.data import load_features

# Load training and testing data
(Xtrain,Ytrain,Xtest,Ytest)   = load_features('../data/G4/G4',categorical=1)       
plot_features_y(Xtrain,Ytrain,'Training Subset')

# Definitions
N           = Xtrain.shape[1]    # training samples
n_0         = Xtrain.shape[0]    # number of inputs (X)
n_m         = Ytrain.shape[0]    # number of outputs (Y)
tmax        = 1000               # max number of iterations
alpha       = 10                 # learning rate
loss_eps    = 0.01               # stop if loss<loss_eps
nh          = [6,12]             # nodes of hidden layers
n           = [n_0]+nh+[n_m]     # nodes of each layer
m           = len(n)-1
ltrain      = np.zeros([tmax,1]) # training loss

# Training
t     = -1
train =  1
W,b   = nn_definition(n,N)                            # (step 1)
while train:
    t         = t+1
    a         = nn_forward_propagation(Xtrain,W,b)    # (step 2)
    dW,db     = nn_backward_propagation(Ytrain,a,W,b) # (step 3)
    W,b       = nn_parameters_update(W,b,dW,db,alpha) # (step 4)
    ltrain[t] = nn_loss_function(a,Ytrain)            # (step 5)
    train     = ltrain[t]>=loss_eps and t<tmax-1

# Loss function on training and validation subsets
plot_loss(ltrain)

# Evaluation on training and testing subsets 
a = nn_forward_propagation(Xtrain,W,b)    # output layer is a[m]
show_confusion_matrix(a[m],Ytrain,'Training',categorical=1)
a = nn_forward_propagation(Xtest,W,b)     # output layer is a[m]
show_confusion_matrix(a[m],Ytest,'Testing',categorical=1)






