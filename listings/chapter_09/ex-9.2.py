from  pyxvis.learning.cnn import CNN

# execution type
type_exec    =  0 # training & testing

# pacthes' file for training and testing
patches_file  = '../data/weld32x32.mat'

# architechture
p = [9,7,5,3]        # Conv2D mask size 
d = [32,64,128,256]  # Conv2D channels
f = [64,32]          # fully connected

# training and testing 
CNN(patches_file,type_exec,p,d,f)

