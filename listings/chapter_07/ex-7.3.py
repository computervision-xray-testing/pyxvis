from  pyxvis.learning.cnn import CNN

# execution type
type_exec     =  0 # training & testing

# patches' file for training and testing
patches_file  = '../data/C1/C1'

# architecture
p = [7,5,3]   # Conv2D mask size 
d = [4,12,8]  # Conv2D channels
f = [12]      # fully connected

# training and testing 
CNN(patches_file,type_exec,p,d,f)


