import numpy as np
from pyxvis.io.data import load_features,save_features
from pyxvis.learning.evaluation import best_features_classifier
from pyxvis.features.selection import clean_norm,clean_norm_transform
from pyxvis.features.extraction import extract_features_labels

dataname = 'thdata' # prefix of npy files of training and testing data
fxnew    = 1        # the features are (0) loaded or (1) extracted and saved
if fxnew:
    # features to extract
    fx        = ['flusser','hugeo','basicgeo','fourierdes','gupta']
    # feature extraction in training images
    path      = '../images/threatobjects/'
    X,d       = extract_features_labels(fx,path+'train','jpg',segmentation = 'bimodal')
    # feature extraction in testing images
    Xt,dt     = extract_features_labels(fx,path+'test','jpg',segmentation = 'bimodal')
    # backup of extracted features
    save_features(X,d,Xt,dt,dataname)
else:
    X,d,Xt,dt = load_features(dataname)
Nx            = X.shape[1]
X,sclean,a,b  = clean_norm(X)
Xt            = clean_norm_transform(Xt,sclean,a,b)
# Classifiers to evaluate
ss_cl         = ['maha','bayes-kde','svm-lin','svm-rbf','qda','lda','knn3','knn7','nn']
# Number of features to select
ff            = [2,3,5,10,15,20]
# Feature selectors to evaluate
ss_fs         = ['fisher','qda','svm-lin','svm-rbf']

clbest,ssbest = best_features_classifier(ss_fs,ff,ss_cl,X,d,Xt,dt,
                                         'Accuracy in Threat Objects')
print('   Extracted Features: '+str(Nx))
print('     Cleaned Features: '+str(len(sclean)))
print('    Selected Features: '+str(len(ssbest))+ ' > '+str((np.sort(sclean[ssbest]))))
