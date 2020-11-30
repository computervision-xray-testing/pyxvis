import numpy as np
from pyxvis.io.data import load_features,save_features
from pyxvis.learning.evaluation import best_features_classifier
from pyxvis.features.selection import clean_norm,clean_norm_transform
from pyxvis.features.extraction import extract_features_labels

dataname = 'fbdata' # prefix of npy files of training and testing data
fxnew    = 1        # the features are (0) loaded or (1) extracted and saved
if fxnew:
    # features to extract
    fx        = ['basicint','gabor-ri','lbp-ri','haralick-2','fourier','hog']
    # feature extraction in training images
    path      = '../images/fishbones/'
    X,d       = extract_features_labels(fx,path+'train','jpg')
    # feature extraction in testing images
    Xt,dt     = extract_features_labels(fx,path+'test','jpg')
    # backup of extracted features
    save_features(X,d,Xt,dt,dataname)
else:
    X,d,Xt,dt = load_features(dataname)

X,sclean,a,b  = clean_norm(X)
Xt            = clean_norm_transform(Xt,sclean,a,b)
# Classifiers to evaluate
ss_cl         = ['maha','bayes-kde','svm-lin','svm-rbf','qda','lda','knn3','knn7','nn']
# Number of features to select
ff            = [3,5,10,12,15]
# Feature selectors to evaluate
ss_fs         = ['fisher','qda','svm-lin','svm-rbf']

clbest,ssbest = best_features_classifier(ss_fs,ff,ss_cl,X,d,Xt,dt,
                                         'Accuracy in Fishbones')
print('    Selected Features: '+str((np.sort(sclean[ssbest]))))
