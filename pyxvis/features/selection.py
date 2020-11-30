import numpy as np
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from pybalu.feature_selection import clean, exsearch
from pybalu.feature_transformation import normalize, pca
from pybalu.feature_selection import sfs as sfsfisher


def defineModel(bcl):
    model = eval(bcl[0]+'('+bcl[1]+')')
    return model

def fse_sbs(bcl,X,d,m):
    estimator = defineModel(bcl)
    selector  = RFE(estimator, m, step=1, verbose = 2)
    selector  = selector.fit(X, d)
    sel       = np.nonzero(selector.support_)[0]
    return sel


def fse_sbs_cv(bcl,X,d,m):
    estimator = defineModel(bcl)
    selector  = RFECV(estimator, min_features_to_select=m,step=1, cv=5)
    selector  = selector.fit(X, d)
    print(selector.grid_scores_)
    sel       = np.nonzero(selector.support_)[0]
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    plt.show()    
    return sel




def fse_sfs(bcl,X,d,m,cv = 0, show = 0):
    estimator = defineModel(bcl)
    sfs = SFS(estimator, k_features=m,forward=True,floating=False,verbose=2,scoring='accuracy',cv=cv)
    sfs = sfs.fit(X, d)
    sel = sfs.k_feature_idx_
    print(' ')
    if show:
        plot_sfs(sfs.get_metric_dict(), kind='std_err')
        plt.title('Sequential Forward Selection (w. StdErr)')
        plt.grid()
        plt.show()
    return sel


def fsel(bcl,X,d,m,forward=True,floating=False,cv = 0, show = 0):
    if show>0:
        print('Feature Selection - '+bcl[0]+':  - number of features reducing from ' + str(X.shape[1]) + ' to '+str(m)+' ...')
    if bcl[0]=='Fisher':
        sel = sfsfisher(X,d,m)
    else:
        estimator = defineModel(bcl)
        sfs = SFS(estimator, k_features=m,forward=True,floating=False,verbose=show,scoring='accuracy',cv=cv)
        sfs = sfs.fit(X, d)
        sel = list(sfs.k_feature_idx_)
        if show>0:
            print(' ')
        if show:
            plot_sfs(sfs.get_metric_dict(), kind='std_err')
            plt.title('Sequential Forward Selection')
            plt.grid()
            plt.show()
    return sel

def sfspca_old(X,d,m1,m2,cc,ex,m3):
    s1 = sfsfisher(X,d,m1)
    X  = X[:,s1]
    Y, _, _, _, _ = pca(X, n_components=m2)
    if cc == 1:
        Y = np.concatenate((X,Y),axis=1)
    if ex == 1:
        s2 = exsearch(Y,d,m3)
    else:
        s2 = sfs(Y,d,m3)
    X = Y[:,s2]
    return X

def sfspca(bcl,X,d,m1,m2,cc,ex,m3):
    s1 = fsel(bcl,X,d,m1)
    X  = X[:,s1]
    if m2>0:
        Y, _, _, _, _ = pca(X, n_components=m2)
    else:
        Y = X
    if cc == 1:
        Y = np.concatenate((X,Y),axis=1)
    if m3 > 0:
        if ex == 1:
            s2 = exsearch(Y,d,m3)
        else:
            s2 = fsel(bcl,Y,d,m3)
        X = Y[:,s2]
    else:
        X = Y
    return X


def clean_norm(X):
    sclean        = clean(X,show=True)
    X             = X[:,sclean]
    X,a,b         = normalize(X)
    return X,sclean,a,b

def clean_norm_transform(Xt,sclean,a,b):
    Xt            = Xt[:,sclean]
    Xt            = Xt*a + b
    return Xt




def fse_model(sto):
    st = sto.lower()
    if st == 'lr':
        name    = 'LogisticRegression'            # name of the classifier
        params  = ''                              # parameters of the classifier
    elif st == 'ridge':
        name    = 'Ridge'                         # name of the classifier
        params  = ''                              # parameters of the classifier
    elif st == 'lda':
        name    = 'LinearDiscriminantAnalysis'    # name of the classifier
        params  = ''                              # parameters of the classifier
    elif st == 'qda':
        name    = 'QuadraticDiscriminantAnalysis' # name of the classifier
        params  = ''                              # parameters of the classifier
    elif st == 'svm-lin':
        name    = 'SVC'                           # name of the classifier
        params  = 'kernel="linear"'               # parameters of the classifier
    elif st == 'svm-rbf':
        name = 'SVC'
        params = 'kernel = "rbf", gamma=0.01, C=0.01'
    elif st == 'bayes-naive':
        name = 'GaussianNB'
        params = ''
    elif st == 'nn':
        name = 'MLPClassifier'
        params = 'solver="adam", alpha=1e-5,hidden_layer_sizes=(10,), random_state=1,max_iter=2000'
    elif st == 'rf':
        name = 'RandomForestClassifier'
        params = 'n_estimators=20,random_state = 0'
    elif st == 'fisher':
        name = 'Fisher'
        params = ''
    return name,params


