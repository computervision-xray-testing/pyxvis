import numpy               as np
from seaborn                       import heatmap
from sklearn.ensemble              import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network        import MLPClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.neighbors             import NearestCentroid
from sklearn.neighbors             import KNeighborsClassifier 
from sklearn.linear_model          import LogisticRegression
from sklearn.ensemble              import RandomForestClassifier
from sklearn.tree                  import DecisionTreeClassifier
from sklearn.svm                   import SVC
from sklearn.metrics               import confusion_matrix, accuracy_score
from sklearn.model_selection       import cross_val_score
from sklearn.base                  import BaseEstimator, ClassifierMixin
from sklearn.neighbors             import KernelDensity
from pyxvis.features.selection     import fse_model, fsel

def define_classifier(bcl):
    if isinstance(bcl,str):
        bcl = clf_model(bcl)
    st = bcl[0]+'('+bcl[1]+')'
    # print('evaluating '+st+'...')
    clf = eval(st)
    return clf

def load_classifiers():
    clfs = [
        'NearestCentroid',               '',
        'KNeighborsClassifier',          'n_neighbors=1',
        'KNeighborsClassifier',          'n_neighbors=3',
        'KNeighborsClassifier',          'n_neighbors=5',
        'KNeighborsClassifier',          'n_neighbors=7',
        'LinearDiscriminantAnalysis',    '',
        'QuadraticDiscriminantAnalysis', '',
        'GaussianNB',                    '',
        'AdaBoostClassifier',            'n_estimators=100',
        'RandomForestClassifier',        'n_estimators=20,random_state = 0',
        'DecisionTreeClassifier',        'max_depth = 4, min_samples_leaf = 8,random_state = 0',
        'SVC',                           'kernel = "linear", gamma=0.2, C=0.1',
        'SVC',                           'kernel = "poly", gamma=0.2, degree = 3, C=0.1',
        'SVC',                           'kernel = "rbf", gamma=0.2,C=0.1',
        'SVC',                           'kernel = "sigmoid", gamma=0.01, C=0.01',
        'LogisticRegression',            'C=0.1,solver="lbfgs"',
        'MLPClassifier',                 'solver="adam", alpha=1e-5,hidden_layer_sizes=(3,2), random_state=1,max_iter=2000',
        'MLPClassifier',                 'solver="adam", alpha=1e-5,hidden_layer_sizes=(10,), random_state=1,max_iter=2000',
        ]
    m = len(clfs)
    bcl = dict()
    bcl['name'] = clfs[0:m:2]
    bcl['parm'] = clfs[1:m:2]
    return bcl

def clf_model(sto,show=0):
    st = sto.lower()
    if st == 'lr':
        name    = 'LogisticRegression'            # name of the classifier
        params  = 'C=0.1,solver="lbfgs"'                              # parameters of the classifier
    elif st == 'dmin':
        name    = 'NearestCentroid'    # name of the classifier
        params  = ''                              # parameters of the classifier
    elif st[0:3] == 'knn':
        name   = 'KNeighborsClassifier'
        params = 'n_neighbors='+st[3:]
    elif st == 'sklda':
        name    = 'LinearDiscriminantAnalysis'    # name of the classifier
        params  = ''                              # parameters of the classifier
    elif st == 'skqda':
        name    = 'QuadraticDiscriminantAnalysis' # name of the classifier
        params  = ''                              # parameters of the classifier
    elif st == 'lda':
        name    = 'LDA'                           # name of the classifier
        params  = ''                              # parameters of the classifier
    elif st == 'qda':
        name    = 'QDA'                           # name of the classifier
        params  = ''                              # parameters of the classifier
    elif st == 'maha':
        name    = 'Mahalanobis'                   # name of the classifier
        params  = 'covi=1'                        # parameters of the classifier
    elif st == 'maha-0':
        name    = 'Mahalanobis'                   # name of the classifier
        params  = 'covi=0'                        # parameters of the classifier
    elif st == 'svm-lin':
        name    = 'SVC'                           # name of the classifier
        params  = 'kernel="linear"'               # parameters of the classifier
    elif st[0:7] == 'svm-rbf':
        name   = 'SVC'
        if len(st)==7:
            C = 1
            gamma = 0.1
        else:
            (gamma,C) = eval(st[7:])
        params = 'kernel = "rbf", gamma='+str(gamma)+',C='+str(C)

    elif st[0:7] == 'svm-pol':
        name   = 'SVC'
        if len(st)==7:
            C = 1
            gamma = 0.1
        else:
            (gamma,C,degree) = eval(st[7:])
        params = 'kernel = "poly", gamma='+str(gamma)+',C='+str(C)+',degree='+str(degree)
        # params = 'kernel = "poly", gamma=0.1, degree = 3, C='+str(C)
    elif st[0:7] == 'svm-sig':
        name   = 'SVC'
        if len(st)==7:
            C = 0.1
            gamma = 0.001
        else:
            (gamma,C) = eval(st[7:])
        params = 'kernel = "sigmoid", gamma='+str(gamma)+',C='+str(C)
    elif st == 'bayes-naive':
        name = 'GaussianNB'
        params = ''
    elif st == 'bayes-kde':
        name = 'KDEClassifier'
        params = 'bandwidth=1.0'
    elif st[0:2] == 'nn':
        name   = 'MLPClassifier'
        if len(st)==2:
            hst = '(10,)'
        else:
            hst = st[2:]
        params = 'solver="adam", alpha=1e-5, random_state=1,max_iter=2000,hidden_layer_sizes='+hst
    elif st == 'rf':
        name = 'RandomForestClassifier'
        params = 'n_estimators=20,random_state = 0'
    elif st == 'tree':
        name = 'DecisionTreeClassifier'
        params = 'max_depth = 4, min_samples_leaf = 8,random_state = 0'
    elif st == 'adaboost':
        name = 'AdaBoostClassifier'
        params = 'n_estimators=100'
    else:
        print('Error: '+st+' does not exist as classification method in clf_model.')
    if show==1:
        print('using classifier: ' + name+'('+params+')...')
    return name,params

def train_classifier(clf,X,d):
    clf = clf.fit(X,d)
    return clf

def test_classifier(clf,Xt):
    ds  = clf.predict(Xt)
    return ds

def train_test_classifier(bcl,X,d,Xt):
    clf = define_classifier(bcl)
    clf = train_classifier(clf,X,d)
    ds  = test_classifier(clf,Xt)
    return ds,clf


class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE
    from https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

class Mahalanobis(BaseEstimator, ClassifierMixin):

    def __init__(self, covi=1):
        self.covi = covi # 0 covi = Cte, 1 = covi is different
        
    def fit(self, X, y):
        self.classes  = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes]
        m             = X.shape[1]
        n             = len(self.classes)
        self.mc       = np.zeros((n,m))
        self.Ck       = np.zeros((m,m,n))
        for k in range(n):
            Xk             = training_sets[k]
            self.mc[k]     = np.mean(Xk,axis=0)
            self.Ck[:,:,k] = np.cov(Xk.T)
        if self.covi == 0:
            C       = np.zeros((m,m))
            for k in range(n):
                C  = C + self.Ck[:,:,k]
                C = C/n
            for k in range(n):
                self.Ck[:,:,k] = C
        for k in range(n):
            self.Ck[:,:,k] = np.linalg.inv(self.Ck[:,:,k])
        return self

    def predict_distances(self, Xt):
        n = len(self.classes)
        nt = Xt.shape[0]
        distances = np.zeros((nt,n))
        for k in range(n):
            mk = self.mc[k]
            for i in range(nt):
                xdi  = Xt[i]-mk
                dik   = np.matmul(xdi, self.Ck[:,:,k])
                distances[i,k] = dik.dot(xdi)
        return distances
            
    def predict(self, X):
        return self.classes[np.argmin(self.predict_distances(X), 1)]

class LDA(BaseEstimator, ClassifierMixin):

    def __init__(self, probability=1):
        self.probability = probability

    def fit(self, X, y):
        self.classes  = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes]
        N             = X.shape[0]
        m             = X.shape[1]
        n             = len(self.classes)
        self.mc       = np.zeros((n,m))
        Cw            = np.zeros((m,m))
        pk            = np.zeros((n,1))
        for k in range(n):
            Xk    = training_sets[k]
            Lk    = Xk.shape[0]
            pk[k] = Lk/N
            C     = np.linalg.inv(np.cov(Xk.T))
            Cw    = Cw + C*(Lk-1)
            self.mc[k] = np.mean(Xk,axis=0)
        Cw = Cw/(N-n)
        self.invCw = np.linalg.inv(Cw)
        if self.probability == 1:
            self.logp = np.log(np.ones((n,1))/n)
        elif self.probability == 0:
            self.logp = np.log(pk)
        else:
            self.logp = np.log(self.probability)
        return self

    def predict_distances(self, Xt):
        n = len(self.classes)
        nt = Xt.shape[0]
        distances = np.zeros((nt,n))
        C  = self.invCw
        for k in range(n):
            mk = self.mc[k]
            for i in range(nt):
                xdi  = Xt[i]-mk
                dik   = np.matmul(xdi, C)
                distances[i,k] = -0.5*dik.dot(xdi)+self.logp[k]
        return distances
            
    def predict(self, X):
        return self.classes[np.argmax(self.predict_distances(X), 1)]

class QDA(BaseEstimator, ClassifierMixin):

    def __init__(self, probability=1):
        self.probability = probability

    def fit(self, X, y):
        self.classes  = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes]
        N             = X.shape[0]
        m             = X.shape[1]
        n             = len(self.classes)
        self.Ck       = np.zeros((m,m,n))
        self.mc       = np.zeros((n,m))
        pk            = np.zeros((n,1))
        for k in range(n):
            Xk             = training_sets[k]
            Lk             = Xk.shape[0]
            pk[k]          = Lk/N
            self.mc[k]     = np.mean(Xk,axis=0)
            self.Ck[:,:,k] = np.linalg.inv(np.cov(Xk.T))
        if self.probability == 1:
            self.logp = np.log(np.ones((n,1))/n)
        elif self.probability == 0:
            self.logp = np.log(pk)
        else:
            self.logp = np.log(self.probability)
        return self

    def predict_distances(self, Xt):
        n = len(self.classes)
        nt = Xt.shape[0]
        distances = np.zeros((nt,n))
        for k in range(n):
            C  = self.Ck[:,:,k]
            Cp = -0.5*np.log(np.linalg.det(C))+self.logp[k]
            mk = self.mc[k]
            for i in range(nt):
                xdi  = Xt[i]-mk
                dik   = np.matmul(xdi, C)
                distances[i,k] = -0.5*dik.dot(xdi) + Cp
        return distances
            
    def predict(self, X):
        return self.classes[np.argmax(self.predict_distances(X), 1)]

def nn_definition(n,N):
    W  = [None]
    b  = [None]
    m1 = len(n)
    for k in range(1,m1):
        W    = W  + [np.random.rand(n[k],n[k-1])]
        b    = b  + [np.random.rand(n[k],1)]
    return W,b

def nn_forward_propagation(X,W,b):
    a = [None]
    a[0]  = X
    m1 = len(W)
    for k in range(1,m1):
        zk = W[k].dot(a[k-1])+b[k]
        a = a + [1/(1+np.exp(-zk))]
    return a

def nn_backward_propagation(Y,a,W,b):
    m = len(W)-1
    N = Y.shape[1]
    dam = a[m]-Y
    dW = [None]
    db = [None]

    # Derivatives
    for k in range(1,m+1):
        dW   = dW + [np.zeros([W[k].shape[0],W[k].shape[1]])]
        db   = db + [np.zeros([b[k].shape[0],1])]
    for k in range(m,0,-1):
        if k == m:
            dak = dam
        ds    = np.multiply(a[k], 1-a[k]) 
        Gk    = np.multiply(dak,ds)
        dW[k] = np.matmul(Gk,a[k-1].transpose())/N
        db[k] = (np.sum(Gk,axis=1,keepdims=True))/N 
        dak   = np.matmul(W[k].transpose(),Gk)
    
    return dW,db

def nn_parameters_update(W,b,dW,db,alpha):
    m1 = len(W)

    # Updates
    for k in range(1,m1):
        b[k] = b[k] - alpha*db[k]
        W[k] = W[k] - alpha*dW[k]

    return W,b

def nn_loss_function(a,Y):
    m = len(a)-1
    N = Y.shape[1]
    Ys = a[m]
    dam = Ys-Y
    d2 = np.multiply(dam,dam)
    ds = np.sqrt(np.sum(d2,axis=0,keepdims=True))/N
    loss = np.sum(ds)
    return loss

