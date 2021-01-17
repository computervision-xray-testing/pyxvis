import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical


def load_features(prefix, full=0, categorical=0):
    """
    Loads feature from path.

    Args:
        prefix (string): source path of the features data.
        full (int):
        categorical (int):

    Returns:
        X (np.array): the set of features.
        Y, d (np.array): the vector of labels.
    """
    X = np.load(prefix + '_Xtrain.npy')             # load training samples
    Xt = np.load(prefix + '_Xtest.npy')             # load testing samples
    d = np.ravel(np.load(prefix + '_dtrain.npy'))   # load training labels
    dt = np.ravel(np.load(prefix + '_dtest.npy'))   # load testing labels

    if full == 0:
        print('Training data: ' + str(X.shape[0]) + ' samples with ' + str(X.shape[1]) + ' features')
        print(' Testing data: ' + str(Xt.shape[0]) + ' samples with ' + str(Xt.shape[1]) + ' features')
        print('      Classes: ' + str(int(np.min(d))) + '...' + str(int(np.max(d))))

        if categorical == 1:
            dmin = np.min(d)
            Ytrain = np.transpose(to_categorical(d-dmin))
            Ytest  = np.transpose(to_categorical(dt-dmin))
            Xtrain = np.transpose(X)
            Xtest  = np.transpose(Xt)
            return Xtrain,Ytrain,Xtest,Ytest
        else:
            return X,d,Xt,dt
    else:
        X = np.concatenate((X,Xt),axis=0)
        d = np.concatenate((d,dt),axis=0)
        print('Available data: '+str(X.shape[0]) +' samples with '+str(X.shape[1]) +' features')
        print('       Classes: '+str(int(np.min(d)))+'...'+str(int(np.max(d))))
        if categorical==1:
            dmin = np.min(d)
            Y = np.transpose(to_categorical(d-dmin))
            X = np.transpose(X)
            return X,Y
        else:
            return X,d

def save_features(X,d,Xt,dt,st):
    print('Training data (X,d)  : '+str(X.shape[0]) +' samples with '+str(X.shape[1]) +' features')
    print(' Testing data (Xt,dt): '+str(Xt.shape[0])+' samples with '+str(Xt.shape[1])+' features')
    print('      Classes: '+str(int(np.min(d)))+'...'+str(int(np.max(d))))
    print('saving training data (X,d) and testing data (Xt,dt)...')
    print('... training features in ' +st+'_Xtrain.npy')
    np.save(st+'_Xtrain',X)
    print('... training labels   in ' +st+'_dtrain.npy')
    np.save(st+'_dtrain',d)
    print('... testing  features in ' +st+'_Xest.npy')
    np.save(st+'_Xtest',Xt)
    print('... testing  labels   in ' +st+'_dest.npy')
    np.save(st+'_dtest',dt)



def load_cnn_patches(st_file):
    # patches are of 32 x 32 pixels (uint8)
    # classes are two: 0 and 1
    # X_train and X_test are uint8 4D matrices of size N_train x 1 x 32 x 32 and N_test x 1 x 32 x 32 respectively
    # Y_train and Y_test are uint8 column vectors with 0 and 1 of size N_train x 1 and N_test x 1 respectively
    print('loading training/testing data from '+ st_file +' ...')
    if st_file[-4:]=='.mat':
        data              = loadmat(st_file)
        X_train  = data['X_train']
        Y_train  = data['Y_train']
        X_test   = data['X_test']
        Y_test   = data['Y_test']
    else:
        X_train  = np.load(st_file+'_Xtrain.npy')
        Y_train  = np.load(st_file+'_Ytrain.npy')
        X_test   = np.load(st_file+'_Xtest.npy')
        Y_test   = np.load(st_file+'_Ytest.npy')
    
    Y_train  = to_categorical(Y_train)
    print('X train size: {}'.format(X_train.shape))
    print('y train size: {}'.format(Y_train.shape))

    Y_test   = to_categorical(Y_test)
    print('X test  size: {}'.format(X_test.shape))
    print('y test  size: {}'.format(Y_test.shape))

    classes  = [0, 1]

    return X_train, Y_train, X_test, Y_test, classes

def load_cnn_test_patches(st_file):
    print('loading testing data from '+ st_file +' ...')
    if st_file[-4:]=='.mat':
        data              = loadmat(st_file)
        X_test   = data['X_test']
        Y_test   = data['Y_test']
    else:
        X_test   = np.load(st_file+'_Xtest.npy')
        Y_test   = np.load(st_file+'_Ytest.npy')
    Y_test   = to_categorical(Y_test)
    print('X test  size: {}'.format(X_test.shape))
    print('y test  size: {}'.format(Y_test.shape))

    classes  = [0, 1]

    return X_test, Y_test, classes

def init_data(X,k):
    d = k*np.ones([X.shape[0],],dtype=int)
    return X,d

def append_data(X,d,Xa,ja):
    (Xa,da) = init_data(Xa,ja)
    X = np.concatenate((X,Xa),axis=0)
    d = np.concatenate((d,da),axis=0)
    return X,d

