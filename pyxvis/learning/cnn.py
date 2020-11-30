import keras
import cv2
import os                   
import numpy                as np 
import matplotlib.pyplot    as plt
import tensorflow           as tf
from   keras.models         import Sequential, Model
from   keras.layers         import Dense, Dropout, Flatten, Activation, BatchNormalization # regularizers
from   keras.layers         import Conv2D, MaxPooling2D
from   keras.callbacks      import ModelCheckpoint, EarlyStopping
from   keras                import backend as K
from   keras.backend        import tensorflow_backend
from   sklearn.metrics      import confusion_matrix
from   pyxvis.io.plots      import show_confusion_matrix, plot_cnn_history
from   pyxvis.io.data       import load_cnn_patches, load_cnn_test_patches

# definition of the CNN architecture using arguments p, d and f:
# there are  n = len(p)  conv2D layers, and m = len(f) fully connected layers
# each conv2D layer has a mask of size p[i] x p[i] x d[i], i=0...n-1
# each conv2D layer has: convolution, ReLU, MaxPooling and Dropout (see details in sequentialCNN in cnn_utils.py)
# each fully connected layer as f[j] nodes, j = 0...m-1
# p             = [5, 3]   # Conv2D mask size
# d             = [4, 6]   # Conv2D channels
# f             = [8]      # fully connected

def CNN(patches_file,type_exec,p,d,f):

    # parameters
    model_file    = 1   # 1 store the best trained model, 0 store the last trained model
    epochs        = 100 # maximal number of epochs in training stage

    # execution type
    if     type_exec == 0: # training and testing
        print('Execution Type 0: Training and testing...')
        do_train      = 1  # 0 means no train, only evaluation,1 means train 
        only_test     = 0  # 1 loads only test data (eg. for sliding windows)
        ev_train      = 1  # accuracy on train data is computed
        ev_test       = 1  # accuracy on test data is computed
        do_conf_mtx   = 1  # compute confusion matrix on testing data
        train_predict = 0  # prediction output of train data is stored in train_predic.npy, 0 = no
        test_predict  = 0  # prediction output of test data is stored in test_predic.npy, 0 = no
        train_layer   = 0  # output layer of train data is stored in train_output.npy, 0 = no
        test_layer    = 0  # output layer of test data is stored in test_output.npy, 0 = no

    elif   type_exec == 1: # eval on testing only
        print('Execution Type 1: Evaluation on testing set only...')
        do_train      = 0  # 0 means no train, only evaluation,1 means train
        only_test     = 0  # 1 loads only test data (eg. for sliding windows)
        ev_train      = 0  # accuracy on train data is computed
        ev_test       = 1  # accuracy on test data is computed
        do_conf_mtx   = 1  # compute confusion matrix on testing data
        train_predict = 0  # prediction output of train data is stored in train_predic.npy, 0 = no
        test_predict  = 0  # prediction output of test data is stored in test_predic.npy, 0 = no
        train_layer   = 0  # output layer of train data is stored in train_output.npy, 0 = no
        test_layer    = 0  # output layer of test data is stored in test_output.npy, 0 = no

    elif   type_exec == 2: # eval on training and testing
        print('Execution Type 2: Evaluation on training and testing set...')
        do_train      = 0  # 0 means no train, only evaluation,1 means train
        only_test     = 0  # 1 loads only test data (eg. for sliding windows)
        ev_train      = 1  # accuracy on train data is computed
        ev_test       = 1  # accuracy on test data is computed
        do_conf_mtx   = 1  # compute confusion matrix on testing data
        train_predict = 0  # prediction output of train data is stored in train_predic.npy, 0 = no
        test_predict  = 0  # prediction output of test data is stored in test_predic.npy, 0 = no
        train_layer   = 0  # output layer of train data is stored in train_output.npy, 0 = no
        test_layer    = 0  # output layer of test data is stored in test_output.npy, 0 = no

    elif   type_exec == 3: # eval layer's output training and testing
        print('Execution Type 3: Computation of layer-output...')
        do_train      = 0  # number of epochs, 0 means no train, only evaluation
        only_test     = 0  # 1 loads only test data (eg. for sliding windows)
        ev_train      = 0  # accuracy on train data is computed
        ev_test       = 0  # accuracy on test data is computed
        do_conf_mtx   = 0  # compute confusion matrix on testing data
        train_predict = 0  # prediction output of train data is stored in train_predic.npy, 0 = no
        test_predict  = 0  # prediction output of test data is stored in test_predic.npy, 0 = no
        train_layer   = 1  # output layer of train data is stored in train_output.npy, 0 = no
        test_layer    = 1  # output layer of test data is stored in test_output.npy, 0 = no

    plot_curves   = 1  # plot loss and accuracy curves
    best_model    = 'cnn_best_model.h5'
    last_model    = 'cnn_last_model.h5'

    #size of parameters
    batch_size    = 128
    droprate      = 0.25

    # tensorflow configuration
    K.set_image_dim_ordering('th')
    print(K.image_data_format())
    config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

    # prepare callbacks
    callbacks    = defineCallBacks(best_model)

    # load patches
    if only_test == 1:
        X_test, Y_test, classes = load_cnn_test_patches(patches_file)
    else:
        X_train, Y_train, X_test, Y_test, classes = load_cnn_patches(patches_file)

    num_classes  = len(classes)
    input_shape  = (X_test.shape[1], X_test.shape[2],X_test.shape[3])

    # CNN architecture defintion
    model        = sequentialCNN(input_shape,droprate,num_classes,p,d,f)

    # training
    if do_train == 1 and only_test == 0:
        deleteWeights(best_model,last_model)
        history = model.fit(X_train, Y_train,
            batch_size      = batch_size,
            epochs          = epochs,
            verbose         = 1,
            validation_data = (X_test, Y_test),
            shuffle         = True,
            callbacks       = callbacks)
        model.save_weights(last_model)
        if plot_curves == 1:
            plot_cnn_history(history)

    # load trained model
    if model_file == 1: # best model
        print('loading best model from '+best_model+' ...')
        model.load_weights(best_model)
    else:               # last model
        print('loading last model from ' +last_model+' ...')
        model.load_weights(last_model)

    # print results in training/testing data
    print('results using '+patches_file+':')
    if ev_train == 1:
        evaluateCNN(model,X_train,Y_train,'training')
    if ev_test == 1:
        evaluateCNN(model,X_test,Y_test,'testing')

    # confusion matrix
    if do_conf_mtx == 1:
        computeconfusionMatrix(model,X_test,Y_test)

    # evaluation of layer outputs
    if test_layer > 0  or test_predict == 1 :
        evaluateLayer(model,K,X_test,'test',test_layer,test_predict)
    if train_layer > 0 or train_predict == 1:
        evaluateLayer(model,K,X_train,'train',train_layer,train_predict)


def sequentialCNN(input_shape,droprate,num_classes,p,d,f):

    n = len(p) # number of conv2D layers
    m = len(f) # number of fc layers

    #Start Neural Network
    model = Sequential()
  
    #convolution 1st layer
    model.add(Conv2D(d[0], kernel_size=(p[0],p[0]), padding="same",activation="relu",input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(droprate))
  
    #convolution layer i
    for i in range(1,n):
        # model.add(Conv2D(d[i], kernel_size=(p[i],p[i]), activation="relu",border_mode="same"))
        model.add(Conv2D(d[i], kernel_size=(p[i],p[i]), activation="relu",padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Dropout(droprate))

    #Fully connected layers
    model.add(Flatten())
    for j in range(m):
        model.add(Dense(f[j],use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(droprate))

    #Fully connected final layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss        = keras.losses.categorical_crossentropy,
                  optimizer   = keras.optimizers.RMSprop(),
                  metrics     = ['accuracy'])

    model.summary()

    return model

def deleteWeights(best_model,last_model):
    if os.path.exists(best_model):
        os.remove(best_model)
    if os.path.exists(last_model):
        os.remove(last_model)

def evaluateCNN(model,X,y,st):
    print('evaluating performance in '+st+' set ('+str(y.shape[0])+' samples)...')
    score   = model.evaluate(X,y,verbose=0)
    print(st+' loss:', score[0])
    print(st+' accuracy:', score[1])

def defineCallBacks(model_file):
    callbacks = [
        EarlyStopping(
            monitor        = 'val_acc', 
            patience       = 10,
            mode           = 'max',
            verbose        = 1),
        ModelCheckpoint(model_file,
            monitor        = 'val_acc', 
            save_best_only = True, 
            mode           = 'max',
            verbose        = 0)
    ]
    return callbacks

def sliding_window(image, stepSize, windowSize):
	for i in range(0, image.shape[0], stepSize):
		for j in range(0, image.shape[1], stepSize):
			yield (i, j, image[i:i+windowSize,j:j+windowSize])

def loadSliWinPatches(st_file):
    print('loading image '+ st_file +' for sliding windows...')
    image3 = cv2.imread(st_file)
    image  = image3[:,:,0]
    ROI3   = cv2.imread('ROI.png')
    ROIo   = ROI3[:,:,0]
    ROIo   = ROIo.astype('uint8')
    kernel = np.ones((11,11), np.uint8)
    ROI    = cv2.erode(ROIo, kernel, iterations=1)
    w      = 24
    q      = 12
    X      = np.zeros((100000,1,32,32))
    k      = -1
    for (i, j, window) in sliding_window(image, stepSize=6, windowSize=w):
        if (window.shape[0]==w) and (window.shape[1]==w) and (ROI[i+q,j+q]==1):
            im = cv2.resize(window[:,:], (32,32),interpolation = cv2.INTER_CUBIC)
            k = k+1
            X[k,:,:,:] = im
    print('saving X_test.npy with '+str(k) + ' patches...')
    X_test = X[0:k,:,:,:]/255.0
    Y_test = np.zeros((k,1), dtype=int)
    classes  = [0, 1]
    X_test = X_test.astype('float32')
    np.save('X_test',X_test)
    return X_test, Y_test, classes

def outSliWinPatches(st_file):
    print('computing output image for '+ st_file +' using sliding windows...')
    image3 = cv2.imread(st_file)
    image  = image3[:,:,0]
    ROI3   = cv2.imread('ROI.png')
    ROIo   = ROI3[:,:,0]
    ROIo   = ROIo.astype('uint8')
    kernel = np.ones((11,11), np.uint8)
    ROI = cv2.erode(ROIo, kernel, iterations=1)
    w      = 24
    q      = 12 # w/2
    y_predic = np.load('test_predict.npy')
    n = y_predic.shape[0]
    print(str(n)+' points will be evaluated...')
    image3[0,0,1] = 255 
    image3[0,1,1] = 255 
    image3[1,1,1] = 255 
    image3[1,0,1] = 255 
    t = 0
    k = -1
    for (i, j, window) in sliding_window(image, stepSize=6, windowSize=w):
        if (window.shape[0]==w) and (window.shape[1]==w) and (ROI[i+q,j+q]==1):
            k = k+1
            if k<n:
                if  y_predic[k,1] > 0.95:
                    # print(str(k) + ':' + str(y)+','+str(x)+str(y_predic[k,1]))
                    iq = i+q
                    jq = j+q
                    image3[iq  ,jq  ,0] = 255 
                    image3[iq+1,jq  ,0] = 255 
                    image3[iq  ,jq+1,0] = 255 
                    image3[iq+1,jq+1,0] = 255 
                    image3[iq  ,jq  ,1] = 0 
                    image3[iq+1,jq  ,1] = 0 
                    image3[iq  ,jq+1,1] = 0 
                    image3[iq+1,jq+1,1] = 0 
                    image3[iq  ,jq  ,2] = 0 
                    image3[iq+1,jq  ,2] = 0 
                    image3[iq  ,jq+1,2] = 0 
                    image3[iq+1,jq+1,2] = 0 
                    t = t+1
    print('storing output.png with ' + str(t) +' detected points...')
    cv2.imwrite('output.png',image3)
    return

def evaluateLayer(model,K,X,st,num_layer,test_predict):
    inp        = model.input                                           # input placeholder
    outputs    = [layer.output for layer in model.layers]              # all layer outputs
    functor    = K.function([inp, K.learning_phase()], outputs )       # evaluation function

    test       = X[0]
    test       = test.reshape(1,1,X.shape[2],X.shape[3])
    layer_outs = functor([test, 1.])
    x          = layer_outs[num_layer]
    n          = X.shape[0]
    m          = x.shape[1]

    if test_predict == 1:
        print('computing prediction output in ' +st +' set with '+str(n)+' samples...')
        y = model.predict(X)
        print('saving prediction in '+st+'_predict.npy ...')
        np.save(st+'_predict',y)

    if num_layer>0:
        d = np.zeros((n,m))
        print('computing output layer '+str(num_layer)+ ' in ' +st +' set with '+str(n)+' descriptors of '+str(m)+' elements...')
        for i in range(n):
            test       = X[i]
            test       = test.reshape(1,1,X.shape[2],X.shape[3])
            layer_outs = functor([test, 1.])
            d[i]       = layer_outs[num_layer]
        print('saving layer output in '+st+'_layer_'+str(num_layer)+'.npy ...')
        np.save(st+'_layer_'+str(num_layer),d)


def computeconfusionMatrix(model,X,y):
    Y_prediction = model.predict(X)
    Y_pred_classes = np.argmax(Y_prediction,axis = 1) # classes to one hot vectors 
    Y_true = np.argmax(y,axis = 1)                    # classes to one hot vectors
    show_confusion_matrix(Y_true, Y_pred_classes,'CNN')


