import os
import numpy as np
import cv2
from keras.layers                    import Dense,GlobalAveragePooling2D
from keras.models                    import Model
from keras.callbacks                 import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image       import ImageDataGenerator
from keras.optimizers                import Adam, SGD
from keras.applications              import MobileNet, MobileNetV2, VGG16, VGG19
from keras.applications.resnet50     import ResNet50
from keras.applications.xception     import Xception
#from keras.applications.densenet     import DenseNet121
#from keras.applications.nasnet       import NASNetLarge, NASNetMobile
from keras.applications.inception_v3 import InceptionV3
from pyxvis.io.misc import dirfiles
from pyxvis.io.plots import plot_confusion

from sklearn.metrics import confusion_matrix, accuracy_score


def tfl_model(base_model):
    if base_model == 1: # MobileNet (87 layers)
        model_name = 'MobileNet'
        bmodel     = MobileNet(weights='imagenet',include_top=False) 
    elif base_model == 2: # InceptionV3 (311 layers)
        model_name = 'InceptionV3'
        bmodel     = InceptionV3(weights='imagenet',include_top=False) 
    elif base_model == 3: # VGG16 (16 layers)
        model_name = 'VGG16'
        bmodel     = VGG16(weights='imagenet',include_top=False) 
    elif base_model == 4: # VGG119 (22 layers)
        model_name = 'VGG19'
        bmodel     = VGG19(weights='imagenet',include_top=False) 
    elif base_model == 5: # ResNet50 (175 layers)
        model_name = 'ResNet50'
        bmodel     = ResNet50(weights='imagenet',include_top=False) 
    elif base_model == 6: # Xception (132 layers)
        model_name = 'Xception'
        bmodel     = Xception(weights='imagenet',include_top=False) 
    elif base_model == 7: # MobileNetV2 (155 layers)
        model_name = 'MobileNetV2'
        bmodel     = MobileNetV2(weights='imagenet',include_top=False) 
    #elif base_model == 8: # DenseNet121 (427 layers)
    #    model_name = 'DenseNet121'
    #    bmodel     = DenseNet121(weights='imagenet',include_top=False) 
    #elif base_model == 9: # NASNetMobile (769 layers)
    #    model_name = 'NASNetMobile'
    #    bmodel     = NASNetMobile(weights='imagenet',include_top=False) 
    #elif base_model == 10: # NASNetLarge (1039 layers)
    #    model_name = 'NASNetLarge'
    #    bmodel     = NASNetLarge(weights='imagenet',include_top=False) 
    else:
        print('error: base model '+str(base_model)+' not defined.')
    bmodel.summary()
    print('\nBase model '+ model_name + ' loaded with '+str(len(bmodel.layers)) + ' layers.')
    return bmodel


def tfl_define_model(bmodel,dense_layers,NumClasses):
    x = bmodel.output
    x = GlobalAveragePooling2D()(x)
    n = len(dense_layers)
    for i in range(n):
        layers_i = dense_layers[i]
        if layers_i > 0:
            x = Dense(layers_i,activation='relu')(x)

    preds = Dense(NumClasses,activation='softmax')(x) #final layer with softmax activation
    model = Model(inputs=bmodel.input,outputs=preds)
    return model


def generate_training_set(val_split, augmentation, batch_size, path_dataset, img_size):
    # augmentation is how aggressive will be the data augmentation/transformation, eg. 0.05
    if val_split > 0:
        image_generator = ImageDataGenerator(rescale=1. / 255,
                                                    rotation_range   = augmentation,
                                                    shear_range      = augmentation,
                                                    zoom_range       = augmentation,
                                                    cval             = augmentation,
                                                    horizontal_flip  = True,
                                                    vertical_flip    = True,
                                                    validation_split = val_split)

        print('\ndefining training subset from '+path_dataset+'/train...')
        train_set   = image_generator.flow_from_directory(batch_size  = batch_size,
                                                    directory   = path_dataset+'/train',
                                                    shuffle     = True,
                                                    target_size = img_size, 
                                                    subset      = "training",
                                                    class_mode  = 'categorical')

        print('\ndefining validation subset from '+path_dataset+'/train...')
        val_set     = image_generator.flow_from_directory(batch_size  = batch_size,
                                                    directory   = path_dataset+'/train',
                                                    shuffle     = True,
                                                    target_size = img_size, 
                                                    subset      = "validation",
                                                    class_mode  = 'categorical')
    else:
        train_generator = ImageDataGenerator(rescale=1. / 255,
                                                    rotation_range  = augmentation,
                                                    shear_range     = augmentation,
                                                    zoom_range      = augmentation,
                                                    cval            = augmentation,
                                                    horizontal_flip = True,
                                                    vertical_flip   = True)

        val_generator = ImageDataGenerator(rescale=1. / 255)

        print('\ndefining training subset from '+path_dataset+'/train...')
        train_set = train_generator.flow_from_directory(path_dataset+'/train',
                                                    target_size = img_size,
                                                    batch_size  = batch_size,
                                                    class_mode  = 'categorical')
        print('\ndefining validation subset from '+path_dataset+'/val...')
        val_set = val_generator.flow_from_directory(path_dataset+'/val',
                                                    target_size = img_size,
                                                    batch_size  = batch_size,
                                                    class_mode  = 'categorical')

    return train_set, val_set

def defineCallBacks(model_file):
    callbacks = [
        EarlyStopping(
            monitor        = 'val_acc', 
            patience       = 8,
            mode           = 'max',
            verbose        = 1),
        ModelCheckpoint(model_file,
            monitor        = 'val_acc', 
            save_best_only = True, 
            mode           = 'max',
            verbose        = 0)
    ]
    return callbacks

def deleteWeights(best_model,last_model):
    if os.path.exists(best_model):
        os.remove(best_model)
    if os.path.exists(last_model):
        os.remove(last_model)

def tfl_train(bmodel,model,opti_method,nb_layers,train_set,train_steps,val_set,val_steps,nb_epochs,path_dataset,nb_classes,img_size):

    best_model   = 'tfl_best_model.h5'
    last_model   = 'tfl_last_model.h5'
    top_model    = 'tfl_top_model.h5'

    nb = len(bmodel.layers)
    nm = len(model.layers)

    if nb_layers < 0:
        nb_layers = nb + nb_layers

    print('\nBase model has '+str(nb)+' layers, the first '+str(nb_layers)+' ones are frozen.')

    if nb_layers > nb:
        print('Error: frezzed layers exceeds the total number of layers.')

    accmax = 0

    for j in range(3):
    
        if nb_epochs[j] > 0:

            if j==0:
                st = 'After Training-1 (new layers)'
                print('\nTraining on only new layers (top '+str(nm-nb)+' layers): '+str(nb_epochs[j])+' epochs')
                for layer in bmodel.layers:
                    layer.trainable = False
            elif j==1:
                st = 'After Training-2 (not frozen + new layers)'
                print('\nTraining on not frozen layers and ew layers (top '+str(nm-nb_layers)+' layers): '+str(nb_epochs[j])+' epochs')
                for layer in model.layers[:nb_layers]:
                    layer.trainable = False
                for layer in model.layers[nb_layers:]:
                    layer.trainable = True
            elif j==2:
                st = 'After Training-3 (all layers)'
                print('\nTraining on all layers ('+str(nm)+' layers): '+str(nb_epochs[j])+' epochs')
                for layer in model.layers:
                    layer.trainable = True

            if opti_method == 1: # Adam
                model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
            else: # SGD
                model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

            callbacks    = defineCallBacks(best_model)
            deleteWeights(best_model,last_model)

            model.fit_generator(
                train_set,
                steps_per_epoch  = train_steps,
                validation_data  = val_set, 
                validation_steps = val_steps,
                epochs           = nb_epochs[j],
                callbacks        = callbacks
            )

            model.save_weights(last_model)
            print('loading best model from '+best_model+' ...')
            model.load_weights(best_model)
            print('\n'+st+':')
            C,acc = tfl_testing_accuracy(model,path_dataset,nb_classes,img_size)
            if acc>accmax:
                print('*** top model achieved '+st+' ***')
                jmax     = j
                accmax   = acc
                Cmax     = C
                topmodel = model
                print('saving top model: '+top_model+' ...')
                model.save_weights(top_model)
            plot_confusion(C,acc,st,0,nb_classes)
    print('Top model achieved after Training-'+str(jmax+1))
    print('Confusion matrix:')
    print(Cmax)
    accst = f'Acc = {acc:.4f}'    
    print(accst)
    return topmodel,Cmax,accmax


def tfl_classify_image(model,imgpath,img_size,show):
    
    img = cv2.imread(imgpath)
    img = cv2.resize(img,(img_size[0],img_size[1]))
    img = np.reshape(img,[1,img_size[0],img_size[1],3])/255
    x = model.predict(img)
    cl = np.argmax(x[0])
    if show == 1:
        print(imgpath+' > class '+str(cl) +': ' + '%6.4f' % x[0][cl])
    return cl


def tfl_testing_accuracy(model,path_dataset,nb_classes,img_size):
    print('\nEvaluation on testing dataset ('+str(nb_classes)+' classes):')
    ygt = np.ones([0,1]) # ground truth
    y   = np.ones([0,1]) # predictions
    nt  = 0
    for i in range(nb_classes):
        path_i = path_dataset+'/test/class_'+str(i)+'/'
        img_test_i = dirfiles(path_i,'*.jpg')
        n_i    = len(img_test_i)
        nt     = nt + n_i
        ygt_i  = i*np.ones([n_i,1])
        ygt    = np.concatenate((ygt, ygt_i), axis=0)
        y_i    = np.zeros([n_i,1])
        print('   > class '+str(i)+'/'+str(nb_classes-1)+': evaluating model on '+str(n_i)+' images of '+path_i+' ...')
        for j in range(n_i):
            img_path = img_test_i[j]
            y_i[j]   = tfl_classify_image(model,path_i+img_path,img_size,0)
        y  = np.concatenate((y, y_i), axis=0)

    print(str(nt)+' test images evaluated.')
 
    C   = confusion_matrix(ygt, y) 
    acc = accuracy_score(ygt,y)

    return C, acc



