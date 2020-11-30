# Pre-trained models
# This code can extract the embeddings of many pre-trained models. 
# The idea is to extract one embedding for a set of images 
# 
# variables: model_id output_layer
# 
# model_id can be (in parenthesis the available output_layer)
# 
#      0 : ResNet50 (0,1,2)
#      1 : VGG16 (0,1,2)        
#      2 : VGG19 (0,1,2)
#      3 : InceptionV3 (0)
#      4 : InceptionResNetV2 (0)
#      5 : Xception (0)
#      6 : MobileNet (1,2)
#      7 : SqueezeNet (1,2)        
#      8 : AlexNet (1,2)        
#      9 : GoogleNet (1,2)        
#     10 : ShuffleNet (1,2)        
#     11 : DenseNet121 (1,2)        
#     12 : ZfNet512 (1,2)        
#     13 : RCNN_ILSVRC13 (1,2)     
# 
# output_layer can be
# 
#      0 : Keras, layer before softmax
#      1 : ONNX,  last layer (after softmax) with 1000 features
#      2 : ONNX,  layer before softmax 
# 
# Example 1: Extraction of ResNet50 using Keras
# variables: model_id = 0, output_layer = 0
# 
# Example 2: Extraction of AlexNet using ONNX, layer before softmax 
# variables: model_id = 8, output_layer = 2

import os, fnmatch
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt

from mxnet.gluon.data.vision import transforms
from mxnet.contrib.onnx.onnx2mx.import_model import import_model


from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as proc0

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as proc1

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as proc2

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as proc3

#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.inception_resnet_v2 import preprocess_input as proc4

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as proc5

from keras.preprocessing import image

from sklearn.metrics import confusion_matrix

from pyxvis.io.misc import dirfiles,num2fixstr


def prt_model(model_id,output_layer):
    # ResNet50

    print('loading model...')
    # 0 ResNet50
    if model_id == 0: 
        if output_layer == 0: # Keras
            model_name = 'ResNet50_Keras'
            model = ResNet50(weights='imagenet', include_top=False)
        img_size = 224
        #img_size = 30
        model_path = '../models/resnet50.onnx'
        datain = 'gpu_0/data_0'
        if output_layer == 1: # ONNX - Last Layer
            model_name = 'ResNet50_ONNX_L'
            layerout = 'last'
        elif output_layer == 2: # ONNX - Previous Layer
            model_name = 'ResNet50_ONNX_P'
            layerout = 'flatten0_output'
    
    # 1 VGG16        
    elif model_id == 1: 
        if output_layer == 0: # Keras
            model_name = 'VGG16_Keras'
            model = VGG16(weights='imagenet', include_top=False)
        #img_size = 224
        img_size = 60
        model_path = '../models/vgg16.onnx'
        datain = 'data'
        if output_layer == 1: # ONNX - Last Layer
            model_name = 'VGG16_ONNX_L'
            layerout = 'last'
        elif output_layer == 2: # ONNX - Previous Layer
            model_name = 'VGG16_ONNX_P'
            layerout = 'flatten2_output'

    # 2 VGG19
    elif model_id == 2: 
        if output_layer == 0: # Keras
            model_name = 'VGG19_Keras'
            model = VGG19(weights='imagenet', include_top=False)
        #img_size = 224
        img_size = 60
        model_path = '../models/vgg19.onnx'
        datain = 'data_0'
        if output_layer == 1: # ONNX - Last Layer
            model_name = 'VGG19_ONNX_L'
            layerout = 'last'
        elif output_layer == 2: # ONNX - Previous Layer
            model_name = 'VGG19_ONNX_P'
            layerout = 'flatten2_output'

    # 3 InceptionV3
    elif model_id == 3: 
        if output_layer == 0: # Keras
            model_name = 'InceptionV3_Keras'
            model = InceptionV3(weights='imagenet', include_top=False)
        #img_size = 224
        img_size = 90

    # 4 InceptionResNetV2
    #elif model_id == 4: 
    #    if output_layer == 0: # Keras
    #        model_name = 'InceptionResNetV2_Keras'
    #        model = InceptionResNetV2(weights='imagenet', include_top=False)
    #    #img_size = 224
    #    img_size = 90

    # 5 Xception
    elif model_id == 5: 
        if output_layer == 0: # Keras
            model_name = 'Xception_Keras'
            model = Xception(weights='imagenet', include_top=False)
        #img_size = 224
        img_size = 30
    
    # 6 MobileNet
    elif model_id == 6: 
        model_path = '../models/mobilenet.onnx'
        datain = 'data'
        if output_layer == 1: # ONNX - Last Layer
            model_name = 'MobileNet_ONNX_L'
            layerout = 'last'
        elif output_layer == 2: # ONNX - Previous Layer
            model_name = 'MobileNet_ONNX_P'
            layerout = 'mobilenetv20_features_pool0_fwd_output' # not good

    # 7 SqueezeNet        
    elif model_id == 7:
        model_path = '../models/squeezenet.onnx'
        datain = 'data'
        if output_layer == 1: # ONNX - Last Layer
            model_name = 'SqueezeNet_ONNX_L'
            layerout = 'last'
        elif output_layer == 2: # ONNX - Previous Layer
            model_name = 'SqueezeNet_ONNX_P'
            layerout = 'pooling3_output' # not good

    # 8 AlexNet        
    elif model_id == 8:
        model_path = '../models/alexnet.onnx'
        datain = 'data_0'
        if output_layer == 1: # ONNX - Last Layer
            model_name = 'AlexNet_ONNX_L'
            layerout = 'last'
        elif output_layer == 2: # ONNX - Previous Layer
            model_name = 'AlexNet_ONNX_P'
            layerout = 'flatten2_output'

    # 9 GoogleNet        
    elif model_id == 9:
        model_path = '../models/googlenet.onnx'
        datain = 'data_0'
        if output_layer == 1: # ONNX - Last Layer
            model_name = 'GoogleNet_ONNX_L'
            layerout = 'last'
        elif output_layer == 2: # ONNX - Previous Layer
            model_name = 'GoogleNet_ONNX_P'
            layerout = 'flatten0_output' # not good

    # 10 ShuffleNet        
    elif model_id == 10:
        model_path = '../models/shufflenet.onnx'
        datain = 'gpu_0/data_0'
        if output_layer == 1: # ONNX - Last Layer
            model_name = 'ShuffleNet_ONNX_L'
            layerout = 'last'
        elif output_layer == 2: # ONNX - Previous Layer
            model_name = 'ShuffleNet_ONNX_P'
            layerout = 'flatten0_output' # not good

    # 11 DenseNet121        
    elif model_id == 11:
        model_path = '../models/densenet121.onnx'
        datain = 'data_0'
        if output_layer == 1: # ONNX - Last Layer
            model_name = 'DenseNet121_ONNX_L'
            layerout = 'last'
        elif output_layer == 2: # ONNX - Previous Layer
            model_name = 'DenseNet121_ONNX_P'
            layerout = 'pad124_output' # not good

    # 12 ZfNet512        
    elif model_id == 12:
        model_path = '../models/zfnet512.onnx'
        datain = 'gpu_0/data_0'
        if output_layer == 1: # ONNX - Last Layer
            model_name = 'ZfNet512_ONNX_L'
            layerout = 'last'
        elif output_layer == 2: # ONNX - Previous Layer
            model_name = 'ZfNet512_ONNX_P'
            layerout = 'flatten2_output' # not good

    # 13 RCNN_ILSVRC13     
    elif model_id == 13:
        model_path = '../models/rcnn_ilsvrc13.onnx'
        datain = 'data_0'
        if output_layer == 1: # ONNX - Last Layer
            model_name = 'RCNN_ILSVRC13_ONNX_L'
            layerout = 'last'
        elif output_layer == 2: # ONNX - Previous Layer
            model_name = 'RCNN_ILSVRC13_ONNX_P'
            layerout = 'flatten2_output' # not good



    if output_layer > 0:  
        print('loading model '+model_path+'...')
        sym, arg_params, aux_params = import_model(model_path)
        if len(mx.test_utils.list_gpus())==0:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(0)
        
        all_layers = sym.get_internals()
        print(all_layers.list_outputs())
        if layerout == 'last':
            sym3 = sym
        else:
            sym3 = all_layers[layerout]
        model = mx.mod.Module(symbol=sym3, context=ctx, label_names=None, data_names=[datain] )
        image_size = (224,224)
        img_size = image_size[1]
        model.bind(data_shapes=[(datain, (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)

    print('model '+model_name+' loaded.')
    return model,img_size,model_name

def get_image(path, show=False):
    img = mx.image.imread(path)
    if img is None:
        return None
    if show:
        plt.imshow(img.asnumpy())
        plt.axis('off')
    return img

def preprocess(img):   
    transform_fn = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0)
    return img

def get_features(mod,path):
    img = get_image(path, show=True)
    img = preprocess(img)
    input_blob = img
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    mod.forward(db, is_train=False)
    embedding = mod.get_outputs()[0].asnumpy()
    return embedding

def extract_prt_features_img(model_id,output_layer,model,nn,st):
    if output_layer<1:
        img = image.load_img(st, target_size=(nn,nn))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = eval('proc'+str(model_id)+'(x)')
        features = model.predict(x)
    else:
        features = get_features(model,st)
    return features


def extract_prt_features(model_id,output_layer,model,img_size,model_name,dirpath,st=''):
    
    img_names = dirfiles(dirpath,'*.png')
    print(output_layer)
    n = len(img_names)
    if output_layer < 1:
        kch = 3
    else:
        kch = 1
    print(kch)
    for i in range(n):
        img_path = img_names[i]
        print(model_name+'_'+st +': '+ num2fixstr(i,4)+'/'+num2fixstr(n,4)+ ': reading '+img_path+'...')
        features = extract_prt_features_img(model_id,output_layer,model,img_size,dirpath+img_path)
        if i==0:
            m = features.shape[kch]
            data = np.zeros((n,m))
            print(data.shape)
            print('size of extracted features:')
            print(features.shape)
        # data[i]    = features[:][0][0] 
        x = np.reshape(features, m, order='F')
        # print(x.shape)    
        data[i]    = x
        # data[i]    = features
    return data






