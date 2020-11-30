import numpy as np
from sklearn.metrics import accuracy_score
from pyxvis.learning.pretrained import prt_model, extract_prt_features
from pyxvis.io.gdxraydb import DatasetBase
from pyxvis.learning.classifiers import clf_model, define_classifier
from pyxvis.learning.classifiers import train_classifier, test_classifier
from pyxvis.learning.evaluation import precision_recall
from pyxvis.io.data import init_data, append_data
from pyxvis.io.plots import print_confusion

gdxray       = DatasetBase()
path         = gdxray.dataset_path + '/Baggages/' 
model_id     = 6 # 0 ResNet50, 1 VGG16, 2 VGG19, ... 6 MobileNet, ... 13 RCNN_ILSVRC13
output_layer = 2 # 0 Keras-Last, 1 ONNX-Last, 2 ONNX-Previous

# Classifiers to evaluate
ss_cl        = ['knn1','knn3','svm-lin','svm-rbf','nn']
(model,size,model_name) = prt_model(model_id,output_layer)
X49 = extract_prt_features(model_id,output_layer,model,size,model_name,path+'B0049/')
X50 = extract_prt_features(model_id,output_layer,model,size,model_name,path+'B0050/')
X51 = extract_prt_features(model_id,output_layer,model,size,model_name,path+'B0051/')
X78 = extract_prt_features(model_id,output_layer,model,size,model_name,path+'B0078/')
X79 = extract_prt_features(model_id,output_layer,model,size,model_name,path+'B0079/')
X80 = extract_prt_features(model_id,output_layer,model,size,model_name,path+'B0080/')
X81 = extract_prt_features(model_id,output_layer,model,size,model_name,path+'B0081/')
X82 = extract_prt_features(model_id,output_layer,model,size,model_name,path+'B0082/')

best_performance = 0 # initial value
for i in range(len(ss_cl)):
    cl_name = ss_cl[i]
    print('\nEvaluation of '+cl_name+' using '+model_name+'...')
    (Q_v,Q_t) = (0,0) # initial score Q values for validation and testing
    for j in range(4):
        if j==0:
            (c0,c1,c2,c3) = (1,0,0,0)
            st = 'Gun'
        elif j==1:
            (c0,c1,c2,c3) = (0,1,0,0)
            st = 'Shuriken'
        elif j==2:
            (c0,c1,c2,c3) = (0,0,1,0)
            st = 'Blade'
        elif j==3:
            (c0,c1,c2,c3) = (0,1,2,3)
            st = 'All'

        print('building dataset for '+st+' using ' + model_name +' ...')
        # Training data
        (X,d)   = init_data(X49[0:200],c0)             # Gun
        (X,d)   = append_data(X,d,X50[0:100,:],c1)     # Shuriken
        (X,d)   = append_data(X,d,X51[0:100,:],c2)     # Blade
        (X,d)   = append_data(X,d,X78[0:500,:],c3)     # Other
        # Validation data
        (Xv,dv) = init_data(X79[0:50,:],c0)            # Gun
        (Xv,dv) = append_data(X,d,X80[0:50,:],c1)      # Shuriken
        (Xv,dv) = append_data(X,d,X81[0:50,:],c2)      # Blade
        (Xv,dv) = append_data(X,d,X82[0:200,:],c3)     # Other
        # Testing data
        (Xt,dt) = init_data(X79[50:150],c0)            # Gun
        (Xt,dt) = append_data(X,d,X80[50:150,:],c1)    # Shuriken
        (Xt,dt) = append_data(X,d,X81[50:150,:],c2)    # Blade
        (Xt,dt) = append_data(X,d,X82[200:600,:],c3)   # Other

        print('training '+cl_name+' for '+st+' using ' + model_name +' ...')
        (name,params) = clf_model(cl_name)               # function name and parameters
        clf           = define_classifier([name,params]) # classifier definition
        clf           = train_classifier(clf,X,d)        # classifier training
        ds_v          = test_classifier(clf,Xv)          # clasification of validation
        ds_t          = test_classifier(clf,Xt)          # clasification of testing
        print('Results - ' + st + ' ('+cl_name+') for the detectors:')
        if j<3: # detection of three treat objects
            # performance on validation subset
            (pr_v,re_v) = precision_recall(dv,ds_v)
            Q_v         = Q_v + np.sqrt(pr_v*re_v)
            print(f'Pr_val   = {pr_v:.4f}')
            print(f'Re_val   = {re_v:.4f}')
            # performance on testing subset
            (pr_t,re_t) = precision_recall(dt,ds_t)
            Q_t         = Q_t + np.sqrt(pr_t*re_t)
            print(f'Pr_test  = {pr_t:.4f}')
            print(f'Re_test  = {re_t:.4f}')
        else: 
            # summary of three detections
            Q_v = Q_v/3         # score Q on validation
            print(f'Q_val    = {Q_v:.4f} of all detectors')
            Q_t = Q_t/3         # score Q on testing
            print(f'Q_test   = {Q_v:.4f} of all detectors')
            # four-class classification
            print('Results - ' + st + ' ('+cl_name+') for the 4-class classifier:')
            acc_v = accuracy_score(dt,ds_t)
            print(f'Acc_val  = {acc_v:.4f}')
            acc_t = accuracy_score(dv,ds_v)
            print(f'Acc_test = {acc_t:.4f}')
            print(f'Acc_t = {acc_t:.4f}')
            print_confusion(dt,ds_t)
    performance = (acc_v+Q_v)/2
    if performance>best_performance:
        print(f'performance = {performance:.4f} *** new max ***')
        best_performance = performance
        best_Q           = Q_t
        best_acc         = acc_t
        best_clf         = cl_name
print('Best result: classifier = '+best_clf)
print(f'                 Q_test = {best_Q:.4f}')
print(f'               acc_test = {best_acc:.4f}')
