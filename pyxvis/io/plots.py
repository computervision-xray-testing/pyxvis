import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from seaborn import heatmap
from mpl_toolkits.mplot3d import Axes3D
# import os, fnmatch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def imshow(image):
    pil_image = Image.fromarray(image)
    pil_image.show()

def get_image(path, show=False):
    img = cv2.imread(path)
    if show:
        imshow(img)
    return img

def accuracy(Ys,Y,st):
    print(st)
    if Y.shape[1]>Y.shape[0]:
        Y = Y.transpose()
        Ys = Ys.transpose()
    if Y.shape[1]>1:
        d   = np.argmax(Y,axis = 1)
        ds  = np.argmax(Ys,axis = 1)
    else:
        d  = Y
        ds = Ys
    C   = confusion_matrix(d,ds) 
    acc = accuracy_score(d,ds)
    print('Confusion Matrix:')
    print(C)
    print('Accuracy = '+str(acc))
    print()
    nm = C.shape[0]
    plt.figure(figsize=(7,5))
    heatmap(C, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlim(0,nm)
    plt.ylim(nm,0)
    plt.title('Confusion Matrix '+st,fontsize=14)
    plt.show()
    return acc

def get_ellipse_path(params):
    cx    = params[0]
    cy    = params[1]
    a     = params[2]
    b     = params[3]
    alpha = params[4]
    ell   = Ellipse((cx,cy),a*2.,b*2.,alpha)
    coord = ell.get_verts()
    xs    = coord[:,0]
    ys    = coord[:,1]
    return xs,ys


def plot_ellipses(ellipse_list):
    '''Plot a list of ellipses'''
    fig=plt.figure()
    ax2=fig.add_subplot(111)
    n = len(ellipse_list)
    for i in range(n):
        cii = ellipse_list[i]
        x=cii[:,0]
        y=cii[:,1]
        if i==0:
            ax2.plot(x,y,'.')
        else:
            ax2.plot(x,y,'-')
    plt.show()

def plot_ellipses_image(I,x,y=0):
    if y==0:
        x,y  = get_ellipse_path(x)
    fig=plt.figure()
    ax2=fig.add_subplot(111)
    implot = plt.imshow(I,cmap='gray')    
    ax2.plot(x,y,'-',color='Red')
    plt.show()

def plot_confusion(confusion_mtx,acc,st,dmin,dmax,p100=True):
    C = confusion_mtx.copy()
    print('Confusion matrix - '+st+':')
    print(C)
    accst = f'Acc = {acc:.4f}'    
    print(accst)
    plt.figure(figsize=(10,8))
    n = C.shape[0]
    if p100:
        for i in range(n):
            si = np.sum(C[i,:])
            C[i,:] = C[i,:]/si*100
        heatmap(C, annot=True, fmt="d",cmap="YlGnBu",vmin=0, vmax=100)
        st = 'Confusion Matrix [%] - '+st+': '+accst
    else:
        heatmap(C, annot=True, fmt="d",cmap="YlGnBu")
        st = 'Confusion Matrix - '+st+': '+accst
    plt.xlim(dmin, dmax)
    plt.ylim(dmax, dmin)
    plt.title(st,fontsize=14)
    plt.show()



def show_confusion_matrix(Yt,Ys,st,categorical=0):
    # Yt : ground truth
    # Ys : prediction
    if categorical:
        if Yt.shape[1]>Yt.shape[0]:
            Yt = Yt.transpose()
            Ys = Ys.transpose()
        dt  = np.argmax(Yt,axis = 1)
        ds  = np.argmax(Ys,axis = 1)
    else:
        dt = Yt
        ds = Ys
    confusion_mtx = confusion_matrix(dt,ds) 
    acc = accuracy_score(dt,ds)
    dmin = np.min(dt)
    if np.min(dt)==1:
        dmin = dmin-1
        dmax = np.max(dt)
    else:
        dmax = np.max(dt)+1
        dmin = 0        
    plot_confusion(confusion_mtx,acc,st,dmin,dmax)    






def plot_confusion_matrix(dt,ds,st):
    print('computing confusion matrix...')
    print('WARNING: use show_confusion_matrix!')
    confusion_mtx = confusion_matrix(dt,ds) 
    print(confusion_mtx)
    plt.figure(figsize=(10,8))
    n = confusion_mtx.shape[0]
    for i in range(n):
        si = np.sum(confusion_mtx[i,:])
        confusion_mtx[i,:] = confusion_mtx[i,:]/si*100
    heatmap(confusion_mtx, annot=True, fmt="d",cmap="YlGnBu",vmin=0, vmax=100)
    # dmin = np.min(dt)
    if np.min(dt)==1:
        dmax = np.max(dt)
    else:
        dmax = np.max(dt)+1
        dmin = 0        
    plt.xlim(dmin, dmax)
    plt.ylim(dmax, dmin)
    acc = accuracy_score(dt,ds)    
    accst = f'Acc = {acc:.4f}' 
    print('Accuracy = '+accst)   
    plt.title('Confusion Matrix [%] - '+st+': '+accst,fontsize=14)
    plt.show()

def print_confusion(dt,ds):
    # dt - true, ds - predicted
    C   = confusion_matrix(dt,ds) 
    print('Confusion Matrix:')
    print(C)
    acc = accuracy_score(dt,ds) 
    print('Accuracy = '+str(acc))

def plot_decision_lines(clf,X,show=0,decisionline=1):
    # based on example of https://scikit-learn.org
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    if decisionline == 1:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Reds, alpha=0.8)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    if show==1:
        plt.show()

def plot_features(X,d,st,show=1):
    dmin = int(np.min(d)) 
    dmax = int(np.max(d)) 
    for j in range(dmin,dmax+1):
        plt.scatter(X[d==j,0],X[d==j,1],label='Class '+str(j),cmap=plt.cm.autumn,s=17)
    plt.grid(True)
    plt.legend()
    plt.xlabel('$x_1$',fontsize=14)
    plt.ylabel('$x_2$',fontsize=14)
    plt.title('Feature Space - '+st,fontsize=14)
    if show==1:
        plt.show()

def plot_features2(X,d,st,show=1):
    # based on example of https://scikit-learn.org
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    print(X.shape)
    plt.scatter(X[:, 0], X[:, 1], c=d, cmap=plt.cm.autumn)
    plt.xlabel('$x_1$',fontsize=14)
    plt.ylabel('$x_2$',fontsize=14)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.title('Feature Space - '+st,fontsize=14)
    #plt.grid(True)
    #plt.legend()
    if show == 1:
        plt.show()

def plot_features3(X,d,st,show=1,view=(30,60)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=d, cmap=plt.cm.autumn)
    ax.set_xlabel('$x_1$',fontsize=14)
    ax.set_ylabel('$x_2$',fontsize=14)
    ax.set_zlabel('$x_3$',fontsize=14)
    plt.title('Feature Space - '+st,fontsize=14)
    ax.view_init(view[0],view[1])
    # plt.axis('off')
    # plt.grid(b=None)
    if show == 1:
        plt.show()

def plot_features_y(X,Y,st):

    if X.shape[1]<X.shape[0]:
        X = X.transpose()
    if Y.shape[1]<Y.shape[0]:
        Y = Y.transpose()
    if Y.shape[0]>1:
        d  = np.argmax(Y,axis = 0)
    else:
        d = Y
    K = np.max(d)+1
    color = ['blue', 'orange', 'green', 'red']
    for j in range(K):
        # plt.scatter(X[d==j,0],X[d==j,1],label='Class '+str(j),color='tab:'+color[j],s=1)
        plt.scatter(X[0,d==j],X[1,d==j],label='Class '+str(j),color='tab:'+color[j],s=17)

    plt.grid(True)
    plt.legend()
    plt.xlabel('$x_1$',fontsize=14)
    plt.ylabel('$x_2$',fontsize=14)
    plt.xticks(())
    plt.yticks(())
    plt.title('Feature Space - '+st,fontsize=14)
    plt.show()



def show_clf_results(clf,X,d,Xt,dt,d0,ds,st,decisionline=1):
    gs = gridspec.GridSpec(1, 2)
    fig = plt.figure(figsize=(18, 6))
    print('Training:')
    acc = accuracy_score(d,d0) 
    accst = f'Acc = {acc:.4f}'    
    ax = plt.subplot(gs[0,0])
    print_confusion(d,d0)                             # confusion matrix in training
    plot_decision_lines(clf,X,0,decisionline)          # decision lines
    plot_features(X,d,st+' - Training: '+accst,0)     # feature space in training   
    ax = plt.subplot(gs[0,1])
    print('Testing:')
    acc = accuracy_score(ds,dt) 
    accst = f'Acc = {acc:.4f}'    
    print_confusion(dt,ds)                            # confusion matrix in testing
    plot_decision_lines(clf,X,0,decisionline)          # decision lines
    plot_features(Xt,dt,st+' - Testing: '+accst,1)    # feature space in testing
    
def plot_ROC(fpr,tpr,label='ROC',auc=0,seq=[0, 1]):
    k = seq[0]
    n = seq[1]

    if auc>0:
        label = label + f': AUC = {auc:.4f}' 

    plt.plot(fpr, tpr,  label=label)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend()
    if k==n-1:
        plt.show()

def plot_precision_recall(precision,recall,label='precision/recall',ap=0,seq=[0, 1]):
    k = seq[0]
    n = seq[1]

    if ap>0:
        label = label + f': AP = {ap:.4f}' 

    plt.plot(recall, precision, label=label)
    
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision/Recall Curve')
    plt.grid(True)
    plt.legend()
    if k==n-1:
        plt.show()



def plot_cnn_history(history):
    # loss curves
    print('displaying loss and accuracy curves...')
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.show()

    # accuracy curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.show()

def plot_loss(loss_train):
    plt.figure(figsize=[8,6])
    plt.plot(loss_train,'r',linewidth=1.5)
    # plt.plot(loss_val,'b',linewidth=1.5)
    # plt.legend(['Training loss', 'Validation Loss'],fontsize=14)
    plt.xlabel('Epochs ',fontsize=14)
    plt.ylabel('Loss',fontsize=14)
    plt.title('Training Loss',fontsize=14)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.xlim(0,xmax)
    plt.ylim(0,ymax)
    plt.grid(True)
    plt.show()
