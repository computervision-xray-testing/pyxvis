import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from pyxvis.features.selection import fse_model, fsel
from pyxvis.learning.classifiers import define_classifier, train_test_classifier, clf_model


def hold_out(bcl,X,d,Xt,dt):
    ds,clf  = train_test_classifier(bcl,X,d,Xt)
    acc     = accuracy_score(dt,ds)     
    return ds,acc,clf

def cross_validation(bcl,X,d,folds):
    clf    = define_classifier(bcl)
    scores = cross_val_score(clf, X, d, cv=folds)
    acc    = np.mean(scores)
    return acc

def leave_one_out(bcl,X,d):
    classes   = np.sort(np.unique(d))
    dd        = [d[d == di] for di in classes]
    nc        = len(classes)
    nd        = np.zeros((nc,))
    for i in range(nc):
        nd[i] = dd[i].shape[0]
    folds     = int(np.max(nd))
    acc       = cross_validation(bcl,X,d,folds)
    return acc

def best_features_classifier(ss_fs,ff,ss_cl,X,d,Xt,dt,stit,folds=5):
    ncl      = len(ss_cl)           # number of classifiers
    nfs      = len(ss_fs)           # number of feature selectors
    nfk      = len(ff)              # number of feature selectors
    accbest  = 0                    # best achieved accuracy
    acctbest = 0                    # best achieved accuracy
    yacc     = np.zeros((1000,))
    yacct    = np.zeros((1000,))
    nt       = np.zeros((1000,))
    clfs     = [None] * 1000
    fses     = [None] * 1000
    t        = 1

    for k in range(nfk):
        nfik = ff[k]
        for i in range(nfs): 
            fs  = ss_fs[i]
            print('analyzing feature felector: sfs+'+fs+' with '+str(nfik)+' features...')
            (fsname,fspar) = fse_model(fs)
            ss             = fsel([fsname,fspar],X,d,nfik,cv = folds, show = 1)
            Xf             = X[:,ss]
            Xft            = Xt[:,ss]
            for j in range(ncl):
                cl = ss_cl[j]
                # print('analyzing classifier '+cl+'...')
                (clname,clpar) = clf_model(cl)
                acc            = cross_validation([clname,clpar],Xf,d,folds)
                _,acct,_       = hold_out([clname,clpar],Xf,d,Xft,dt)
                stn = f'{nfik:2d}' 
                stt = 'feat='+stn+', fsel='+fs+', clf='+cl
                stb = ''
                sti = '(iter = '+str(t)+')'
                if acc>accbest:
                    accbest = acc
                    stb = sti + ' *** new train max found ***'
                    yacc[t] = acc

                    yacct[t] = acct
                    nt[t] = nfik
                    clfs[t] = cl
                    fses[t] = fs
                    if acct > acctbest:
                        stb = sti + ' *** new train/test max found ***'
                        tbest    = t
                        acctbest = acct
                        accbest0 = acc
                        fsnbest  = fsname
                        fspbest  = fspar
                        fsbest   = fs
                        clbest   = cl
                        clnbest  = clname
                        clpbest  = clpar
                        nbest    = nfik
                        ssbest   = ss

                    t        = t + 1
                print(f'{stt:38s} > acc = {acc:.4f}/{acct:.4f}  <best acc = {accbest:.4f}/{acctbest:.4f}> (Train/Test) '+stb)

    print('---------------------------------------------------------------------------')
    print('       Best iteration: '+str(tbest)+' (maximum of testing accuracy)')
    print('     Feature Selector: ' +fsbest+' with '+str(nbest)+' features')
    print('                     : ('+fsnbest+', '+fspbest+')')
    print('           Classifier: ' +clbest)
    print('                     : ('+clnbest+', '+clpbest+') CrossVal with '+str(folds)+' folds')
    print(f'         Training-Acc: {accbest0:.4f}  ')
    print(f'          Testing-Acc: {acctbest:.4f}  ')
    y1 = yacc[0:t]
    y2 = yacct[0:t]
    x = range(t)
    
    # fig = plt.figure()
    fig = plt.figure(figsize=(t+2,5))
    ax = fig.add_subplot(111)
    plt.plot(x[1:t], y1[1:t], marker='o', markerfacecolor='blue', markersize=6, label = 'Training') 
    plt.plot(x[1:t], y2[1:t], marker='o', markerfacecolor='red', markersize=6, label = 'Testing') 
    plt.xlabel('iterations') 
    plt.ylabel('accuracy') 
    plt.title(stit) 
    yf = 0.405
    dy = 0.05
    dd = 0.02
    fts = 10
    ax.text(dd, yf, 'features:', fontsize=fts)
    ax.text(dd, yf-dy, 'f-selector:', fontsize=fts)
    ax.text(dd, yf-dy*2, 'classifier:', fontsize=fts)
    ax.text(dd, yf-dy*3, 'acc-train:', fontsize=fts)
    ax.text(dd, yf-dy*4, 'acc-test:', fontsize=fts)
    for i in range(1,t):
        col = 'black'
        if i==tbest:
            col = 'green'
        di = i+dd
        ax.text(di, yf, str(int(nt[i])), fontsize=fts,color=col)
        ax.text(di, yf-dy, fses[i], fontsize=fts,color=col)
        ax.text(di, yf-dy*2, clfs[i], fontsize=fts,color=col)
        accs = f'{y1[i]:.4f}'    
        accts = f'{y2[i]:.4f}'    
        ax.text(di, yf-dy*3, accs,  fontsize=fts,color=col)
        ax.text(di, yf-dy*4, accts, fontsize=fts,color=col)
    ax.text(tbest-0.5, yf+0.2, 'best test accuracy', fontsize=fts,color='green')
    ax.annotate(' ',xy=(tbest, acctbest-0.05), xytext=(tbest, yf+0.2),
            arrowprops=dict(facecolor='green', shrink=0.05))
    ax.plot(tbest,acctbest, 'o')
    plt.xticks(range(t+1))    
    plt.grid(True)
    plt.legend()
    plt.xlim(0, t)
    plt.ylim(0, 1.1)
    plt.show() 
    return clbest, ssbest

def precision_recall(dt,ds):
    # dt - true, ds - predicted
    #         0   1
    #     0  TN  FP
    #     1  FN  TP
    # pr = TP/(TP+FP)
    # re = TP/(TP+FN)

    C = confusion_matrix(dt,ds)

    pr = C[1,1]/(C[1,1]+C[0,1])
    re = C[1,1]/(C[1,1]+C[1,0])
    return pr,re

