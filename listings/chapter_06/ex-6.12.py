from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from pyxvis.learning.classifiers import clf_model,define_classifier,train_classifier
from pyxvis.io.plots import plot_features, plot_ROC, plot_precision_recall
from pyxvis.io.data import load_features

(X,d,Xt,dt)   = load_features('../data/F2/F2')        # load train/test data
plot_features(X,d,'F2 dataset')                       # plot of feature space

ss_cl = ['nn(3,)','nn(4,)','nn(8,)']                  # classifiers to evaluate

curve = 1                                             # 0 = ROC curve, 
                                                      # 1 = precision/recall curve
for k in range(len(ss_cl)):
    cl_name       = ss_cl[k]
    (name,params) = clf_model(cl_name)                # function name and parameters
    clf           = define_classifier([name,params])  # classifier definition
    clf           = train_classifier(clf,X,d)         # classifier training
    p             = clf.predict_proba(Xt)[:,1]        # classification probabilities
    if curve == 0: # ROC curve
        auc       = roc_auc_score(dt, p)              # area under curve
        fpr,tpr,_ = roc_curve(dt, p)                  # false and true positive rates
        plot_ROC(fpr,tpr,cl_name,auc,[k,n])           # ROC curve
    else:          # precision/recall curve
        ap        = average_precision_score(dt, p)    # area under curve
        pr,re,_   = precision_recall_curve(dt, p)     # precision and recall values
        plot_precision_recall(pr,re,cl_name,ap,[k,n]) # precision/recall curve
