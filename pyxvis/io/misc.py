#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#import cv2
#from PIL import Image
#import numpy as np
#from sklearn.metrics import confusion_matrix, accuracy_score
#from seaborn import heatmap
#from mpl_toolkits.mplot3d import Axes3D
import os, fnmatch
#import matplotlib.pyplot as plt
#from matplotlib.patches import Ellipse

def dirfiles(img_path,img_ext):
    img_names = fnmatch.filter(sorted(os.listdir(img_path)),img_ext)
    if '.DS_Store' in img_names:
        img_names.remove('.DS_Store')
    return img_names

def num2fixstr(x,d):
    st = '%0*d' % (d,x)
    return st

