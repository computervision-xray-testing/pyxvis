import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation as imdilate
from scipy.fftpack import dct
from skimage.segmentation import find_boundaries
from pybalu.feature_extraction import lbp_features # > pip3 install scipy==1.2
from pybalu.feature_extraction import gabor_features # > pip3 install scipy==1.2
from pybalu.feature_extraction import haralick_features # > pip3 install scipy==1.2
from pybalu.feature_extraction import fourier_features # > pip3 install scipy==1.2
from pybalu.feature_extraction import hog_features # > pip3 install scipy==1.2
from pybalu.feature_extraction import basic_int_features # > pip3 install scipy==1.2
from pybalu.feature_extraction import basic_geo_features # > pip3 install scipy==1.2
from pybalu.feature_extraction import fourier_des_features # > pip3 install scipy==1.2
from pybalu.feature_extraction import hugeo_features # > pip3 install scipy==1.2
from pybalu.feature_extraction import flusser_features # > pip3 install scipy==1.2
from pybalu.feature_extraction import gupta_features # > pip3 install scipy==1.2
from pybalu.img_processing import segbalu # > pip3 install scipy==1.2
from pyxvis.processing.images import Edge, im_grad
from pyxvis.processing.helpers.kfunctions  import gaussian_kernel
from pyxvis.processing.segmentation import seg_bimodal
from pyxvis.io.misc import dirfiles
from pyxvis.io.plots import get_image


def extract_features(fx, img=None, bw=None):
    # img grayscale image, bw binary image
    # check if I2 is a binary image (segmented region)
    if type(fx)==str:
        fx = [fx]
    n = len(fx)
    for i in range(n):
        X = extract_features_image(fx[i], img=img, bw=bw)
        if i == 0:
            Xf = X
        else:
            Xf = np.concatenate((Xf,X))
    features = np.asarray(Xf)
    return features


def extract_features_image(fxi, img=None, bw=None):
    # img grayscale image, bw binary image
    # check if I2 is a binary image (segmented region)
    if fxi == 'lbp':
        X = lbp_features(img, region=bw, hdiv=1, vdiv=1, mapping='nri_uniform')
    elif fxi == 'lbp-ri':
        X = lbp_features(img, region=bw,hdiv=1, vdiv=1, mapping='uniform')
    elif fxi == 'gabor':
        nr = 8
        nd = 8
        X = gabor_features(img, region=bw,rotations=nr, dilations=nd)
    elif fxi == 'gabor-ri':
        nr = 8
        nd = 8
        Y = gabor_features(img, region=bw,rotations=nr, dilations=nd)
        X = np.zeros((nd+3,))

        for j in range(nd):
            X[j] = np.sum(X[j*nr:(j+1)*nr-1])
        X[nr] = Y[nr*nd]
        X[nr+1] = Y[nr*nd+1]
        X[nr+2] = Y[nr*nd+2]
    elif fxi == 'hog':
        X = hog_features(img, region=bw, v_windows=1, h_windows=1, n_bins=9,
                         normalize=False, labels=False, show=False)
    elif fxi == 'haralick-1':
        X     = haralick_features(img, region=bw,distance=1)
    elif fxi == 'haralick-2':
        X     = haralick_features(img, region=bw,distance=2)
    elif fxi == 'haralick-3':
        X     = haralick_features(img, region=bw,distance=3)
    elif fxi == 'haralick-5':
        X     = haralick_features(img, region=bw,distance=5)
    elif fxi == 'haralick-7':
        X     = haralick_features(img, region=bw,distance=7)
    elif fxi == 'fourier':
        X     = fourier_features(img, region=bw)
    elif fxi == 'basicint':
        X     = basic_int_features(img, region=bw)
    elif fxi == 'clp':
        X     = clp_features(img)
    elif fxi == 'contrast':
        X     = contrast_features(img, region=bw)
    elif fxi == 'dct':
        X     = dct_features(img, region=bw)
    elif fxi == 'basicgeo':
        X = basic_geo_features(bw)
    elif fxi == 'centroid':
        f = basic_geo_features(bw)
        X = f[0:2]
    elif fxi == 'fourierdes':
        X = fourier_des_features(bw)
    elif fxi == 'flusser':
        X = flusser_features(bw)
    elif fxi == 'gupta':
        X = gupta_features(bw)
    elif fxi == 'hugeo':
        X = hugeo_features(bw)
    elif fxi == 'ellipse':
        X = ellipse_features(bw)
    else:
        print('ERROR: ' + fxi + ' does not exist as geometric feature extraction method.')
    features = np.asarray(X)
    return features


def extract_features_dir(fx, dirpath, fmt, segmentation=False):
    """

    Args:
        fx:
        dirpath:
        fmt:
        segmentation:

    Returns:

    """
    st = '*.'+fmt
    img_names = dirfiles(dirpath+'/',st)
    n = len(img_names)
    print('Extracting features in ' + str(n) + ' images...')
    for i in range(n):
        img_path = img_names[i]
        print('... reading '+dirpath+'/'+img_path)
        img = get_image(dirpath + '/' + img_path)
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if segmentation=='balu':
            R, _, _ = segbalu(gray)
        elif segmentation=='bimodal':
            R, _ = seg_bimodal(gray)
            R = R*1
        else:
            R = None
        features = extract_features(fx,img=gray,bw=R)
        if i==0:
            m = features.shape[0]
            data = np.zeros((n,m))
            print('Extracting ' + str(features.shape[0]) + ' features...')
        data[i]    = features
    return data


def extract_features_labels(fx,dirpath,fmt,segmentation=False):
    dir_names = dirfiles(dirpath+'/','*')
    c = len(dir_names) # number of classes
    for k in range(c):
        Xk = extract_features_dir(fx,dirpath+'/'+dir_names[k],fmt,segmentation)
        dk = k*np.ones([Xk.shape[0],],dtype=int)
        if k==0:
            X = Xk
            d = dk
        else:
            X = np.concatenate((X,Xk),axis=0)
            d = np.concatenate((d,dk),axis=0)
    return X,d


def fourier_features(I,region=None,Nfourier=64,Mfourier=64,nfourier=4,mfourier=4):
    if region is None:
        region = np.ones_like(I)
    I[region == 0] = 0
    Im       = cv2.resize(I,(Nfourier,Mfourier))
    FIm      = np.fft.fft2(Im)
    Y        = FIm[0:int(Nfourier/2),0:int(Mfourier/2)]
    x        = np.abs(Y)
    F        = cv2.resize(x,(nfourier,mfourier))
    f        = np.reshape(F,(nfourier*mfourier,))
    x        = np.angle(Y)
    A        = cv2.resize(x,(nfourier,mfourier))
    a        = np.reshape(A,(nfourier*mfourier,))
    features = np.concatenate((f,a))
    return features

def dct_features(I,region=None,Ndct=64,Mdct=64,ndct=4,mdct=4):
    if region is None:
        region = np.ones_like(I)
    I[region == 0] = 0
    Im       = cv2.resize(I,(Ndct,Mdct))
    FIm      = dct(Im)
    Y        = FIm[0:int(Ndct/2),0:int(Mdct/2)]
    x        = np.abs(Y)
    F        = cv2.resize(x,(ndct,mdct))
    features = np.reshape(F,(ndct*mdct,))
    return features

def fit_ellipse(x,y):
    # Fitzgibbon, A.W., Pilu, M., and Fischer R.B., 
    # Direct least squares fitting of ellipses, 1996
    x        = x[:,None]
    y        = y[:,None]
    D        = np.hstack([x*x,x*y,y*y,x,y,np.ones(x.shape)])
    S        = np.dot(D.T,D)
    C        = np.zeros([6,6])
    C[0,2]   = C[2,0] = 2
    C[1,1]   = -1
    E,V      = np.linalg.eig(np.dot(np.linalg.inv(S),C))
    n        = np.argmax(E)
    s        = V[:,n]
    a        = s[0]
    b        = s[1]/2.
    c        = s[2]
    d        = s[3]/2.
    f        = s[4]/2.
    g        = s[5]
    dd       = b*b-a*c
    cx       = (c*d-b*f)/dd
    cy       = (a*f-b*d)/dd
    alpha    = 0.5*np.arctan(2*b/(a-c))*180/np.pi
    up       = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1    = (b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2    = (b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    a        = np.sqrt(abs(up/down1))
    b        = np.sqrt(abs(up/down2))
    area     = np.pi*a*b

    if b>a:
        ecc  = a/b
    else:
        ecc  = b/a

    features = [cx,cy,a,b,alpha,ecc,area]

    return features

def ellipse_features(R):
    E        = find_boundaries(R, mode='outer').astype(np.uint8)
    # E        = bwperim(R)
    data     = np.argwhere(E==True)
    y        = data[:,0]
    x        = data[:,1]
    features = fit_ellipse(x,y)
    return features

def clp_features(img):

    # indices of the pixels of the 8 profiles of CLP (each profile is a line of 32 pixels)
    # C[2*k,:]  : coordinate i of profile k, for k=0...7
    # C[2*k+1,:]: coordinate j of profile k, for k=0...7
    C = np.array([
        [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [8,8,9,9,10,10,11,11,12,12,13,14,14,15,15,16,16,17,17,18,18,19,20,20,21,21,22,22,23,23,24,25],
        [8,8,9,9,10,10,11,11,12,12,13,14,14,15,15,16,16,17,17,18,18,19,20,20,21,21,22,22,23,23,24,25],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [25,24,23,23,22,22,21,21,20,20,19,18,18,17,17,16,16,15,15,14,14,13,12,12,11,11,10,10,9,9,8,8],
        [25,24,23,23,22,22,21,21,20,20,19,18,18,17,17,16,16,15,15,14,14,13,12,12,11,11,10,10,9,9,8,8],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]])


    ng = 32
    X = np.array(np.zeros((8,ng)))
    I = cv2.resize(img, (ng, ng))
    
    for i in range(8):
        k = i*2
        for j in range(ng):
            X[i,j] = I[C[k,j],C[k+1,j]]*255
        #plt.plot(range(32),X[i,:])
    #plt.show()
    d = np.abs(X[:,0]-X[0:,31])
    k = np.argmin(d)
    y = X[k,:]
    #plt.plot(range(32),y)
    #plt.show()
    Po  = y/y[0]
    Q   = rampefree(Po)
    Qm  = np.average(Q)
    Qd  = np.max(Q)-np.min(Q)
    Qd1 = np.log(Qd+1)
    Qd2 = 2*Qd/(Po[0]+Po[ng-1])
    Qs  = np.std(Q)
    Qf  = np.abs(np.fft.fft(Q))
    Qf  = Qf[range(1,8)]
    features = [Qm, Qs, Qd, Qd1, Qd2, Qf[0], Qf[1], Qf[2], Qf[3], Qf[4], Qf[5], Qf[6]]
    return features


def rampefree(x):
    k = len(x)-1
    m = (x[k]-x[0])/k
    b = x[0]
    y = x - range(k+1)*m - b
    return y

def contrast_features(img,region=None,neihbors=2):
    img = img*255
    if region is None:
        region = np.ones_like(img)
    
    R = region==1
    Rn = R
    for i in range(neihbors):
        Rn = imdilate(Rn)

    Rn = np.bitwise_and(Rn,~R)

    if np.max(Rn)==1:
        Ir = img*region
        In = img*Rn
        MeanGr = np.average(Ir)
        MeanGn = np.average(In)
        K1 = (MeanGr-MeanGn)/MeanGn            # contrast after Kamm, 1999
        K2 = (MeanGr-MeanGn)/(MeanGr+MeanGn)   # modulation after Kamm, 1999
        K3 = np.log(MeanGr/MeanGn)                # film-contrast after Kamm, 1999
    else:
        K1 = -1        
        K2 = -1        
        K3 = -1

    (nI,mI) = img.shape

    n1 = int(round(nI/2)+1)
    m1 = int(round(mI/2)+1)

    P1 = img[n1,:]    # Profile in i-Direction
    P2 = img[:,m1]    # Profile in j-Direction
    Q1 = rampefree(P1)
    Q2 = rampefree(P2)
    Q = np.concatenate((Q1,Q2))

    Ks = np.std(Q)
    K  = np.log(np.max(Q)-np.min(Q)+1)
        
    features = [K1,K2,K3,Ks,K]
    return features





