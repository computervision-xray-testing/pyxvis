import numpy as np
import cv2
from scipy.ndimage.morphology import binary_dilation as imdilate
def superimpose_xray_images(Ib,If,i,j):

    B = 0.1
    C = 230 
    
    It = Ib.copy()

    i0b = max(0,i)
    i1b = min( max(0, i + If.shape[0]), Ib.shape[0])
    j0b = max(0,j)
    j1b = min( max(0, j + If.shape[1]), Ib.shape[1])
    Jb = (Ib[i0b:i1b,j0b:j1b]-B)/C

    i0f =  max(-i, 0)
    i1f = min(Ib.shape[0] - i, If.shape[0])
    j0f = max(-j, 0)
    j1f = min(Ib.shape[1] - j, If.shape[1])
    Jf = (If[i0f: i1f,j0f  : j1f]- B)/C
    
    It[i0b:i1b,j0b:j1b] = C*Jf*Jb + B

    return It


def draw_bounding_box(I,y1,x1,dy,dx,label):
    # Modified from  Project: Traffic_sign_detection_YOLO   
    # Author: AmeyaWagh   
    # File: objectDetectorYOLO.py    
    # License: MIT 
    y2 = y1+dy
    x2 = x1+dx
    cv2.rectangle(I,(x1,y1),(x2,y2),(0,255,0),6)
    font = cv2.FONT_HERSHEY_PLAIN
    labelSize=cv2.getTextSize(label,font,0.5,2)
    _x1 = x1
    _y1 = y1
    _x2 = _x1+labelSize[0][0]*12
    _y2 = y1-int(labelSize[0][1])*12
    cv2.rectangle(I,(_x1,_y1),(_x2,_y2),(0,255,0),cv2.FILLED)
    cv2.putText(I,label,(x1-5,y1-5),font,5.5,(200,200,200),3,cv2.LINE_AA)
    return I


def mask_simulation(I,h,i0,j0):
    J = I.copy()
    (mi,mj) = h.shape
    mi2  = int(np.fix(mi/2))
    mj2  = int(np.fix(mj/2))
    i1 = i0-mi2 
    i2 = i0+mi2+1
    j1 = j0-mj2 
    j2 = j0+mj2+1
    J[i1:i2,j1:j2] = I[i1:i2,j1:j2]*(1+h)
    return   J

def funcf(m,K):
    x = np.ones((3,1))
    x[0] = m[0]
    x[1] = m[1]
    y = np.matmul(K,x)
    return y


def ellipsoid_simulation(I,K,SSe,f,abc,var_mu,xmax):

    J = I.copy()
    (N,M) = I.shape
    

    R = np.zeros((N,M)) # ROI of simulated defect

    invK  = np.linalg.inv(K)

    if len(abc)==3: # elliposoid
        (a,b,c) = abc 
    else: # sphere
        a = abc 
        b = a
        c = a 

    # Computation of the 3 x 3 matrices Phi and L
    H     = np.linalg.inv(SSe)
    h0    = H[0,:]/a
    h1    = H[1,:]/b
    h2    = H[2,:]/c
    Hs    = np.zeros((3,3))
    Hs[:,0] = h0[0:3]
    Hs[:,1] = h1[0:3]
    Hs[:,2] = h2[0:3]
    hd      = np.zeros((3,1))
    hd[0] = h0[3]
    hd[1] = h1[3]
    hd[2] = h2[3]
    Phi = np.matmul(Hs,Hs.T)
    hhd = np.matmul(hd,hd.T)
    hhd1 = 1-np.matmul(hd.T,hd)
    L = np.matmul(np.matmul(Hs,hhd),Hs.T) + hhd1*Phi

    # Location of the superimposed area
    A     = L[0:2,0:2]
    mc    = np.array(-f*np.matmul(np.linalg.inv(A),L[0:2,2]))
    x     = np.linalg.eig(A)[0]
    C     = np.array([x[1],x[0]])
    la    = C
    a00   = np.linalg.det(L)/np.linalg.det(A)
    ae    = f*np.sqrt(-a00/la[0])
    be    = f*np.sqrt(-a00/la[1])
    al    = np.arctan2(C[1],C[0])+np.pi
    ra    = np.array( [ae*np.cos(al), ae*np.sin(al)] )
    rb    = np.array( [be*np.cos(al+np.pi/2), be*np.sin(al+np.pi/2)] )
    u1    = funcf(mc+ra,K)
    u2    = funcf(mc+rb,K)
    u3    = funcf(mc-ra,K)
    u4    = funcf(mc-rb,K)
    uc    = funcf(mc,K)
    e1    = u1+u2-uc
    e2    = u1+u4-uc
    e3    = u3+u2-uc
    e4    = u3+u4-uc
    E     = np.concatenate((e1,e2,e3,e4),axis=1)
    Emax  = np.max(E,axis=1)
    Emin  = np.min(E,axis=1)
    umin  = int(max(np.fix(Emin[0]), 1))
    vmin  = int(max(np.fix(Emin[1]), 1))
    umax  = int(min(np.fix(Emax[0]+1), N))
    vmax  = int(min(np.fix(Emax[1]+1), M))
    q  = 255/(1-np.exp(var_mu*xmax))
    R[umin:umax,vmin:vmax] = 1
    R = imdilate(R)
    z = np.zeros((2,1))
    for u in range(umin,umax):
        z[0] = u
        for v in range(vmin,vmax):
            z[1] = v
            m = funcf(z,invK)
            m[0:2] = m[0:2]/f
            p = np.matmul(np.matmul(m.T,L),m)
            if p>0:
                x = np.matmul(np.matmul(m.T,Phi),m)
                d = 2*np.sqrt(p)*np.linalg.norm(m)/x
                J[u,v] = np.exp(var_mu*d)*(I[u,v]-q)+q

    return J


def voxels_simulation(N,M,V,s,P):
    Q = np.zeros((N,M))
    (Nx,Ny,Nz) = V.shape
    for x in range(Nx):
        print(str(x)+'/'+str(Nx))
        X = x/s
        for y in range(Ny):
            Y = y/s
            for z in range(Nz):
                if V[x,y,z]:
                    Z = z/s
                    Mp = np.array([X,Y,Z,1])
                    w = np.matmul(P,Mp)
                    w = w/w[2]
                    i1 = int(np.fix(w[0]))
                    if i1 >= 0 and i1<N:
                        j1 = int(np.fix(w[1]))
                        if j1 >= 0 and j1<M:
                            i2 = i1+1
                            j2 = j1+1
                            a1 = w[0]-i1     
                            b1 = w[1]-j1
                            a2 = 1-a1          
                            b2 = 1-b1
                            Q[i1,j1] = Q[i1,j1] + a2*b2
                            Q[i1,j2] = Q[i1,j2] + a2*b1
                            Q[i2,j1] = Q[i2,j1] + a1*b2
                            Q[i2,j2] = Q[i2,j2] + a1*a1

    return Q
