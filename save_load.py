###to allow the import of my own packets from current directory
import os, sys
sys.path.append(os.path.dirname(__file__))
 
###import built-in science packets
import scipy as sc
import numpy as np

import APM
import secular as sec

def save_TE(x, y, w, h, nR, eps, l):  
    
    TE = np.array([], dtype=complex)   #create empty array
    TEp = APM.findz(x, y, w, h, sec.secular_TE, l, nR, TE)  #find pozitive TE roots
    TEp = np.array(sorted(TEp))
    TEm = -np.conj(TEp)         #define negative TE roots by hands
    if (len(TEp) > 1):
        if (np.abs(np.real(TEp[0])) < 1e-10):
            TEm = np.delete(TEm, 0)       #delete root with Re(k)=0 (for TE it exist only for odd l)
    TE = np.array(sorted(sc.hstack((TEm, TEp))))   #stack negative and pozitive roots

    #save to .../rootdata/eps__/TE__.npy
    EPS = str(np.round(np.real(eps), 5))
    L = str(l)
    np.save(os.path.dirname(__file__)+'/rootdata/eps'+EPS+'/TE'+L+'.npy', TE)
    print ('TE'+L+' done!')
    return 
    
def save_TM(x, y, w, h, nR, eps, l):  
       
    TM = np.array([], dtype=complex)   #create empty array
    TMp = APM.findz(x, y, w, h, sec.secular_TM, l, nR, TM)  #find pozitive TE roots
    TMp = np.array(sorted(TMp))
    TMm = -np.conj(TMp)         #define negative TE roots by hands
    if (len(TMp) > 1):
        if (np.abs(np.real(TMp[0])) < 1e-10):
            TMm = np.delete(TMm, 0)       #delete root with Re(k)=0 (for TE it exist only for even l)
    TM = np.array(sorted(sc.hstack((TMm, TMp))))   #stack negative and pozitive roots
    
    #save to .../rootdata/eps__/TM__.npy
    EPS = str(np.round(np.real(eps),5))
    L = str(l)
    np.save(os.path.dirname(__file__)+'/rootdata/eps'+EPS+'/TM'+L+'.npy', TM)
    print ('TM'+L+' done!')
    return 
   
def load_TE(k_max, eps, l_min, l_max):
    EPS = str(np.round(np.real(eps),5))
    TE = np.array([], dtype = complex)
    l_TE = np.array([], dtype = int)
    #load from .../rootdata/eps__/TE__.npy
    for l in range(l_min, l_max + 1, 2):
        L = str(l)
        TE_temp = np.load(os.path.dirname(__file__)+'/rootdata/eps'+EPS+'/TE'+L+'.npy')
        TE_temp = np.array(sorted(k for k in TE_temp if (abs(k) < k_max)))
        l_temp = l * np.ones(len(TE_temp), dtype=int)
        TE = np.append(TE, TE_temp)
        l_TE = np.append(l_TE, l_temp)        
    return (TE, l_TE)
    
    

#(abs(np.real(k)) <= k_max) and (-np.imag(k) <= 100.)

def load_TM(k_max, eps, l_min, l_max):
    EPS = str(np.round(np.real(eps),5))
    TM = np.array([], dtype = complex)
    l_TM = np.array([], dtype = int)
    #load from .../rootdata/eps__/TM__.npy
    for l in range(l_min, l_max + 1, 2):
        L = str(l)
        TM_temp = np.load(os.path.dirname(__file__)+'/rootdata/eps'+EPS+'/TM'+L+'.npy')
        TM_temp = np.array(sorted(k for k in TM_temp if abs(k) < k_max))
        l_temp = l * np.ones(len(TM_temp), dtype=int)
        TM = np.append(TM, TM_temp)
        l_TM = np.append(l_TM, l_temp)     
    return (TM, l_TM)
    
def load_LE(l_min, l_max):
    #create LE array by hands using LE_0
    LE = np.array([], dtype = complex)
    l_LE = np.array([], dtype = int)
    LE_0 = -1e-6j
    for l in range(l_min, l_max + 1, 2):
        LE = np.append(LE, LE_0)
        l_LE = np.append(l_LE, l)
    return (LE, l_LE)  