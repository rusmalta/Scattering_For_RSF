# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:42:23 2016

@author: rusmalta63
"""

def uniq(seq, idfun=None): 
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

import os, sys
import numpy as np
sys.path.append(os.path.dirname(__file__))




def dataZ():
    
    omega = []
    alpha_0 = [] 
    
    with open(os.path.dirname(__file__)+'/dataZ/alpha_0_x.txt', 'r') as f:
        for columns in ( raw.strip().split() for raw in f ):  
            omega.append(float(columns[0]))
            alpha_0.append(float(columns[1]))
            
#    
#    omega = uniq(omega)
#    l_omega = len(omega)
#    
#    arr = np.split(alpha_0, l_omega)
#    l_root = len(arr[0])
#    
#    
#    
#    root = np.zeros((l_lambd, l_root), dtype = complex)
#    for i in range(0, len(lambd)):
#        root[i, :] = np.array(arr[i])
#    
#    lambd = np.array(lambd)
    
    return omega, alpha_0


root, lambd = dataZ()