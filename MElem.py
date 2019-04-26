import os, sys
sys.path.append(os.path.dirname(__file__))
import scipy as sc
import scipy.special as sp
import numpy as np
from scipy.special import spherical_jn as jn


###Supplemental functions   
# Normalized Legendre polynomials P_l^m(x)
def leg(th, l, m): 
    L = np.float_(l)
    M = abs(m)
    fact_minus = np.float_(sp.factorial(L - M))
    fact_plus = np.float_(sp.factorial(L + M))
    norm = np.sqrt((2. * L + 1.) / 2. * fact_minus / fact_plus)
    result = norm * sp.lpmv(M, l, th)
    return result
# Derivative of normalized Legendre polynomials P_l^m(x) with respect to x    
def dleg(th, l, m):
    L = np.float_(l)
    M = abs(m)
    fact_minus = np.float_(sp.factorial(L - M))
    fact_plus = np.float_(sp.factorial(L + M))
    norm = np.sqrt((2. * L + 1.) / 2. * fact_minus / fact_plus)
    weight = 1. / (1. - th * th)
    Legendre_l1 = sp.lpmv(M, l-1, th)
    Legendre_l0 = sp.lpmv(M, l, th)
    result = norm * (Legendre_l1 * (L + M) - th * L * Legendre_l0) * weight
    return result
def Dleg(th, l, m, Legendre_l0):
    L = np.float_(l)
    M = abs(m)
    fact_minus = np.float_(sp.factorial(L - M))
    fact_plus = np.float_(sp.factorial(L + M))
    norm = np.sqrt((2. * L + 1.) / 2. * fact_minus / fact_plus)
    weight = 1. / (1. - th * th)
    Legendre_l1 = sp.lpmv(M, l-1, th)
    result = (norm * Legendre_l1 * (L + M) - th * L * Legendre_l0) * weight
    return result

# Spherical Bessel function of 1-st kind j_l(x)
def bess(r, kn, l): 
    knr = kn * r
    SphBesR = jn(l, knr)
    SphBes = jn(l, kn)
    result = SphBesR / SphBes
    return result  
# Derivative of (j_l(x) * x)
def dbess(r, kn, l):
    knr = kn * r
    SphBesR = jn(l, knr)
    dSphBesR = jn(l, knr, True)
    SphBes = jn(l, kn)
    result = (SphBesR + knr * dSphBesR) / SphBes 
    return result


### normalization constants
# TE-modes
def A_TE(l, nR):
    L = np.float_(l)
    l_factor = 2. / (L * (L + 1.) )
    result = np.sqrt(l_factor / (nR * nR - 1.) )
    return result 
# TM-modes
def A_TM(l, nR, kn):
    L = np.float_(l)
    TE = A_TE(l, nR)
    SphBes_l1 = sp.spherical_jn(l - 1, nR * kn)
    SphBes_l0 = sp.spherical_jn(l, nR * kn)
    term_1 = SphBes_l1 / SphBes_l0 - L / (nR * kn)
    term_2 = L * (L + 1.) / (kn * kn)
    result =  TE * nR / np.sqrt(term_1 * term_1 + term_2)
    return result
# LE-modes
def A_LE(l, nR, kn):
    L = np.float_(l)
    result = A_TM(l, nR, kn) * np.sqrt(L * (nR * nR - 1.))
    return result

###Radial integrals, depend on spherical bessel functions
   
def T1_ker(r, nR, kn1, kn2, l1, l2): 
    SphBes_1 = bess(r, nR * kn1, l1)
    SphBes_2 = bess(r, nR * kn2, l2)
    weight = r * r
    result = SphBes_1 * SphBes_2 * weight
    return result  
def T2_ker(r, nR, kn1, kn2, l1, l2): 
    SphBes_1 = bess(r, nR * kn1, l1)
    SphBes_2 = bess(r, nR * kn2, l2)
    result = SphBes_1 * SphBes_2
    return result  
def T3_ker(r, nR, kn1, kn2, l1, l2): 
    SphBes_1 = dbess(r, nR * kn1, l1)
    SphBes_2 = dbess(r, nR * kn2, l2)
    result = SphBes_1 * SphBes_2
    return result
def T4_ker(r, nR, kn1, kn2, l1, l2): 
    SphBes_1 = bess(r, nR * kn1, l1)
    SphBes_2 = dbess(r, nR * kn2, l2)
    weight = r
    result = SphBes_1 * SphBes_2 * weight
    return result  


###Azimutal integrals 
#Notations: [th1=cos(theta2), th2=cos(theta1)]
                                       
def Q1_ker(th, l1, l2, m1, m2): 
    Legendre_1 = leg(th, l1, m1)
    Legendre_2 = leg(th, l2, m2)
    weight = 1. / (1. - th * th)
    result = Legendre_1 * Legendre_2 * weight
    return result
def Q2_ker(th, l1, l2, m1, m2): 
    Legendre_1 = dleg(th, l1, m1)
    Legendre_2 = dleg(th, l2, m2)
    weight = (1. - th * th)
    result = Legendre_1 * Legendre_2 * weight
    return result
def Q3_ker(th, l1, l2, m1, m2): 
    Legendre_1 = leg(th, l1, m1)
    Legendre_2 = leg(th, l2, m2)
    result = Legendre_1 * Legendre_2
    return result
def Q4_ker(th, l1, l2, m1, m2): 
    Legendre_1 = leg(th, l1, m1)
    Legendre_2 = leg(th, l2, m2)
    Legendre_3 = dleg(th, l1, m1)
    Legendre_4 = dleg(th, l2, m2)
    weight = -1.    
    result = (Legendre_1 * Legendre_4 + Legendre_2 * Legendre_3) * weight
    return result

def TQ(r, th, sign, t, q):
    temp = sign * sc.integrate.cumtrapz(q, th, axis = 1)
    temp = np.insert(temp, 0, 0, axis = 1)
    temp = t * temp
    result = sc.integrate.cumtrapz(temp, r, axis = 1)
    return result[:, -1]

def TQ_old(r, th, sign, t, q):
    temp = sign * sc.integrate.cumtrapz(q, th, axis = 2)
    temp = np.insert(temp, 0, 0, axis = 2)
    temp = t * temp
    result = sc.integrate.cumtrapz(temp, r, axis = 2)
    return result[:, :, -1]
    
#Matrix elements 
                        
def V_TE_TE(delta_eps, nR, m, TQ11, TQ12, A_TE1, A_TE2):
    M = float(m)
    norm = A_TE1 * A_TE2
    t_Q = (M * M) * TQ11 + TQ12
    return delta_eps * norm * t_Q

def V_TM_TM(delta_eps, nR, kn1, kn2, l1, l2, m, TQ31, TQ32, TQ23, A_TM1, A_TM2):
    L1 = np.float_(l1);     L2 = np.float_(l2);     M = float(m)
    norm = A_TM1 * A_TM2 / (nR * nR * nR * nR) / (kn1 * kn2)
    t_Q1 = L1 * (L1 + 1.) * L2 * (L2 + 1.) * TQ23
    t_Q2 = (M * M) * TQ31 + TQ32
    return delta_eps * norm * (t_Q1 + t_Q2)

def V_TE_TM(delta_eps, nR, kn2, m, TQ44, A_TE, A_TM):
    M = float(m)
    norm = A_TE * A_TM / (nR * nR) / (kn2)
    t_Q = M * TQ44
    return delta_eps * norm * t_Q
