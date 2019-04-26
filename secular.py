
import os, sys
sys.path.append(os.path.dirname(__file__))

import scipy.special as sp

##### Secular equations
## l>0 (for l=0 electric field vanishes)
 
def spherical_ddjn(l, z):
    L = float(l)
    SphBesR0 = sp.spherical_jn(l, z)
    dSphBesR1 = sp.spherical_jn(l - 1, z, True)
    dSphBesR0 = sp.spherical_jn(l, z, True)
    result = dSphBesR1 + (L + 1) / z ** 2. * SphBesR0 - (L + 1) / z * dSphBesR0
    return result
    
def spherical_ddyn(l, z):
    L = float(l)
    SphBesR0 = sp.spherical_yn(l, z)
    dSphBesR1 = sp.spherical_yn(l - 1, z, True)
    dSphBesR0 = sp.spherical_yn(l, z, True)
    result = dSphBesR1 + (L + 1) / z ** 2. * SphBesR0 - (L + 1) / z * dSphBesR0
    return result
    
def spherical_hn(l, z):
    result = sp.spherical_jn(l, z) + 1j * sp.spherical_yn(l, z)
    return result
    
def spherical_dhn(l, z):
    result = sp.spherical_jn(l, z, True) + 1j * sp.spherical_yn(l, z, True)
    return result
    
def spherical_ddhn(l, z):
    result = spherical_ddjn(l, z) + 1j * spherical_ddyn(l, z)
    return result
       
def secular_TE(z, l, nR):
    weight = z ** 2.
    dweight = 2. * z
    SphJ = sp.spherical_jn(l, nR * z)
    dSphJ = sp.spherical_jn(l, nR * z, True)
    ddSphJ = spherical_ddjn(l, nR * z)
    SphH = spherical_hn(l, z)
    dSphH = spherical_dhn(l, z)
    ddSphH = spherical_ddhn(l, z)
    f_ker =  ( nR * dSphJ * SphH - SphJ * dSphH )
    f = weight * f_ker
    df_ker = ( nR ** 2. * ddSphJ * SphH - SphJ * ddSphH )
    df = dweight * f_ker + weight * df_ker
    return (f,df)
    
def secular_TM(z, l, nR):
    weight = z
    dweight = 1.
    SphJ = sp.spherical_jn(l, nR * z)
    dSphJ = sp.spherical_jn(l, nR * z, True)
    ddSphJ = spherical_ddjn(l, nR * z)
    SphH = spherical_hn(l, z)
    dSphH = spherical_dhn(l, z)
    ddSphH = spherical_ddhn(l, z)
    f_ker = (nR * dSphJ * SphH * z - nR ** 2. * dSphH * SphJ * z - 
             (nR ** 2. - 1.) * SphJ * SphH)
    f = weight * f_ker
    df_ker = (nR ** 2. * ddSphJ * SphH * z - nR ** 2. * ddSphH * SphJ * z  + 
              nR * (1. - nR ** 2.) * dSphJ * dSphH * z - 
              (2. * nR ** 2. - 1.) * dSphH * SphJ - 
              (nR ** 2. - 2.) * nR * dSphJ * SphH)
    df = dweight * f_ker + weight * df_ker
    return (f,df)
    