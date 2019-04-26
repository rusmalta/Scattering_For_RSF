###to allow the import of my own packets from current directory
import os, sys
sys.path.append(os.path.dirname(__file__))
 
###import built-in science packets
import scipy as sc
import numpy as np
import scipy.special as sp
import time
from scipy.special import spherical_jn as jn

###import my modules
import MElem as ME
 

def V(r, th, sign, delta_eps, TE, LE, TM, nR, l_TE, l_LE, l_TM, m):
    
    #basis vectors
    TMM = sc.hstack((LE, TM))
    
    #normalization constants
    A_TE = ME.A_TE(l_TE, nR)
    A_LE = ME.A_LE(l_LE, nR, LE)
    A_TM = ME.A_TM(l_TM, nR, TM)
    A_TMM = sc.hstack((A_LE, A_TM))
      
    ##############HAMILTIONIAN MATRIX ELEMENTS
    
    ### TE BLOCK
    #grids of basis vectors
    root_1, root_2 = np.meshgrid(TE, TE, indexing='ij')

    SphBes_0_TE = jn(l_TE, nR * TE )
    t_01, t_02 = np.meshgrid(SphBes_0_TE, SphBes_0_TE, indexing='ij')
    
    TE_2d, r_2d = np.meshgrid(TE, r, indexing='ij')
    l_TE_2d, th_2d = np.meshgrid(l_TE, th, indexing='ij')
    TQ11_TE = np.ones((len(TE),len(TE)), dtype = complex)
    TQ12_TE = np.ones((len(TE),len(TE)), dtype = complex)
    
    SphBes_TE = jn(l_TE_2d, nR * TE_2d * r_2d)
    Legendre_TE = ME.leg(th_2d, l_TE_2d, m)
    dLegendre_TE = ME.Dleg(th_2d, l_TE_2d, m, Legendre_TE)
    
    for i in range(0, len(TE)):
        t1 = SphBes_TE[i, :] * SphBes_TE[i:, :] * r_2d[i, :] * r_2d[i, :] 
        q1 = Legendre_TE[i, :] * Legendre_TE[i:, :] / (1. - th_2d[i, :] * th_2d[i, :])
        q2 = dLegendre_TE[i, :] * dLegendre_TE[i:, :] * (1. - th_2d[i, :] * th_2d[i, :])
        TQ11_TE[i:, i] = ME.TQ(r_2d[i:, :], th_2d[i:, :], sign, t1, q1)
        TQ12_TE[i:, i] = ME.TQ(r_2d[i:, :], th_2d[i:, :], sign, t1, q2)
        
    TQ11_TE = TQ11_TE / (t_01 * t_02)    
    TQ12_TE = TQ12_TE / (t_01 * t_02)

    #normalization grid
    A_1, A_2 = np.meshgrid(A_TE, A_TE, indexing='ij')  

    #calculation of matrix elements between modes via ME module
    h_TE_TE = ME.V_TE_TE(delta_eps, nR, m, TQ11_TE, TQ12_TE, A_1, A_2)
    
    h_TE_TE = h_TE_TE / 2. / (np.sqrt(root_1) * np.sqrt(root_2))
    h_TE_TE = np.tril(h_TE_TE, -1).T + np.tril(h_TE_TE, 0)
    
    ### LE&TM BLOCK
    #grids of basis vectors
    root_1, root_2 = np.meshgrid(TMM, TMM, indexing='ij')  

    #grid of orbital numbers
    l_TMM = sc.hstack((l_LE, l_TM))      
    l_1, l_2 = np.meshgrid(l_TMM, l_TMM, indexing='ij')
    
    SphBes_0_TM = jn(l_TMM, nR * TMM )
    t_01, t_02 = np.meshgrid(SphBes_0_TM, SphBes_0_TM, indexing='ij')

    TM_2d, r_2d = np.meshgrid(TMM, r, indexing='ij')
    l_TM_2d, th_2d = np.meshgrid(l_TMM, th, indexing='ij')
    TQ31_TM = np.ones((len(TMM),len(TMM)), dtype = complex)
    TQ32_TM = np.ones((len(TMM),len(TMM)), dtype = complex)
    TQ23_TM = np.ones((len(TMM),len(TMM)), dtype = complex)
    
    SphBes_TM = jn(l_TM_2d, nR * TM_2d * r_2d)
    dSphBes_TM = jn(l_TM_2d, nR * TM_2d * r_2d, True)
    DSphBes_TM = SphBes_TM + nR * TM_2d * r_2d * dSphBes_TM
    Legendre_TM = ME.leg(th_2d, l_TM_2d, m)
    dLegendre_TM = ME.Dleg(th_2d, l_TM_2d, m, Legendre_TM)

    #integration
    for i in range(0, len(TMM)):
        t2 = SphBes_TM[i, :] * SphBes_TM[i:, :]
        t3 = DSphBes_TM[i, :] * DSphBes_TM[i:, :]
        q3 = Legendre_TM[i, :] * Legendre_TM[i:, :]
        q1 = q3 / (1. - th_2d[i, :] * th_2d[i, :])
        q2 = dLegendre_TM[i, :] * dLegendre_TM[i:, :] * (1. - th_2d[i, :] * th_2d[i, :])
        TQ31_TM[i:, i] = ME.TQ(r_2d[i:, :], th_2d[i:, :], sign, t3, q1)
        TQ32_TM[i:, i] = ME.TQ(r_2d[i:, :], th_2d[i:, :], sign, t3, q2)
        TQ23_TM[i:, i] = ME.TQ(r_2d[i:, :], th_2d[i:, :], sign, t2, q3)
      
    TQ31_TM = TQ31_TM / (t_01 * t_02)
    TQ32_TM = TQ32_TM / (t_01 * t_02) 
    TQ23_TM = TQ23_TM / (t_01 * t_02)

    #normalization grid
    A_1, A_2 = np.meshgrid(A_TMM, A_TMM, indexing='ij')

    #calculation of matrix elements between modes via ME module
    h_TM_TM = ME.V_TM_TM(delta_eps, nR, root_1, root_2, l_1, l_2, m, TQ31_TM, TQ32_TM, TQ23_TM, A_1, A_2) 
    
    h_TM_TM = h_TM_TM  / 2. / (np.sqrt(root_1) * np.sqrt(root_2))
    h_TM_TM = np.tril(h_TM_TM, -1).T + np.tril(h_TM_TM, 0)
    
    ### TE-TM BLOCK
    #grids of basis vectors
    root_1, root_2 = np.meshgrid(TE, TMM, indexing='ij')
    t_01, t_02 = np.meshgrid(SphBes_0_TE, SphBes_0_TM, indexing='ij')
    TQ44_TE_TM = np.zeros((len(TE),len(TMM)), dtype = complex)

    #integration
    for i in range(0, len(TE)):
        t4 = SphBes_TE[i, :] * DSphBes_TM * r
        q4 = (-1.) * (Legendre_TE[i, :] * dLegendre_TM + dLegendre_TE[i, :] * Legendre_TM)
        TQ44_TE_TM[i, :] = ME.TQ(r_2d, th_2d, sign, t4, q4)
     
    TQ44_TE_TM = TQ44_TE_TM / (t_01 * t_02)

    #normalization grid
    A_1, A_2 = np.meshgrid(A_TE, A_TMM, indexing='ij')

    #calculation of matrix elements between modes via ME module
    h_TE_TM = ME.V_TE_TM(delta_eps, nR, root_2, m, TQ44_TE_TM, A_1, A_2)
    
    h_TE_TM = h_TE_TM / 2. / (np.sqrt(root_1) * np.sqrt(root_2))
    h_TM_TE = np.transpose(h_TE_TM)
  
    #############HAMILTIONIAN FORMATION
    h_up = sc.hstack((h_TE_TE, h_TE_TM))
    h_low = sc.hstack((h_TM_TE, h_TM_TM))
    h = sc.vstack((h_up, h_low))   

    return h
    
  
def Hsolve(N, h, delta_eps, TE, LE, TM, nR, l_TE, l_LE, l_TM, m): 

    l_TMM = sc.hstack((l_LE, l_TM))
    fact_TE = pow(-1, l_TE)
    fact_TMM = -pow(-1, l_TMM)
    fact_0 = sc.hstack((fact_TE, fact_TMM))
    fact = np.outer(fact_0, fact_0)
    
    rho = np.sqrt(1. - h ** 2.) 
    
    delta = 1e-8     #to overcome integral discontinuities at 0 and 1
  
    ###CYLINDER: top
    R1 = h + delta
    R2 = 1. - delta

    r = np.linspace(R1, R2, N + 1)
    th = h / r
    sign = -1.
    
    V1 = V(r, th, sign, delta_eps, TE, LE, TM, nR, l_TE, l_LE, l_TM, m)    
    
    ###CYLINDER: bottom
    V2 = fact * V1    

    ###CYLINDER: topside
    R1 = rho + delta
    R2 = 1. - delta
    r = np.linspace(R1, R2, N + 1)
    th = np.sqrt(1. - rho ** 2. / r ** 2.)
    sign = 1.
    
    V3 = V(r, th, sign, delta_eps, TE, LE, TM, nR, l_TE, l_LE, l_TM, m)  
    
    ###CYLINDER: bottomside
    V4 = fact * V3
    
    ##############FULL-HAMILTIONIAN FORMATION
    root = sc.hstack((TE, LE, TM))
    H = np.diag(1. / root) + V1 + V2 + V3 + V4
    
    #searching eigenvalues of H
    result = sc.linalg.eigvals(H)       
        
    return result
 
def V_TE(r, th, sign, delta_eps, TE, nR, l_TE, m):
    
    #normalization constants
    A_TE = ME.A_TE(l_TE, nR)
    
    ##############HAMILTIONIAN MATRIX ELEMENTS
    ### TE BLOCK
    #grids of basis vectors
    root_1, root_2 = np.meshgrid(TE, TE, indexing='ij')
    
    SphBes_0 = jn(l_TE, nR * TE )
    t_01, t_02 = np.meshgrid(SphBes_0, SphBes_0, indexing='ij')
    
    TE_2d, r_2d = np.meshgrid(TE, r, indexing='ij')
    l_TE_2d, th_2d = np.meshgrid(l_TE, th, indexing='ij')
    TQ12_TE = np.zeros((len(TE),len(TE)), dtype = complex)
    
    SphBes = jn(l_TE_2d, nR * TE_2d * r_2d)
    Legendre = ME.dleg(th_2d, l_TE_2d, m)
    
    for i in range(0, len(TE)):
        t = SphBes[i, :] * SphBes[i:, :] * r_2d[i, :] * r_2d[i, :]
        q = Legendre[i, :] * Legendre[i:, :] * (1. - th_2d[i, :] * th_2d[i, :])
        TQ12_TE[i:, i] = ME.TQ(r_2d[i:, :], th_2d[i:, :], sign, t, q)
        
    TQ12_TE = TQ12_TE.T + TQ12_TE - np.diag(np.diag(TQ12_TE)) 
    TQ12_TE = TQ12_TE / (t_01 * t_02)

    #normalization grid
    A_1, A_2 = np.meshgrid(A_TE, A_TE, indexing='ij')  

    #calculation of matrix elements between modes via ME module
    h_TE_TE = ME.V_TE_TE(delta_eps, nR, m, 0., TQ12_TE, A_1, A_2)
    
    h_TE_TE = h_TE_TE / 2. / (np.sqrt(root_1) * np.sqrt(root_2))
#    h_TE_TE = h_TE_TE / 2.

    return h_TE_TE   

def Hsolve_TE(N, h, delta_eps, TE, nR, l_TE, m): 

    fact_TE = pow(-1, l_TE)
    fact = np.outer(fact_TE, fact_TE)
    
    rho = np.sqrt(1. - h ** 2.)
    
    delta = 1e-8     #to overcome integral discontinuities at 0 and 1
  
    ###CYLINDER: top
    R1 = h + delta
    R2 = 1. - delta

    r = np.linspace(R1, R2, N + 1)
    th = h / r
    sign = -1.
    
    V1 = V_TE(r, th, sign, delta_eps, TE, nR, l_TE, m)    
    
    ###CYLINDER: bottom
    V2 = fact * V1    

    ###CYLINDER: topside
    R1 = rho + delta
    R2 = 1. - delta
    r = np.linspace(R1, R2, N + 1)
    th = np.sqrt(1. - rho ** 2. / r ** 2.)
    sign = 1.
    
    V3 = V_TE(r, th, sign, delta_eps, TE, nR, l_TE, m)  
    
    ###CYLINDER: bottomside
    V4 = fact * V3
    
    ##############FULL-HAMILTIONIAN FORMATION
    root = TE
    V_cyl =  V1 + V2 + V3 + V4
    H = np.diag(1. / root) + V_cyl 
#    H = np.eye(root.shape[0]) + V_cyl
    
    result = np.linalg.eig(H)    
#    result = sc.linalg.eig(H, np.diag(root))  
        
    return result


def V_TM(r, th, sign, delta_eps, LE, TM, nR, l_LE, l_TM, m):
    
    #basis vectors
    TMM = sc.hstack((LE, TM))
    
    #normalization constants
    A_LE = ME.A_LE(l_LE, nR, LE)
    A_TM = ME.A_TM(l_TM, nR, TM)
    A_TMM = sc.hstack((A_LE, A_TM))
      
    ##############HAMILTIONIAN MATRIX ELEMENTS
    
    ### LE&TM BLOCK
    #grids of basis vectors
    root_1, root_2 = np.meshgrid(TMM, TMM, indexing='ij')  

    #grid of orbital numbers
    l_TMM = sc.hstack((l_LE, l_TM))      
    l_1, l_2 = np.meshgrid(l_TMM, l_TMM, indexing='ij')
    
    SphBes_0_TM = jn(l_TMM, nR * TMM )
    t_01, t_02 = np.meshgrid(SphBes_0_TM, SphBes_0_TM, indexing='ij')

    TM_2d, r_2d = np.meshgrid(TMM, r, indexing='ij')
    l_TM_2d, th_2d = np.meshgrid(l_TMM, th, indexing='ij')
    TQ31_TM = np.ones((len(TMM),len(TMM)), dtype = complex)
    TQ32_TM = np.ones((len(TMM),len(TMM)), dtype = complex)
    TQ23_TM = np.ones((len(TMM),len(TMM)), dtype = complex)
    
    SphBes_TM = jn(l_TM_2d, nR * TM_2d * r_2d)
    dSphBes_TM = jn(l_TM_2d, nR * TM_2d * r_2d, True)
    DSphBes_TM = SphBes_TM + nR * TM_2d * r_2d * dSphBes_TM
    Legendre_TM = ME.leg(th_2d, l_TM_2d, m)
    dLegendre_TM = ME.Dleg(th_2d, l_TM_2d, m, Legendre_TM)

    #integration
    for i in range(0, len(TMM)):
        t2 = SphBes_TM[i, :] * SphBes_TM[i:, :]
        t3 = DSphBes_TM[i, :] * DSphBes_TM[i:, :]
        q3 = Legendre_TM[i, :] * Legendre_TM[i:, :]
        q1 = q3 / (1. - th_2d[i, :] * th_2d[i, :])
        q2 = dLegendre_TM[i, :] * dLegendre_TM[i:, :] * (1. - th_2d[i, :] * th_2d[i, :])
        TQ31_TM[i:, i] = ME.TQ(r_2d[i:, :], th_2d[i:, :], sign, t3, q1)
        TQ32_TM[i:, i] = ME.TQ(r_2d[i:, :], th_2d[i:, :], sign, t3, q2)
        TQ23_TM[i:, i] = ME.TQ(r_2d[i:, :], th_2d[i:, :], sign, t2, q3)
      
    TQ31_TM = TQ31_TM / (t_01 * t_02)
    TQ32_TM = TQ32_TM / (t_01 * t_02) 
    TQ23_TM = TQ23_TM / (t_01 * t_02)

    #normalization grid
    A_1, A_2 = np.meshgrid(A_TMM, A_TMM, indexing='ij')

    #calculation of matrix elements between modes via ME module
    h_TM_TM = ME.V_TM_TM(delta_eps, nR, root_1, root_2, l_1, l_2, m, TQ31_TM, TQ32_TM, TQ23_TM, A_1, A_2) 
    
    h_TM_TM = h_TM_TM  / 2. / (np.sqrt(root_1) * np.sqrt(root_2))
    h_TM_TM = np.tril(h_TM_TM, -1).T + np.tril(h_TM_TM, 0)
    

    return h_TM_TM
    
def Hsolve_TM(N, h, delta_eps, LE, TM, nR, l_LE, l_TM, m): 

    l_TMM = sc.hstack((l_LE, l_TM))
    fact_TMM = -pow(-1, l_TMM)
    fact = np.outer(fact_TMM, fact_TMM)
    
    rho = np.sqrt(1. - h ** 2.) 
    
    delta = 1e-8     #to overcome integral discontinuities at 0 and 1
  
    ###CYLINDER: top
    R1 = h + delta
    R2 = 1. - delta

    r = np.linspace(R1, R2, N + 1)
    th = h / r
    sign = -1.
    
    V1 = V_TM(r, th, sign, delta_eps,LE, TM, nR, l_LE, l_TM, m)    
    
    ###CYLINDER: bottom
    V2 = fact * V1    

    ###CYLINDER: topside
    R1 = rho + delta
    R2 = 1. - delta
    r = np.linspace(R1, R2, N + 1)
    th = np.sqrt(1. - rho ** 2. / r ** 2.)
    sign = 1.
    
    V3 = V_TM(r, th, sign, delta_eps,LE, TM, nR, l_LE, l_TM, m)   
    
    ###CYLINDER: bottomside
    V4 = fact * V3
    
    ##############FULL-HAMILTIONIAN FORMATION
    root = sc.hstack((LE, TM))
    H = np.diag(1. / root) + V1 + V2 + V3 + V4
    
    #searching eigenvalues of H
    result = sc.linalg.eig(H)       
        
    return result


##### EXACT MATRIX ELEMENTS for sphere
#def V_TE_exact_sph(ksi, delta_eps, TE, nR, l_TE):
#    L = l_TE
#    l_1, l_2 = np.meshgrid(l_TE, l_TE, indexing='ij')
#    fact = np.ones(len(l_TE)) - abs(np.sign(l_1-l_2))
#    
#    norm = delta_eps / (nR ** 2. - 1.) * ksi ** 3.
#
#    z = ksi * nR * TE
#    x, y =  np.meshgrid(ksi * nR * TE, ksi * nR * TE) 
#    
#    SphB0sqrd0 = sp.spherical_jn(L, z / ksi) ** 2.
#    SphB0sqrd = sp.spherical_jn(L, z) ** 2.
#    SphBm1 = sp.spherical_jn(L - 1, z)
#    SphBp1 = sp.spherical_jn(L + 1, z)  
#    SphB0x0 = sp.spherical_jn(L, x / ksi)
#    SphB0y0 = sp.spherical_jn(L, y / ksi)
#    SphB0x = sp.spherical_jn(L, x)
#    SphB0y = sp.spherical_jn(L, y)
#    SphBm1x = sp.spherical_jn(L - 1, x)
#    SphBm1y = sp.spherical_jn(L - 1, y)
#    
#    h_diag = norm * (SphB0sqrd - SphBm1 * SphBp1) / SphB0sqrd0 
#    h_TE_TE = norm * 2. / (x * x - y * y) * (y * SphBm1y * SphB0x - x * SphBm1x * SphB0y) /  (SphB0x0 * SphB0y0)
#    
#    for i in range(0, len(TE)): h_TE_TE[i, i] = h_diag[i]
#    
#    h_TE_TE = h_TE_TE * fact
#    
#    return h_TE_TE
#
#def V_Surface_TE(TE, nR, l_TE):
#    L = l_TE
#    l_1, l_2 = np.meshgrid(l_TE, l_TE, indexing='ij')
#    fact = np.ones(len(l_TE)) - abs(np.sign(l_1-l_2))
#    
#    norm = 1 / (nR * nR - 1.)
#
#    x, y =  np.meshgrid(nR * TE, nR * TE) 
#    L1, L2 =  np.meshgrid(l_TE, l_TE) 
#    
#    SphBx = sp.spherical_jn(L, x)
#    SphBy = sp.spherical_jn(L, y)
#    dSphBx = sp.spherical_jn(L, x, True)
#    dSphBy = sp.spherical_jn(L, y, True)
#    L_matrix = L1 * (L1 + 1)
#    
#    h = fact * norm * (L_matrix - y * ( y / (nR * nR) + dSphBy / SphBy * (1 + x * dSphBx / SphBx)))
#    
#    return h

    
#
#def AnalyticBand(N, lambd2, delta_eps, TE, nR, l_TE, m, V_cyl): 
#
#    fact_TE = pow(-1, l_TE)
#    fact = np.outer(fact_TE, fact_TE)
#    
#    h2 = 1. / np.sqrt(1 + 4. * lambd2 * lambd2)
#    rho2 = np.sqrt(1. - h2 * h2)  
#    
#    delta = 1e-8     #to overcome integral discontinuities at 0 and 1
#    
#    V_cyl_1 = V_cyl
#    
#    ###CYLINDER: top - 2
#    R1 = h2 + delta
#    R2 = 1. - delta
#
#    r = np.linspace(R1, R2, N + 1)
#    th = h2 / r
#    sign = -1.
#    
#    V1 = V_TE(r, th, sign, delta_eps, TE, nR, l_TE, m)    
#    
#    ###CYLINDER: bottom - 2
#    V2 = fact * V1    
#
#    ###CYLINDER: topside - 2
#    R1 = rho2 + delta
#    R2 = 1. - delta
#    r = np.linspace(R1, R2, N + 1)
#    th = np.sqrt(1. - rho2 * rho2 / (r * r))
#    sign = 1.
#    
#    V3 = V_TE(r, th, sign, delta_eps, TE, nR, l_TE, m)  
#    
#    ###CYLINDER: bottomside - 2
#    V4 = fact * V3
#    
#    V_cyl_2 = V1 + V2 + V3 + V4
#    
#    ##############FULL-matrix FORMATION
#    W = (- V_cyl_1 + V_cyl_2)
#        
#    return W, V_cyl_2