#route to the modules with program parts
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))
 
###import built-in science packets
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sc
import cmath
from matplotlib import cm
#from scipy.misc import imread
#import matplotlib.cbook as cbook


###import my modules
import eigh
import save_load as svld


### Timer
#startTime = time.time()
#elapsedTime = time.time() - startTime
#print 'TIME:', int(elapsedTime * 1000)


def Normalization(TE, coeff, H):
    coeff_l = coeff.reshape(1, len(TE))
    coeff_r = coeff.reshape(len(TE), 1)
    normalization = np.dot(np.dot(coeff_l, H) , coeff_r)
    coeff_normalized = coeff / np.sqrt(normalization)
    return coeff_normalized
        


#def AnalyticsOneBand(k, coeff, N, N_lam, lambd_start, lambd_max, delta_eps, TE, nR, l_TE, m, V_cyl):
#    lambd_arr = np.linspace(lambd_start + 1e-10, lambd_max, N_lam)
#    kappa = np.zeros(len(lambd_arr), dtype = complex)
#    i = 0
#    for lambd in lambd_arr:
#        startTime = time.time()
#        W = eigh_new.AnalyticBand(N, lambd, delta_eps, TE, nR, l_TE, m, V_cyl)
#        V =  np.dot(np.dot(coeff.reshape(1, len(TE)), W), coeff.reshape(len(TE), 1))
#        det = (1. + V)
#        kappa[i] = k / det[0, 0]
#        i = i + 1
#        elapsedTime = time.time() - startTime
#        print ('TIME:', round(elapsedTime, 3), 's')
#        print (round( float(i) / len(lambd_arr) * 100, 2), "%")
#        
#    return lambd_arr, kappa
#    
#
#def AnalyticsTwoBand(k1, k2, coeff1, coeff2, N, N_lam, lambd_start, lambd_max, delta_eps, TE, nR, l_TE, m, V_cyl):
#    lambd_arr = np.linspace(lambd_start , lambd_max, N_lam)
#    kappa1 = np.zeros(len(lambd_arr), dtype = complex)
#    kappa2 = np.zeros(len(lambd_arr), dtype = complex)
#    i = 0
#    for lambd in lambd_arr:
#        startTime = time.time()
#        W, U = eigh_new.AnalyticBand(N, lambd, delta_eps, TE, nR, l_TE, m, V_cyl)
#        coeff1_l = coeff1.reshape(1, len(TE))
#        coeff2_l = coeff2.reshape(1, len(TE))
#        coeff1_r = coeff1.reshape(len(TE), 1)
#        coeff2_r = coeff2.reshape(len(TE), 1)
#        V11 =  np.dot(np.dot(coeff1_l , W), coeff1_r)
#        V12 =  np.dot(np.dot(coeff1_l , W), coeff2_r)
#        V22 = np.dot(np.dot(coeff2_l , W), coeff2_r)
##        print (V11,V12,V22, k1,k2)
#        V_av = (V11 + V22) / 2.
#        dV = (V11 - V22) / 2.
#        k_av = (k1 + k2) / 2.
#        dk = (k1 - k2) / 2.
#        det = (1. + V11) * (1. + V22) - V12 ** 2.
#        term1 = k_av * (1. + V_av) - dk * dV
#        term2 = np.sqrt((dk * (1. + V_av) - k_av * dV) ** 2. + k1 * k2 * V12 ** 2.)
#        kappa1[i] = ((term1 - term2) / det)[0, 0]
#        kappa2[i] = ((term1 + term2) / det)[0, 0]
#        i = i + 1
#        elapsedTime = time.time() - startTime
#        print ('TIME:', round(elapsedTime, 3), 's')
#        print (round( float(i) / len(lambd_arr) * 100, 2), "%")
#        
#    return lambd_arr, kappa1, kappa2

#def AnalyticsTwoBand_modified(k1, k2, coeff1, coeff2, N, N_lam, lambd_start, lambd_max, delta_eps, TE, nR, l_TE, m, V_cyl):
#    lambd_arr = np.linspace(lambd_start , lambd_max, N_lam)
#    kappa1 = np.zeros(len(lambd_arr), dtype = complex)
#    kappa2 = np.zeros(len(lambd_arr), dtype = complex)
#    i = 0
#    
#    coeff1_l = coeff1.reshape(1, len(TE))
#    coeff2_l = coeff2.reshape(1, len(TE))
#    coeff1_r = coeff1.reshape(len(TE), 1)
#    coeff2_r = coeff2.reshape(len(TE), 1)
#    
#    V_old = V_cyl
#    k_old_1 = k1 
#    k_old_2 = k2 
#    for lambd in lambd_arr:
#        startTime = time.time()
#        W, V_new = eigh_new.AnalyticBand(N, lambd, delta_eps, TE, nR, l_TE, m, V_old)
#
#        V11 =  np.dot(np.dot(coeff1_l , W), coeff1_r)
#        V12 =  np.dot(np.dot(coeff1_l , W), coeff2_r)
#        V22 = np.dot(np.dot(coeff2_l , W), coeff2_r)
##        print (V11,V12,V22, k1,k2)
#        V_av = (V11 + V22) / 2.
#        dV = (V11 - V22) / 2.
#        k_av = (k1 + k2) / 2.
#        dk = (k1 - k2) / 2.
#        det = (1. + V11) * (1. + V22) - V12 ** 2.
#        term1 = k_av * (1. + V_av) - dk * dV
#        term2 = np.sqrt((dk * (1. + V_av) - k_av * dV) ** 2. + k1 * k2 * V12 ** 2.)
#        kappa1[i] = ((term1 - term2) / det)[0, 0]
#        kappa2[i] = ((term1 + term2) / det)[0, 0]
#        
#        V_old = V_new
#        i = i + 1
#        elapsedTime = time.time() - startTime
#        print ('TIME:', round(elapsedTime, 3), 's')
#        print (round( float(i) / len(lambd_arr) * 100, 2), "%")
#        
#    return lambd_arr, kappa1, kappa2



def Loader(m, pol, k_max, eps, delta_eps, nR, l_min, l_max, h_arr, N):
    # pol: 1 = TE, 2 = TM
    
    if m == 0:
        if pol == 1:           
            TE, l_TE = svld.load_TE(k_max, eps, l_min, l_max)
            
            root = np.zeros((len(h_arr), np.size(TE)), dtype = complex)

            
            print ('N_basis = ', np.size(root, 1))
            print ('l_TE_max = ', np.amax(l_TE))
            print ('TE_max = ', np.amax(np.real(TE)))
            
            startTime0 = time.time() 

            for i in range(0, len(h_arr)):
               startTime = time.time()
               
               h = h_arr[i]        
               
               res, eigvec = eigh.Hsolve_TE(N, h, delta_eps, TE, nR, l_TE, m)
               nu = 1 / res
               
               rho = np.sqrt(1. - h ** 2.)
               root[i, :] = rho * nu
                         
              
               elapsedTime = time.time() - startTime
               print ('TIME:', round(elapsedTime, 3), 's')
               print (round( (i+1) / len(h_arr) * 100, 2), "%")
            elapsedTime0 = time.time() - startTime0
            print ('TIME TOTAL:', round(elapsedTime0, 3), 's')

        
        elif pol == 2:
            TM, l_TM = svld.load_TM(k_max, eps, l_min, l_max)
            LE, l_LE = svld.load_LE(l_min, 20)
            
            root = np.zeros((len(h_arr), np.size(LE)+np.size(TM)), dtype = complex)

            
            print ('N_basis = ', np.size(root, 1))
            print ('l_TM_max = ', np.amax(l_TM))
            print ('TM_max = ', np.amax(np.real(TM)))
            
            startTime0 = time.time() 

            for i in range(0, len(h_arr)):
               startTime = time.time()
               
               h = h_arr[i]        
               
               res, eigvec = eigh.Hsolve_TM(N, h, delta_eps, LE, TM, nR, l_LE, l_TM, m)
               nu = 1 / res
               
               rho = np.sqrt(1. - h ** 2.)
               root[i, :] = rho * nu
                         
              
               elapsedTime = time.time() - startTime
               print ('TIME:', round(elapsedTime, 3), 's')
               print (round( (i+1) / len(h_arr) * 100, 2), "%")
            elapsedTime0 = time.time() - startTime0
            print ('TIME TOTAL:', round(elapsedTime0, 3), 's')
        
        elif pol == 3:
            TE, l_TE = svld.load_TE(k_max, eps, l_min + 1, l_max + 1)
            
            root = np.zeros((len(h_arr), np.size(TE)), dtype = complex)

            
            print ('N_basis = ', np.size(root, 1))
            print ('l_TE_max = ', np.amax(l_TE))
            print ('TE_max = ', np.amax(np.real(TE)))
            
            startTime0 = time.time() 
 
            for i in range(0, len(h_arr)):
               startTime = time.time()
               
               h = h_arr[i]
                            
               res, eigvec = eigh.Hsolve_TE(N, h, delta_eps, TE, nR, l_TE, m)
               nu = 1 / res
                       
               rho = np.sqrt(1. - h ** 2.)
               root[i, :] = rho * nu
              
               elapsedTime = time.time() - startTime
               print ('TIME:', round(elapsedTime, 3), 's')
               print (round( (i+1) / len(h_arr) * 100, 2), "%")
            elapsedTime0 = time.time() - startTime0
            print ('TIME TOTAL:', round(elapsedTime0, 3), 's')

        
    elif ((m%2 == 1) and (pol == 1)) or ((m%2 == 0) and (pol == 2)):
            
        TE, l_TE = svld.load_TE(k_max, eps, l_min+1, l_max+1)
        TM, l_TM = svld.load_TM(k_max, eps, l_min, l_max)
        LE, l_LE = svld.load_LE(l_min, 20)            
        root = np.zeros((len(h_arr), np.size(TE)+np.size(TM)+np.size(LE)), dtype = complex)        
        
        
        print ('N_basis = ', len(root))
        print ('l_TE_max = ', np.amax(l_TE))
        print ('l_TM_max = ', np.amax(l_TM))
        print ('TE_max = ', np.amax(np.real(TE)))
        print ('TM_max = ', np.amax(np.real(TM)))
        
        startTime0 = time.time() 
        for i in range(0, len(h_arr)):
           startTime = time.time()
               
           h = h_arr[i]
           res = eigh.Hsolve(N, h, delta_eps, TE, LE, TM, nR, l_TE, l_LE, l_TM, m)
           nu = 1 / res
               
           rho = np.sqrt(1. - h ** 2.)
           root[i, :] = rho * nu
           
           elapsedTime = time.time() - startTime
           print ('TIME:', round(elapsedTime, 3), 's')
           print (round( i / len(h_arr) * 100, 2), "%")
        elapsedTime0 = time.time() - startTime0
        print ('TIME TOTAL:', round(elapsedTime0, 3), 's')
    
    
    else:
        TE, l_TE = svld.load_TE(k_max, eps, l_min, l_max)
        TM, l_TM = svld.load_TM(k_max, eps, l_min+1, l_max+1)
        LE, l_LE = svld.load_LE(l_min+1, 20)            
        root = np.zeros((len(h_arr), np.size(TE)+np.size(TM)+np.size(LE)), dtype = complex)
        print ('N_basis = ', len(root))
        print ('l_TE_max = ', np.amax(l_TE))
        print ('l_TM_max = ', np.amax(l_TM))
        print ('TE_max = ', np.amax(np.real(TE)))
        print ('TM_max = ', np.amax(np.real(TM)))
        
        startTime0 = time.time() 
        for i in range(0, len(h_arr)):
           startTime = time.time()
               
           h = h_arr[i]
           res = eigh.Hsolve(N, h, delta_eps, TE, LE, TM, nR, l_TE, l_LE, l_TM, m)
           nu = 1 / res
               
           rho = np.sqrt(1. - h ** 2.)
           root[i, :] = rho * nu
           
           elapsedTime = time.time() - startTime
           print ('TIME:', round(elapsedTime, 3), 's')
           print (round( i / len(h_arr) * 100, 2), "%")
        elapsedTime0 = time.time() - startTime0
        print ('TIME TOTAL:', round(elapsedTime0, 3), 's')
        
    return root
            


#### MAIN PROGRAM BODY
#eps_e = 1.32 ** 2  #water
#eps_i = 3.50 ** 2  #Si
eps_e = 1
eps_i = 12.6527
eps = eps_i / eps_e
nR = np.sqrt(eps)     #fractive index
delta_eps = - eps + 1.   #perturbation of sphere permittivity


#####  FIND AND SAVE ROOTS
###coordinates of initial rectangle (in complex plane) to find zeros of secular equation inside it
#x = 10
#y = -10
#w = x + 1e-3
#h = abs(y) + 1e-3
#
#l_min = 1
#l_max = 85
#
#for l1 in range(l_min, l_max + 1 , 2):
#    startTime = time.time()
##    svld.save_TE(x, y, w, h, nR, eps, l1)
#    svld.save_TM(x, y, w, h, nR, eps, l1)
#    elapsedTime = time.time() - startTime
#    print ('TIME:', round(elapsedTime, 3), 's')
#print ('')
#
#for l1 in range(l_min + 1, l_max + 2 , 2):
#    startTime = time.time() 
#    svld.save_TE(x, y, w, h, nR, eps, l1)  
#    svld.save_TM(x, y, w, h, nR, eps, l1)
#    elapsedTime = time.time() - startTime
#    print ('TIME:', round(elapsedTime, 3), 's')


#####   LOAD ROOTS 
k_max = 14
l_min = 1
l_max = 80


m = 2
pol = 1     ## polarization of incident light (normal to side edge):
            ## 1 = TE (excite only even modes), 2 = TM (excite only odd modes)
     
N = 200     ## size of integration grid
N_h = 400    ## size of R/H grid 



### R/H grid (=lambd) 

lambd = np.linspace(0.5, 0.9, N_h)
#lambd1 = np.linspace(0.1, 1.0, 300)


h_arr = 1. / np.sqrt(1 + 4. * lambd ** 2.)



root  = Loader(m, pol, k_max, eps, delta_eps, nR, l_min, l_max, h_arr, N)
root = root / np.sqrt(eps_e)  
Q = abs(np.real(root) / 2 / np.imag(root))


#root_0 = np.load(os.path.dirname(__file__)+'/result_data/r_eps'+str(round(eps, 2))+'_m'+str(m)+'_lmbd'+str(round(lambd[0],2))+'-'+str(round(lambd[-1],2))+'.npy')
#root_1 = np.load(os.path.dirname(__file__)+'/result_data/r_eps'+str(round(eps, 2))+'_m'+str(m)+'_lmbd'+str(round(lambd[0],2))+'-'+str(round(lambd[-1],2))+'.npy')
#root_2 = np.load(os.path.dirname(__file__)+'/result_data/r_eps'+str(round(eps, 2))+'_m'+str(m)+'_lmbd'+str(round(lambd[0],2))+'-'+str(round(lambd[-1],2))+'.npy')

#Q_0 = abs(np.real(root_0) / 2./ np.imag(root_0))
#Q_1 = abs(np.real(root_1) / 2./ np.imag(root_1))
#Q_2 = abs(np.real(root_2) / 2./ np.imag(root_2))

np.save(os.path.dirname(__file__)+'/result_data/r_eps'+str(round(eps, 2))+'_m'+str(m)+'_pol'+str(pol)+'_lmbd'+str(round(lambd[0],2))+'-'+str(round(lambd[-1],2))+'.npy', root)
#np.save(os.path.dirname(__file__)+'/result_data/Q_eps'+str(int(eps))+'_m'+str(m)+'_1.npy', Q)
#np.save(os.path.dirname(__file__)+'/result_data/lmbd_eps'+str(int(eps))+'_m'+str(m)+'_1.npy', lambd)



####moments!


### PROBLEMS OF LEFT AND RIGHT
#V_cyl = eigh_new.V_TE_exact_sph(1., eps, TE, nR, l_TE) / 2
#H_1 = V_cyl * 2.
#B = np.eye(TE.shape[0]) + V_cyl
#           
#res, eigvec = sc.linalg.eig(B, np.diag(TE))
#nu = 1 / res
##  
##
##
##fact = nu[2]**2 - nu[1]**2 
##    
##x, y = np.meshgrid(1/TEE**2., 1/TEE**2. , indexing = 'ij')    
##aaa1 = np.dot(np.dot(eigvec[:,1].reshape(1, len(TE)), fact*2*H_1 + (x-y)*H_1 ), eigvec_r[:,2].reshape(len(TE),1))
##aaa2 = np.dot(np.dot(eigvec[:,1].T, -4*fact*np.eye(len(TE)) + (x-y)*H_1 ), eigvec_r[:,2])
#
#
#TEE, l_TEE = svld.load_TE(k_max, 2*eps, l_min, l_max)
#HH_1 = eigh_new.V_TE_exact_sph(1., 2*eps, TEE, np.sqrt(2)*nR, l_TEE) 
#
#
#alpha1 = np.sqrt(2 * np.dot(np.dot(eigvec[:,3].T, H_1) , eigvec[:,3])/HH_1[3,3])
#alpha2 = np.sqrt(2 * np.dot(np.dot(eigvec[:,4].T, H_1) , eigvec[:,4])/HH_1[7,7])
#
#eig1 = eigvec[:,3] / alpha1
#eig2 = eigvec[:,4] / alpha2
#
#np.dot(np.dot(eigvec_norm[:,0].T, 2*H_1+H_3 / (nu[0] * nu[0])) , eigvec_norm[:,0])
##- HH_1[4,6]
##np.dot(eigvec_r[:,1],H,)
#



### PROPER NORMALIZATION

#eigvec_norm = np.zeros(eigvec.shape, dtype = complex)
#
#H_1 = eigh_new.V_TE_exact_sph(1., eps, TE, nR, l_TE)
#H_3 = eigh_new.V_Surface_TE(TE, nR, l_TE)
#HH_3 = eigh_new.V_Surface_TE(TEE, nR, l_TEE)
#for i in range(len(TE)):
#    k = nu[i]
#    H_2 = 2 * (np.diag(TE) / k  - np.eye(TE.shape[0]))
#    H = H_1 + H_2 + H_3 / (k * k)
#    eigvec_norm[:, i] = Normalization(TE, eigvec[:, i], H)
#    print (round( i / len(TE) * 100, 2), "%")

    

### TWO BAND MODEL
    
#N_TB = 2000
#N_h_TB = 50
#
#k1 = nu[2]
#k2 = nu[3]
#lambd_start = lambd
#lambd_max = 1.0
#coeff1 = eigvec_norm[:, 2]
#coeff2 = eigvec_norm[:, 3]
#lambd_arr, kappa1, kappa2 = AnalyticsTwoBand(k1, k2, coeff1, coeff2, N_TB, N_h_TB, lambd_start, lambd_max, delta_eps, TE, nR, l_TE, m, V_cyl)
#rho1 = 2. * lambd_arr / np.sqrt(1 + 4. * lambd_arr ** 2.)
#
#kappa1 = kappa1 * rho1
#kappa2 = kappa2 * rho1
#
#Q1 = abs(np.real(kappa1) / 2./ np.imag(kappa1))
#Q2 = abs(np.real(kappa2) / 2./ np.imag(kappa2))
#
#plt.figure(figsize=(7,7))
#plt.scatter(np.real(kappa1), lambd_arr, color = 'b', s = Q1/10, marker = 'o')
#plt.scatter(np.real(kappa2), lambd_arr, color = 'r', s = Q2/10, marker = 'o')
#plt.ylim(0.5, 1.)
#plt.xlim(0.7, 1.2)
#
#plt.xlabel('Re[\omega * r / c')
#plt.ylabel('r / L')
#plt.show
#





#k1 = nu[1]    
#k2 = nu[2]
#x, y = np.meshgrid(TE * TE, TE * TE, indexing = 'ij')
#fact = k2 * k2 - k1 * k1 
#H_2 = -2 * np.eye(len(TE))
#H = fact * 2 * H_1 + H_1 * (x-y)
#a = np.dot(np.dot(eigvec_r[:,1].T, H_1), eigvec_r[:,2])
# 


### N-BAND MODEL

#
#N = 200 #number of integration grid points
#N_h = 100
#
#alpha = np.zeros(np.size(nu)) 
#
#lambd_arr = np.linspace(lambd, 1.0, N_h)  
#
#k = 0
#
#for lambd_iter in lambd_arr:
#    startTime = time.time()
#    A = np.zeros((len(TE), len(TE)), dtype = complex)
#    W = eigh_new.AnalyticBand(N, lambd_iter, delta_eps, TE, nR, l_TE, m, V_cyl)
#    for i in range(0, len(TE)):
#        for j in range(0, len(TE)):
#            coeff_l = eigvec_norm[:, i].reshape(1, len(TE))
#            coeff_r = eigvec_norm[:, j].reshape(len(TE), 1)
#            A[i, j] =  np.dot(np.dot(coeff_l , W), coeff_r)[0,0] + (1 - abs(np.sign(i-j)))
#    eigval = sc.linalg.eigvals(A, np.diag(nu)) 
#    res = 1 / eigval
#    rho_new = 2. * lambd_iter / np.sqrt(1 + 4. * lambd_iter ** 2.)
#    alpha = sc.vstack((alpha,  rho_new * res))
#    elapsedTime = time.time() - startTime
#    print ('TIME:', round(elapsedTime, 3), 's')
#    k = k + 1.
#    print (round( k / len(lambd_arr) * 100, 2), "%")
#        
#alpha = np.delete(alpha,0,0)
#alpha = alpha / np.sqrt(eps_e)    
#Q_a = abs(np.real(alpha) / 2./ np.imag(alpha))    
#   
#
#np.save(os.path.dirname(__file__)+'/alpha.npy', alpha)
#
#    


#ONE-BAND MODEL

#N_TB = 200
#N_h_TB = 100
#
#k = nu[1]
#lambd_start = lambd
#lambd_max = 1.0
#coeff = eigvec_norm[:, 1]
#
#
#lambd_arr, kappa = AnalyticsOneBand(k, coeff, N_TB, N_h_TB, lambd_start, lambd_max, delta_eps, TE, nR, l_TE, m, V_cyl)
#rho1 = 2. * lambd_arr / np.sqrt(1 + 4. * lambd_arr ** 2.)
#kappa = kappa * rho1
#Q_kapp = abs(np.real(kappa) / 2./ np.imag(kappa))
#
#plt.figure(figsize=(7,7))
#plt.scatter(np.real(kappa), lambd_arr, color = 'g', s = Q_kapp, marker = 'o')
#plt.ylim(0.5, 1.0)
#plt.xlim(0.4, 0.52)



#plt.figure(figsize=(5,5))
##datafile = cbook.get_sample_data('Full_map.png')
##img = imread(datafile)
#plt.plot(np.real(root), lambd, color = 'k',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
##plt.plot(np.real(root1)*2.5, np.linspace(0.3, 1.05, 70), color = 'w',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 4)
##plt.plot(np.real(root1), lambd, color = 'g', marker = 'o', linewidth = 0., mew = '1.', markersize = 3)
##plt.plot(np.real(root2), lambd, color = 'b', marker = 'o', linewidth = 0., mew = '1.', markersize = 3)
##plt.plot(np.real(root3), lambd, color = 'g', marker = 'o', linewidth = 0., mew = '1.', markersize = 3)
##plt.imshow(img, zorder=0, extent=[0.25, 0.875, 0.3, 1.3])
##plt.imshow(img, zorder=0, extent=[0.25, 0.875, 0.1, 1.3])
### FULL
#plt.ylim(lambd[0],lambd[-1])
#plt.xlim(0.2, 1.0)
##plt.xlim(1e1,1e3)
##plt.xlabel('Frequency: \omega / c * R')
##plt.ylabel('ratio : R / L')
##plt.savefig('full.eps', format="eps")


#
#plt.figure(figsize=(10,8))
##datafile = cbook.get_sample_data('Full_map.png')
##img = imread(datafile)
#plt.semilogy(lambd, Q[:,11], color = 'r',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.semilogy(lambd, Q[:,13], color = 'r',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.semilogy(lambd, Q1[:,5], color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.semilogy(lambd, Q1[:,4], color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.semilogy(lambd, Q1[:,3], color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
##plt.plot(np.real(root1), lambd, color = 'w',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
##plt.plot(np.real(root1)*2.5, np.linspace(0.3, 1.05, 70), color = 'w',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 4)
##plt.plot(np.real(root1), lambd, color = 'g', marker = 'o', linewidth = 0., mew = '1.', markersize = 3)
##plt.plot(np.real(root2), lambd, color = 'b', marker = 'o', linewidth = 0., mew = '1.', markersize = 3)
##plt.plot(np.real(root3), lambd, color = 'g', marker = 'o', linewidth = 0., mew = '1.', markersize = 3)
##plt.imshow(img, zorder=0, extent=[0.25, 0.875, 0.3, 1.3])
##plt.imshow(img, zorder=0, extent=[0.25, 0.875, 0.1, 1.3])
##
### FULL
#plt.ylim(4e1, 1e5)
#plt.xlim(lambd[0],lambd[-1])
#plt.savefig('Q.eps', format="eps")




#rad1 = abs(np.real(root) * 750 / (2 * np.pi))
#L1 = np.zeros((len(lambd), len(nu)))
#D1 = 2 * rad1
#rad2 = abs(np.real(root) * 800 / (2 * np.pi))
#L2 = np.zeros((len(lambd), len(nu)))
#D2 = 2 * rad2
#rad3 = abs(np.real(root) * 850 / (2 * np.pi))
#L3 = np.zeros((len(lambd), len(nu)))
#D3 = 2 * rad3
#i = 0
#
#for i in range(0, len(lambd)):
#    L1[i, :] = rad1[i, :] / lambd[i]
#    L2[i, :] = rad2[i, :] / lambd[i]
#    L3[i, :] = rad3[i, :] / lambd[i]
#    i = i + 1


#H = np.linspace(615, 715, 200)
#
#lambd_arr, H_arr = np.meshgrid(lambd, H, indexing='ij')
#R_arr = lambd_arr * H_arr
#
#root_arr, H_arr = np.meshgrid(root[:,6], H, indexing='ij')
#
#wvln = 2*np.pi*R_arr/abs(np.real(root_arr))
#
#lambda_arr = 2*np.pi*R_arr/abs(np.real(root_arr))
#Q_arrr = abs(np.real(root_arr)/2/np.imag(root_arr+1e-25))
#
#plt.figure(figsize = (10, 9))
#levels = np.linspace(1550, 1900, 22)
#cp = plt.contour(H_arr , R_arr , lambda_arr, levels, cmap=cm.rainbow)
#plt.colorbar(cp, orientation='vertical')
#plt.plot(H, 0.707*H, color = 'k',  linewidth = 3 )
#plt.title('Wavelength, nm')
#plt.xlabel('Height, nm')
#plt.ylabel('Radius, nm')
#plt.xlim(np.amin(H), np.amax(H))
#plt.ylim(420,520) 
#plt.savefig('map_wvln.eps')
#plt.show()
#
#
#
#plt.figure(figsize = (10, 9))
#levels = np.linspace(10, 160, 31)
#cp = plt.contour(H_arr , R_arr ,Q_arrr , levels, cmap=cm.rainbow)
#plt.plot(H, 0.707*H, color = 'k',  linewidth = 3)
#plt.colorbar(cp, orientation='vertical')
#plt.title('Quality factor, a.u.')
#plt.xlabel('Height, nm')
#plt.ylabel('Radius, nm')
#plt.xlim(np.amin(H), np.amax(H))
#plt.ylim(420,520) 
#plt.savefig('map_Q.eps')
#plt.show()
#
#np.savetxt("wvln.txt",lambda_arr, fmt='%4.1f')
#np.savetxt("R_arr.txt",R_arr, fmt='%3.1f')
#
#np.savetxt("H_arr.txt",H_arr, fmt='%3.1f')
#np.savetxt("Q_arr.txt",Q_arrr, fmt='%3.1f')

#plt.figure(figsize = (5.5, 9))
#levels = np.linspace(1.23, 2.2, 100)
#cp = plt.contourf(H_arr , R_arr , np.log10(abs(np.real(root_arr)/2/np.imag(root_arr+1e-25))), levels, cmap=cm.rainbow)
#plt.plot(H, 0.707*H, color = 'w',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.colorbar(cp, orientation='vertical')
##plt.title('Plot')
##plt.xlabel('k_x, a.u.')
##plt.ylabel('k_y, a.u.')
#plt.xlim(np.amin(H), np.amax(H))
#plt.ylim(420,520) 
#plt.show()

#r = (600 * lambd).reshape(len(lambd), 1)
#R = r
#for i in range(len(root[0,:]) - 1):
#    R = sc.hstack((R, r))
#wavelength = 2 * np.pi * R / np.real(root)

#D = 2 * r

#R = 1055 * np.real(root) / (2 * np.pi)
#H = R / lambd.reshape(len(lambd), 1)

#H = 960 / lambd.reshape(len(lambd), 1)
#wavelength = 2 * np.pi * 960 / np.real(root)

#root_norm = root / (2 * np.pi)

#D_scale = R / 500 * 2
#H_scale = H / 500
#
#ksi = 1/lambd /2
Q=abs(np.real(root)/np.imag(root)/2)

Q2=np.zeros(np.shape(Q))
root2=np.zeros(np.shape(root), dtype='complex')

for i in range(0, len(root[0, :])):
    for j in range(0, N_h):
            if Q[j, i] < 1:
                Q2[j, i] = 1e-25
                root2[j, i] = 1000
            elif Q[j, i] > 20000:
                Q2[j, i] = 1e-25
                root2[j, i] = 1000
            else:
                Q2[j, i] = Q[j, i]
                root2[j, i] = root[j, i]
#
#Q1=abs(np.real(root1)/np.imag(root1)/2)
#
#Q3=np.zeros(np.shape(Q1))
#root3=np.zeros(np.shape(root1), dtype='complex')
#
#for i in range(0, len(root1[0, :])):
#    for j in range(0, N_h):
#            if Q1[j, i] < 10:
#                Q3[j, i] = 1e-25
#                root3[j, i] = 1000
#            elif Q1[j, i] > 10000:
#                Q3[j, i] = 1e-25
#                root3[j, i] = 1000
#            else:
#                Q3[j, i] = Q1[j, i]
#                root3[j, i] = root1[j, i]
                       
##            
#                
#for i in range(0, len(root_1[0, :])):
#    for j in range(0, N_h):
#            if Q_1[j, i] < 3:
#                Q_1[j, i] = 0
#
#for i in range(0, len(root_0[0, :])):
#    for j in range(0, N_h):
#        if j % 5 != 0:     
#            Q_0[j, i] = 1e-10
#            
#for i in range(0, len(root_1[0, :])):
#    for j in range(0, N_h):
#        if j % 5 != 0:     
#            Q_1[j, i] = 1e-10
#
#for i in range(0, len(root_2[0, :])):
#    for j in range(0, N_h):
#        if j % 5 != 0:     
#            Q_2[j, i] = 1e-10
#
#Q_1[250, 24] = 1e4
#Q_1[250, 25] = 1e4




#
#plt.figure(figsize=(8, 8))
##datafile = cbook.get_sample_data('L.png')
###img = imread(datafile)
##plt.plot(np.real(root),  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#for i in range(0, np.size(root,1)):
#    plt.scatter(R[:,i], wavelength[:,i], color = 'b' ,  s = Q[:,i] / 100 , marker = 'o')
#plt.xlim(100, 400)
#plt.ylim(400, 900)
##plt.xlim(0.25, 3)
##plt.xticks(np.arange(50, 700, 10))
##plt.yticks(np.arange(lambd[0], lambd[-1], (lambd[-1] - lambd[0]) / 30))
#plt.xlabel('disk radius, nm')
#plt.ylabel('wavelength, nm')
##plt.xlabel('X = R * \omega /c, a.u.')
##plt.ylabel('R / H, a.u.')
#plt.show
##plt.savefig(os.path.dirname(__file__)+'/1.eps', format="eps")
##plt.savefig('exp_sca.eps', format="eps")
#plt.close



#fact = 5e2
#plt.figure(figsize=(5, 5))
##for i in range(0, np.size(D_scale,1)):
##    plt.scatter(D_scale[:, i], ksi, color = 'k', s = Q[:, i] / 60, marker = 'o')
##for i in range(0, np.size(Q,1)):
##    plt.scatter(wavelength[:, i], lambd, color = 'k', s = Q[:, i] / 60, marker = 'o')
#plt.ylim(0.3, 1.4)
##plt.ylim(0.1, 2.0)
##plt.xlim(0.5, 4.)
#plt.xlim(0.2, 1.2)
##plt.xlim(530, 550)
##plt.xticks(np.arange(50, 700, 10))
##plt.yticks(np.arange(lambd[0], lambd[-1], (lambd[-1] - lambd[0]) / 20))
##plt.xlabel('size parameter x')
##plt.ylabel('aspect ratio, r/l')
##plt.xlabel('X = R * \omega /c, a.u.')
##plt.ylabel('R / H, a.u.')
#plt.show
##plt.savefig(os.path.dirname(__file__)+'/1.eps', format="eps")
##plt.savefig('Q.eps', format="eps")
#plt.close

#
#for i in range(0, np.size(root_0,1)):
#    for j in range(0, 1000):
#        if Q_0[j,i] == 1e1:
#            Q_0[j,i] = 1e-2
#
#for i in range(0, np.size(root_1,1)):
#    for j in range(0, 1000):
#        if Q_1[j,i] == 1e6:
#            Q_1[j,i] = 1e-2
#
#
#            
#for i in range(0, np.size(root_2,1)):
#    for j in range(0, 1000):
#        if Q_2[j,i] == 1e6:
#            Q_2[j,i] = 1e-2
#

plt.figure(figsize=(7, 7))
#datafile = cbook.get_sample_data('L.png')
##img = imread(datafile)
#plt.plot(np.real(root),  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
for i in range(0, np.size(root2,1)):
    plt.scatter(np.real(root2[:,i]),  lambd , color = 'b' ,  s = Q2[:,i] / 8000 , marker = 'o')
#for i in range(0, np.size(root3,1)):
#    plt.scatter(np.real(root3[:,i]),  lambd , color = 'b' ,  s = Q3[:,i] / 10 , marker = 'o')
#plt.plot(1.6773906166701396*np.ones((len(lambd),1)),lambd, color = 'r')
#plt.plot(2*1.6773906166701396*np.ones((len(lambd),1)),lambd, color = 'g')
#plt.plot(2*0.288*2*np.pi*np.ones((len(lambd),1)),lambd, color = 'g')
#plt.plot(np.linspace(0.5,4,np.size(root,1)),0.705*np.ones(np.size(root,1)), color = 'g')
#for i in range(0, np.size(root3,1)):
#    plt.scatter(np.real(root3[:,i]),  lambd , color = 'r' ,  s = Q3[:,i] / 200 , marker = 'o')
#for i in range(0, np.size(root2,1)):
#    plt.scatter(np.real(root2[:,i]),  lambd , color = 'r' ,  s = Q2[:,i] / 200 , marker = 'o')
#for i in range(0, np.size(root_0,1)):
#    plt.scatter(np.real(root_0[:,i]),  lambd , color = 'b' ,  s = Q_0[:,i] / 10 , marker = 'o')
#for i in range(0, np.size(root_1,1)):
#    plt.scatter(np.real(root_1[:,i]),  lambd , color = 'r' ,  s = Q_1[:,i] / 10 , marker = 'o')
plt.ylim(0.5, 0.9)
#plt.ylim(0.1, 2.0)
#plt.xlim(0.5, 4.)
plt.xlim(1.1, 4)
#plt.xlim(530, 550)
#plt.xticks(np.arange(50, 700, 10))
#plt.yticks(np.arange(lambd[0], lambd[-1], (lambd[-1] - lambd[0]) / 20))
#plt.xlabel('H, nm')
#plt.ylabel('H / D, a.u.')
#plt.xlabel('X = R * \omega /c, a.u.')
#plt.ylabel('R / H, a.u.')
plt.show
#plt.savefig(os.path.dirname(__file__)+'/1.eps', format="eps")
#plt.savefig('AlGaAs_FF_0.775.eps', format="eps")
#plt.close


#
#aa = np.zeros((231-180,3))
#
#aa[:,0] = lambd[180:231]
#aa[:,1] = abs(np.real(root2[180:231,11]))/2/np.pi
#aa[:,2] = Q2[180:231,11]
#
#np.savetxt('nonlinear_oBIC_2_SH_1.55um_AlGaAs.txt',aa, fmt='%3.4f')
#
#

#plt.figure(figsize=(4, 6))
#for i in range(2, 7):  
#    cp = plt.scatter(x_1_sc[:,i], lambd, c = np.log10(Q_1[:,i]), s = 1/Q_1[:,i] * fact , marker = 'o', cmap = cm.Blues_r)
#    cp.set_clim([1.5, 4])  
#for i in range(10, 16):
#    cp = plt.scatter(x_sc[:,i], lambd, c = np.log10(Q[:,i]), s = 1/Q[:,i] * fact , marker = 'o', cmap = cm.Reds_r)
#    cp.set_clim([1, 3])  
#plt.xlim(0.075, 0.175)
#plt.ylim(lambd[30], lambd[200])
##plt.yticks(np.linspace(lambd[0], lambd[-1], 16))
##plt.xlabel('k_x * a / (2\pi), a.u.')
##plt.ylabel('\omega * a / (2\pi c), a.u.')
#plt.savefig('Q_0.eps')
#plt.show



#plt.figure(figsize=(4, 6))
#plt.semilogx(Q2[196:234,9],  lambd[196:234], color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
##plt.semilogx(Q[:,4],  lambd, color = 'k',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.xlim(1e2, 2e3)
#plt.ylim(lambd[196], lambd[234])
##plt.savefig('Q_1.eps')
#plt.show
#plt.close
#


#plt.figure(figsize=(4, 6))
#for i in range(0, np.size(root,1)):
#    cp = plt.scatter(x_sc[:,i], lambd, c = np.log10(Q[:,i]), s = 1/Q[:,i] * fact , marker = 'o', cmap = cm.Reds_r)
#    cp.set_clim([1, 3])  
#plt.colorbar(cp, ticks=[1, 2, 3], orientation='horizontal')    
##plt.xlim(0, 0.5 / np.sqrt(2))
#plt.xlim(0.05, 0.25)
#plt.ylim(lambd[0], lambd[-1])
#plt.yticks(np.linspace(lambd[0], lambd[-1], 16))
##plt.xlabel('k_x * a / (2\pi), a.u.')
##plt.ylabel('\omega * a / (2\pi c), a.u.')
##plt.savefig('SPIE_m1.eps')
#plt.show
#
#





















#plt.figure(figsize=(6, 6))
##plt.plot(-np.imag(root),  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#for i in range(0, np.size(root,1)):
#    plt.scatter(-np.imag(root[:,i]),  lambd , color = 'k' ,  s = Q[:,i] / 2000 , marker = 'o')
#plt.ylim(lambd[0], lambd[-1])
#plt.xlim(0.5, 6)
#plt.xlabel('X = R * \omega /c, a.u.')
#plt.ylabel('R / H, a.u.')
#plt.show
#plt.close
#
#
#
#plt.figure(figsize=(6, 6))
#for i in range(0, np.size(root,1)):
#    plt.scatter(-np.imag(root[:,i]),  lambd , color = 'k' ,  s = Q[:,i] / 1000 , marker = 'o')
##plt.plot(-np.imag(root),  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.ylim(lambd[0], lambd[-1])
#plt.xlim(0.0001, 0.001)
#plt.xlabel('X = R * \omega /c, a.u.')
#plt.ylabel('R / H, a.u.')
#plt.show
#plt.close
#
#
#
#plt.figure(figsize=(6, 6))
##plt.semilogx(Q[:,3],  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(Q[:,14],  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(Q[:,15],  lambd, color = 'g',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
##plt.semilogx(Q[:,4],  lambd, color = 'k',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.ylim(lambd[0], lambd[-1])
#plt.xlim(1, 20)
#plt.xlabel('X = R * \omega /c, a.u.')
#plt.ylabel('R / H, a.u.')
#plt.show
#plt.close
#
#
#plt.figure(figsize=(6, 6))
##plt.semilogx(Q[:,3],  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(abs(np.real(root[:,14])),  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(abs(np.real(root[:,15])),  lambd, color = 'g',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(abs(np.real(root[:,13])),  lambd, color = 'g',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(abs(np.real(root[:,17])),  lambd, color = 'g',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
##plt.semilogx(Q[:,4],  lambd, color = 'k',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
##plt.ylim(0.4, lambd[-1])
#plt.xlim(1.29, 1.295)
#plt.xlabel('X = R * \omega /c, a.u.')
#plt.ylabel('R / H, a.u.')
#plt.show
#plt.close
#
#
#plt.figure(figsize=(6, 6))
##plt.semilogx(Q[:,3],  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(abs(np.imag(root[:,14])),  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(abs(np.imag(root[:,15])),  lambd, color = 'g',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(abs(np.imag(root[:,13])),  lambd, color = 'g',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(abs(np.imag(root[:,16])),  lambd, color = 'g',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
##plt.semilogx(Q[:,4],  lambd, color = 'k',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
##plt.ylim(0.4, lambd[-1])
#plt.xlim(0.1, 0.104)
#plt.xlabel('X = R * \omega /c, a.u.')
#plt.ylabel('R / H, a.u.')
#plt.show
#plt.close
#
#phase = -1j*np.log(root/abs(root))/np.pi
#      
#plt.figure(figsize=(6, 6))
#plt.plot(phase[:,4],  lambd, color = 'g',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(phase[:,5],  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
##plt.plot(phase[:,2],  lambd, color = 'g',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(phase[:,3],  lambd, color = 'g',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.ylim(0.4, lambd[-1])
#plt.xlim(-1e-2, 1e-2)
#plt.xlabel('X = R * \omega /c, a.u.')
#plt.ylabel('R / H, a.u.')
#plt.show
#plt.close
#
#
#plt.figure(figsize=(6, 6))
##plt.semilogx(Q[:,3],  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(abs(root[:,5]),  lambd, color = 'b',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.plot(abs(root[:,2]),  lambd, color = 'g',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
##plt.semilogx(Q[:,4],  lambd, color = 'k',  linewidth = 0.,marker = 'o', mew = '1.', markersize = 3)
#plt.ylim(0.4, lambd[-1])
#plt.xlim(0.3, 1.0)
#plt.xlabel('X = R * \omega /c, a.u.')
#plt.ylabel('R / H, a.u.')
#plt.show
#plt.close

#plt.figure(figsize=(7,6))
#plt.semilogy(np.real(TE), -np.imag(TE), color = 'b', marker = 'o', linewidth = 0., mew = '1.') 
##plt.semilogy(np.real(TM), -np.imag(TM), color = 'y', marker = 'o', linewidth = 0., mew = '1.') 
##plt.semilogy(np.real(nu), -np.imag(nu), color = 'r', marker = '+', linewidth = 0., mew = '1.') 
#plt.ylim(1e-30, 2.5e1)
#plt.xlim(0., 25.0) 
#plt.show



#plt.figure(figsize=(10,7))
##datafile = cbook.get_sample_data('L.png')
###img = imread(datafile)
#ax = plt.gca()
#for i in range(0,np.size(root,1)):
#    ax.scatter(np.real(TE), -np.imag(TE), color = 'b', s = coeff2*1000, marker = 'o')
#    ax.scatter(np.real(root[:,11]), -np.imag(root[:,11]), color = 'r', s = 45, marker = 's')
#ax.set_yscale('log')
###for i in range(0,np.size(root2,1)):
###    plt.scatter(np.real(root2[:,i])*2, lambd, color = 'r', s = Q2[:, i]/10, marker = 'o')
###plt.imshow(img, zorder=0, extent=[0.33*2, 0.65*2, 0.3, 1.05])
##    
### FULL
#plt.ylim(1e-9,2e1)
#plt.xlim(-20., 20.)


#np.save(os.path.dirname(__file__)+'/Q_n/p1_S_root_eps80_l1-30_m1_k'+str(int(k_max))+'_Nb40_N40.npy', root)
#root = np.load(os.path.dirname(__file__)+'/Q_n/p1_S_root_eps80_l1-30_m1_k20_Nb20_N40.npy')


