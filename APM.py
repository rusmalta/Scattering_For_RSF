import os, sys
sys.path.append(os.path.dirname(__file__))
import scipy as sc
import numpy as np

##### Argument principle method    
    
def Int0_APM_Re(z, shift, tens, F, l, nR):
    z = tens * z + shift
    f, df = F(z, l, nR)
    weight = 1.
    res = 1. / (2 * np.pi *1j ) * df / f * weight
    return np.real(res)
    
def Int0_APM_Im(z, shift, tens, F, l, nR):
    z = tens * z + shift
    f, df = F(z, l, nR)
    weight = 1.
    res = 1. / (2 * np.pi *1j ) * df / f * weight
    return np.imag(res)
    
def Int1_APM_Re(z, shift, tens, F, l, nR):
    z = tens * z + shift
    f, df = F(z, l, nR)
    weight = z
    res = 1. / (2 * np.pi *1j ) * df / f * weight
    return np.real(res)
    
def Int1_APM_Im(z, shift, tens, F, l, nR):
    z = tens * z + shift
    f, df = F(z, l, nR)
    weight = z
    res = 1. / (2 * np.pi *1j ) * df / f * weight
    return np.imag(res)

def New_Raph(z, F, l, nR, tol):
    #Newton-Raphson method to find the precize value of root
    # z - initial guess, tol - desired relative tolerance
    f, df = F(z, l, nR)
    z1 = z - f / df
    tol_Re = abs(1. - abs(np.real(z1) / np.real(z)))
    tol_Im = abs(1. - abs(np.imag(z1) / np.imag(z)))
    if np.real(z) < 1e-4:
        if (tol_Im > tol):
            res = New_Raph(z1, F, l, nR, tol)
            return res
        else:
            return z1
    elif np.imag(z) < 1e-4:
        if (tol_Re > tol):
            res = New_Raph(z1, F, l, nR, tol)
            return res
        else:
            return z1
    else:
        if (tol_Re > tol) or (tol_Im > tol):
            res = New_Raph(z1, F, l, nR, tol)
        else:
            return z1
        
        
    

def findz(x, y, w, h, F, l, nR, root):
    # Recursive function, that finds zeros of F inside the complex rectangular area with parameters (x,y,w,h).
    # F should not have any poles inside the area. 
    # Zeros must not lie on the border of rectangular or its 1/4, 1/16, .. pieces 
    # x,y - coordinates of rectangle center (both real)
    # w,h - half- height and widht of rectangle (both real)
    # Input: k must be equal to 0 !! (counter), 
    # Input: root must be a complex empty array with size > number of roots inside area
    # Output (after all recursions): root - zeros of F (unordered), k - number of zeros inside the area.
    
    #Argument principle method of 0-order
    y0 = y - h; y1 = y + h;
    x0 = x - w; x1 = x + w;
    I01, tol01 = sc.integrate.quad(Int0_APM_Im, y0, y1, args = (x1, 1j, F, l, nR),limit = 5000000)
    I02, tol02 = sc.integrate.quad(Int0_APM_Im, y1, y0, args = (x0, 1j, F, l, nR),limit = 5000000)
    I03, tol03 = sc.integrate.quad(Int0_APM_Re, x0, x1, args = (1j * y0, 1., F, l, nR),limit = 5000000)
    I04, tol04 = sc.integrate.quad(Int0_APM_Re, x1, x0, args = (1j * y1, 1., F, l, nR),limit = 5000000)


   #Number of roots inside the area
    n = np.round(np.real(I03 + I04 - I01 - I02))
    
    if n == 0.:
        return root
    elif n == 1.:
        #Argument principle method of 1-order
        I11 = np.zeros(2); I12 = np.zeros(2); 
        I13 = np.zeros(2); I14 = np.zeros(2); I = np.zeros(2)
        tol11 = np.zeros(2); tol12 = np.zeros(2); 
        tol13 = np.zeros(2); tol14 = np.zeros(2);
        I11[0], tol11[0] = sc.integrate.quad(Int1_APM_Re, y0, y1, args = (x1, 1j, F, l, nR), limit = 5000000)
        I12[0], tol12[0] = sc.integrate.quad(Int1_APM_Re, y1, y0, args = (x0, 1j, F, l, nR), limit = 5000000)
        I13[0], tol13[0] = sc.integrate.quad(Int1_APM_Re, x0, x1, args = (1j * y0, 1., F, l, nR), limit = 5000000)
        I14[0], tol14[0] = sc.integrate.quad(Int1_APM_Re, x1, x0, args = (1j * y1, 1., F, l, nR), limit = 5000000)
        I11[1], tol11[1] = sc.integrate.quad(Int1_APM_Im, y0, y1, args = (x1, 1j, F, l, nR), limit = 5000000)
        I12[1], tol12[1] = sc.integrate.quad(Int1_APM_Im, y1, y0, args = (x0, 1j, F, l, nR), limit = 5000000)
        I13[1], tol13[1] = sc.integrate.quad(Int1_APM_Im, x0, x1, args = (1j * y0, 1., F, l, nR), limit = 5000000)
        I14[1], tol14[1] = sc.integrate.quad(Int1_APM_Im, x1, x0, args = (1j * y1, 1., F, l, nR), limit = 5000000)        
        I = 1j * (I11 + I12) + I13 + I14
        root_guess = I[0] + 1j * I[1]

        NR_tol_Re = 1e-10    #tolerance for Newton-Raphson
#        NR_tol_Im = 1e-18
#        tol_Re = 1
#        tol_Im = 1
        
        root_guess = New_Raph(root_guess, F, l, nR, NR_tol_Re)     #Newton-Raphson method         
        root_guess = np.real(root_guess) - 1j * abs(np.imag(root_guess))
        
#        if abs(np.imag(root_guess)) < 1e-15:
#            root_guess = np.real(root_guess )
        
#        while (tol_Re > NR_tol_Re) or (tol_Im > NR_tol_Im):
#            if abs(np.real(root_guess)) < 1e-4:
#                            NR_tol_Re = 1
#            if abs(np.imag(root_guess)) < 1e-10:
#                NR_tol = abs(np.imag(root_guess))

        root = np.append(root, root_guess)    
        return root
    elif n > 1.:
        root = findz(x - w / 2., y - h / 2., w / 2., h / 2., F, l, nR, root)
        root = findz(x - w / 2., y + h / 2., w / 2., h / 2., F, l, nR, root)
        root = findz(x + w / 2., y - h / 2., w / 2., h / 2., F, l, nR, root)
        root = findz(x + w / 2., y + h / 2., w / 2., h / 2., F, l, nR, root)
        return root
    else: 
        print ("Error in APM!")
        return root


