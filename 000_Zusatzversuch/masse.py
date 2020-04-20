import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import * 

def masse_fkt(a,b,n,D):
    return a*n*((1/2*D)^2+(b/(2*np.pi*n))^2)^(1/2)

m_1 = 6.728/5
m_2 = 6.555/5
m_3 = 6.906/5
m_4 = 7.287/5
m_5 = 6.171/5

n_1 = 109.36
n_2 = 109.98
n_3 = 108.39
n_4 = 119.07
n_5 = 99.54

D_1 = unp.uarray(3.68,0.007)
D_2 = unp.uarray(3.57,0)
D_3 = unp.uarray(3.818,0.004)
D_4 = unp.uarray(3.69,0.03)
D_5 = unp.uarray(3.684,0.005)

masse = np.array([m_1,m_2,m_3,m_4,m_5])
durchmesser = np.array([D_1-1/2*0.434,D_2-1/2*0.434,D_3-1/2*0.434,D_4-1/2*0.434,D_5-1/2*0.434])
windungen = np.array([n_1,n_2,n_3,n_4,n_5])


params, cov = curve_fit(masse_fkt,xL0A,FA1)
errorsR1 = np.sqrt(np.diag(covR1))
unparamsR1 = unp.uarray(paramsR1,errorsR1) 

paramsR2, covR2 = curve_fit(rate1,L0A,FA2)            
paramsR3, covR3 = curve_fit(rate1,L0A,FA3)           
paramsR4, covR4 = curve_fit(rate1,L0A,FA4)               
paramsR5, covR5 = curve_fit(rate1,L0A,FA5)         