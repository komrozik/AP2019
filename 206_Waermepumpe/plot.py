import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

t,p1,p2,T1,T2,N=np.genfromtxt("data.txt",unpack=True)

p1_0=4.5 
p1 += 1
p2 += 1
t*=60#sec
T1=T1+273.15 #umrechnung in kelvin
T2=T2+273.15 #umrechnung in kelvin
C_Cu=750
C_w=12570
R=8.314


def functionL(x,a,b):
    return -a*x+b


#-----------------------------
#Dampfdruckkurve
params_L, cov_L = curve_fit(functionL,1/T1,np.log(p1/p1_0))
errors_L = np.sqrt(np.diag(cov_L))
unparams_L = unp.uarray(params_L,errors_L)

A =0.9
x_plot = np.linspace(1/T1[0],1/T1[18])
plt.plot(x_plot,functionL(x_plot,*params_L),label='Fit')


plt.plot(1/T1,np.log(p1/p1_0),"rx",label="Dampfdruck")
plt.ylabel(f"ln(p/p_0)")
plt.xlabel(f"$1/T$ in $1/K$")
plt.legend()
plt.savefig("build/plot_L.pdf",bbox_inches='tight')
plt.show()
L_berechnet = params_L[0]*R
print(f"Verdampfungswärme: {L_berechnet}")
plt.close()

#---------------------
#Massendurchsatz