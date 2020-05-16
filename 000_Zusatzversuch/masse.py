import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import * 

def fkt(x,a,b):
    return a*x+b

def masse_D_fkt(D,a,b):
    return a*n_1*((1/2*D)**2+(b/(2*np.pi*n_1))**2)**(1/2)

def masse_n_fkt(n,a,b):
    return a*n*((1/2*noms(D_1))**2+(b/(2*np.pi*n))**2)**(1/2)

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

masse = array([m_1,m_2,m_3,m_4,m_5])
durchmesser = array([D_1-1/2*0.434,D_2-1/2*0.434,D_3-1/2*0.434,D_4-1/2*0.434,D_5-1/2*0.434])
windungen = array([n_1,n_2,n_3,n_4,n_5])


# Ausgleich für den Durchmesser
params_D, cov_D = curve_fit(masse_D_fkt,noms(durchmesser[0:3]),masse[0:3])
errors_D = np.sqrt(np.diag(cov_D))
unparams_D = unp.uarray(params_D,errors_D)
params_D_lin, cov_D_lin = curve_fit(fkt,noms(durchmesser[0:3]),masse[0:3])
errors_D_lin = np.sqrt(np.diag(cov_D_lin))
unparams_D_lin = unp.uarray(params_D_lin,errors_D_lin)
linsp= np.linspace(3.3,3.6,100)
plt.plot(noms(durchmesser[0:3]),masse[0:3],"o",label = "Messwerte")
plt.plot(linsp,masse_D_fkt(linsp,*params_D),"--",label = "Ausgleich")
plt.plot(linsp,fkt(linsp,*params_D_lin),"--",label = "Ausgleich linear")
plt.xlabel("Durchmesser")
plt.ylabel("Masse")
plt.legend()
plt.savefig("plots/Masse_D.pdf")
plt.close()

#Ausgleich für Windungszahl
params_n, cov_n = curve_fit(masse_n_fkt,noms(windungen[2:6]),masse[2:6])
errors_n = np.sqrt(np.diag(cov_n))
unparams_n = unp.uarray(params_n,errors_n)
params_n_lin, cov_n_lin = curve_fit(fkt,noms(windungen[2:6]),masse[2:6])
errors_n_lin = np.sqrt(np.diag(cov_n_lin))
unparams_n_lin = unp.uarray(params_n_lin,errors_n_lin)
linsp= np.linspace(99,120,100)
plt.plot(noms(windungen[2:6]),masse[2:6],"o",label = "Messwerte")
plt.plot(linsp,masse_n_fkt(linsp,*params_n),"--",label = "Ausgleich")
plt.plot(linsp,fkt(linsp,*params_n_lin),"--",label = "Ausgleich linear")
plt.xlabel("Windungen")
plt.ylabel("Masse")
plt.legend()
plt.savefig("plots/Masse_n.pdf")
plt.close()

unparams = (unparams_D + unparams_n)/2

print(f"""
Die Parameter für die Ausgleichsrechnung der Massenkurve sind:
Durchmesser:
"Normal": A = {unparams_D[0]} und B = {unparams_D[1]}
Linear: Steigung = {unparams_D_lin[0]} und y-Achsenabschnitt = {unparams_D_lin[1]}

Windungszahl:
"Normal": A = {unparams_n[0]} und B = {unparams_n[1]}
Linear: Steigung = {unparams_n_lin[0]} und y-Achsenabschnitt = {unparams_n_lin[1]}

Die Parameter ergeben sich somit zu:
A = {unparams[0]} und B = {unparams[1]}
""")