import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Datenimport:

x,F = np.genfromtxt("data.txt",unpack=True)
x=x*0.01 #skalierung in Meter
#x_werte= unp.uarray(x,dx)
F_werte= unp.uarray(F,0.025*F)

#Rechnungen: 

def Federkonstante(x,F):
    return F/x

def Mittelwert(y):
    return (sum(y))/len(y)

def StandardabweichungdesMittelwert(y):
    return Standardabweichung(y)/np.sqrt(len.y)

def fit(x,a,b):
    return a*x+b


#Curve Fit und Plot:
l = np.linspace(0,0.6,1000)

#params,cov = curve_fit(fit,x,F)
params, covariance_matrix = np.polyfit(x, F, deg=1, cov=True)

errors = np.sqrt(np.diag(covariance_matrix))
#plt.plot(l,fit(l,*noms(params)), label='Ausgleichskurve')
#plt.plot(x, y, '.', label="Messwerte")
plt.plot(
    l,
    params[0] * l + params[1],
    'c--',
    label='Lineare Regression',
    linewidth=3,
)


#Ausgaben:
print("Federkonstanten:")
D = Federkonstante(x,F_werte)
#np.savetxt("Federkonstanten.csv",[D],delimiter=" ",fmt='%d')

print (Federkonstante(x,F_werte))
print("Steigung:")
print (params)

print("Mittelwerten der Federkonstanten:")
print(Mittelwert(Federkonstante(x,F_werte)))



#------------------------------------------------------------
#Plot

plt.errorbar(x,F,yerr=stds(F_werte), fmt="rx",label = 'Messwerte',ecolor=['grey'])
plt.xlabel(r'$ \Delta x / \mathrm{m}$')
plt.ylabel(r'$F / \mathrm{N}$')
plt.legend(loc="best")
#plt.figure(figsize=(4.76,2.94))
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')


