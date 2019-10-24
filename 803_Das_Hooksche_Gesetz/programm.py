import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x,F,dx = np.genfromtxt('data.txt', unpack =True)
x=x*0.01 #skalierung in Meter
x_werte= unp.uarray(x,dx)
F_werte= unp.uarray(F,0.01*F)
def Federkonstante(x,F):
    return F/x_werte

def Mittelwert(y):
    return (sum(y))/len(y)

##def Standardabweichung(y):
##    unp.std(y)

def StandardabweichungdesMittelwert(y):
    return Standardabweichung(y)/np.sqrt(len.y)

def fit(x,a,b):
    return a*x+b

l = np.linspace(0,0.6,1000)

#params,cov = curve_fit(fit,x,F)
params, covariance_matrix = np.polyfit(x, F, deg=1, cov=True)

errors = np.sqrt(np.diag(covariance_matrix))
#plt.plot(l,fit(l,*noms(params)), label='Ausgleichskurve')
#plt.plot(x, y, '.', label="Messwerte")
plt.plot(
    l,
    params[0] * l + params[1],
    label='Lineare Regression',
    linewidth=3,
)

print("Federkonstanten:")
D = Federkonstante(x_werte,F)
print (Federkonstante(x_werte,F))
print("Steigung:")
print (params)

print("Mittelwerten der Federkonstanten:")
print(Mittelwert(Federkonstante(x_werte,F)))
##print(Standardabweichung(x_werte))
##print(StandardabweichungdesMittelwert(x_werte))



for name, value, error in zip('ab', params, errors):
    print(f'{name} = {value:.3f} ± {error:.3f}')


#------------------------------------------------------------
#Plot

plt.errorbar(x,F,xerr=dx,yerr=stds(F_werte), fmt="x",label = 'Messwerte')
plt.xlabel(r'$\Delta x / \mathrm{m}$')
plt.ylabel(r'$F / \mathrm{N}$')
plt.legend(loc="best")
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')


