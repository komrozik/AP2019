import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit




#5) LED | d=distanc, V=Spannung
d,V=np.genfromtxt("data3.txt", unpack = True)
d=d-18.6
kehr_V=1/V
i=0
while(i<len(d)-1):
    print(f"{d[i]} & {d[i+1]} \\\\ \n")
    i=i+1


x_plot = np.linspace(0,36)
def function(x,a,b):
    return (a*1/(x**2))+b
params, covariance_matrix = curve_fit(function,d,V)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot,function(x_plot,*params),label='Fit')

plt.plot(d,V,"rx",label="Messwerte")
plt.ylabel(f"Spannung $U \;/\;V$")
plt.xlabel(f"Entfernung$\;/\;cm$")
plt.legend()
plt.savefig('build/plotLED.pdf',bbox_inches='tight')
plt.close()

#plt.show()