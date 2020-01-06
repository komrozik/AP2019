import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit




#5) LED | d=distance, V=Spannung
d,V=np.genfromtxt("data3.txt", unpack = True)
d=d
i=0
# while(i<=len(d)-1):
#     print(f"{d[i]} & {V[i]} \\\\ \n")
#     i=i+1


x_plot = np.linspace(18,55)
def function(x,a,b):
    return (a*1/(x**2))+b
params, covariance_matrix = curve_fit(function,d,V)
print(f"Parameter: {params}")
#print(f"Spannung: {V}")
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot-18.5,function(x_plot,*params),label='Fit')
d=d-18.5
plt.plot(d,V,"rx",label="Messwerte")
plt.ylabel(f"Spannung $U \;/\;V$")
plt.xlabel(f"Entfernung$\;/\;cm$")
plt.legend()
plt.savefig('build/plotLED.pdf',bbox_inches='tight')
#plt.show()
plt.close()

#2) Werte ohne Rauschen | p = phase, V = Spannung
p,V=np.genfromtxt("data1.txt", unpack = True)
# while(i<=len(p)-1):
#     print(f"{p[i]} & {V[i]} \\\\ \n")
#     i=i+1

x_plot = np.linspace(0,360)
def function(x,a,c):
    return (2/np.pi)*a*np.cos((x/180)*np.pi+c)
params, covariance_matrix = curve_fit(function,p,V)
print(f"Parameter: {params}")
#print(f"Spannung: {V}")
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot,function(x_plot,*params),label='Fit')
plt.plot(p,V,"rx",label="Messwerte")
plt.ylabel(f"Spannung $U \;/\;V$")
plt.xlabel(f"Phase$\;/\;°$")
plt.legend()
plt.savefig('build/plot1.pdf',bbox_inches='tight')
#plt.show()
plt.close()

#3) Werte ohne Rauschen | p = phase, V = Spannung
p,V=np.genfromtxt("data2.txt", unpack = True)
# while(i<=len(p)-1):
#     print(f"{p[i]} & {V[i]} \\\\")
#     i=i+1

x_plot = np.linspace(0,360)
def function(x,a,c):
    return (2/np.pi)*a*np.cos((x/180)*np.pi+c)
params, covariance_matrix = curve_fit(function,p,V)
print(f"Parameter: {params}")
#print(f"Spannung: {V}")
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot,function(x_plot,*params),label='Fit')
plt.plot(p,V,"rx",label="Messwerte")
plt.ylabel(f"Spannung $U \;/\;V$")
plt.xlabel(f"Phase$\;/\;°$")
plt.legend()
plt.savefig('build/plot2.pdf',bbox_inches='tight')
#plt.show()
plt.close()