import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import *
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import scipy.constants as const

t_R,N_R =np.genfromtxt("data/Rhodium.dat",unpack=True)
N_R=unp.uarray(N_R,np.sqrt(N_R))

print("Untergrundrate")
N_U = np.array([129,143,144,136,139,126,158])
#Integrationszeit 300s
N_U = np.sum(N_U)/np.size(N_U)

print("Vanadium")

def f(x,m,b):
    return m*x+b

def fN(x,a,b):
    return 

t_V,N_V =np.genfromtxt("data/Vanadium.dat",unpack=True)
#Messintervall 30s
N_U_V = N_U/10 #Durch die Verschiedenen Integrationszeiten wird N_U angepasst (300s / 10 = 30s)
N_V = N_V-N_U_V
N_V=unp.uarray(N_V,np.sqrt(N_V))
x_plot_1 = np.linspace(t_V[0],t_V[np.size(t_V)-1],10000)
x_plot_2 = np.linspace(t_V[0],t_V[20],10000)

params_1,cov_1 = curve_fit(f,t_V,np.log(noms(N_V)))
errors_1 = np.sqrt(np.diag(cov_1))
unparams_1 = unp.uarray(params_1,errors_1)


params_2,cov_2 = curve_fit(f,t_V[0:20],np.log(noms(N_V[0:20])))
errors_2 = np.sqrt(np.diag(cov_2))
unparams_2 = unp.uarray(params_2,errors_2)


plt.errorbar(t_V,noms(N_V),yerr = stds(N_V),fmt = 'rx', label = "Messdaten")
plt.plot(x_plot_1,np.e**f(x_plot_1,*params_1),label = "Ausgleich 1")
plt.plot(x_plot_2,np.e**f(x_plot_2,*params_2),label = "Ausgleich 2")
plt.xlabel(f"Zeit")
plt.ylabel(f"Impulse pro 30s")
yscale('log')
plt.legend()
plt.savefig("plots/Vanadium.pdf",bbox_inches='tight')
plt.close()




#Ausgabe
print(f"""
Hello World!
Untergrundrate: {N_U}
""")
