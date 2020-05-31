import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import *
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import scipy.constants as const


print("Untergrundrate")
N_U = np.array([129,143,144,136,139,126,158])
#Integrationszeit 300s
N_U_error = unp.uarray(N_U,np.sqrt(N_U))
N_U_error = sum(N_U_error)/len(N_U_error)
N_U = np.sum(N_U)/np.size(N_U)

print("Vanadium")

def f(x,m,b):
    return m*x+b
def abweichung(lit,value):
    return (lit-value)/lit*100


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
unparams_1_V = unparams_1


params_2,cov_2 = curve_fit(f,t_V[0:20],np.log(noms(N_V[0:20])))
errors_2 = np.sqrt(np.diag(cov_2))
unparams_2 = unp.uarray(params_2,errors_2)
unparams_2_V = unparams_2


plt.errorbar(t_V,noms(N_V),yerr = stds(N_V),fmt = 'rx', label = "Messdaten")
plt.plot(x_plot_1,np.e**f(x_plot_1,*params_1),label = "Ausgleich 1")
plt.plot(x_plot_2,np.e**f(x_plot_2,*params_2),label = "Ausgleich 2")
plt.xlabel(f"Zeit")
plt.ylabel(f"Impulse pro 30s")
yscale('log')
plt.legend()
plt.savefig("plots/Vanadium.pdf",bbox_inches='tight')
plt.close()

print("Rhodium")
print("Langlebig")

t_R,N_R =np.genfromtxt("data/Rhodium.dat",unpack=True)
#Messintervall 30s
N_U_R = N_U/20 #Durch die Verschiedenen Integrationszeiten wird N_U angepasst (300s / 20 = 15s)
N_R = N_R-N_U_R
N_R=unp.uarray(N_R,np.sqrt(N_R))
x_plot_1 = np.linspace(t_R[0],t_R[np.size(t_R)-1],10000)

params_1,cov_1 = curve_fit(f,t_R[20:np.size(t_R)-1],np.log(noms(N_R[20:np.size(N_R)-1])))
errors_1 = np.sqrt(np.diag(cov_1))
unparams_1 = unp.uarray(params_1,errors_1)

unparams_1_R = unparams_1

plt.errorbar(t_R,noms(N_R),yerr = stds(N_R),fmt = 'rx', label = "Messdaten")
plt.plot(x_plot_1,np.e**f(x_plot_1,*params_1),label = "Ausgleich 1")
plt.xlabel(f"Zeit")
plt.ylabel(f"Impulse pro 15s")
yscale('log')
plt.legend()
plt.savefig("plots/Rhodium_lang.pdf",bbox_inches='tight')
plt.close()

print("Kurzlebig")

N_R = N_R-np.e**f(t_R,*params_1)
x_plot_2 = np.linspace(t_R[0],t_R[20],10000)

params_2,cov_2 = curve_fit(f,t_R[0:17],np.log(noms(N_R[0:17])))
errors_2 = np.sqrt(np.diag(cov_2))
unparams_2 = unp.uarray(params_2,errors_2)

unparams_2_R = unparams_2

plt.errorbar(t_R[0:20],noms(N_R[0:20]),yerr = stds(N_R[0:20]),fmt = 'rx', label = "Korrigierte Messdaten")
plt.plot(x_plot_2,np.e**f(x_plot_2,*params_2),label = "Ausgleich 2")
plt.xlabel(f"Zeit")
plt.ylabel(f"Impulse pro 15s")
yscale('log')
plt.legend()
plt.savefig("plots/Rhodium_kurz.pdf",bbox_inches='tight')
plt.close()

print("Halbertszeiten")

T1_V = unp.log((2))/(-1*unparams_1_V[0])

T2_V = unp.log((2))/(-1*unparams_2_V[0])

T1_R = unp.log((2))/(-1*unparams_1_R[0])

T2_R = unp.log((2))/(-1*unparams_2_R[0])

#Ausgabe
print(f"""
Für die Untergrundrate ergibt sich aus der Mittelung: 
N_U :{N_U_error}
N_U_V : {N_U_error/10}
N_U_R : {N_U_error/20}

Beim Vanadium ergibt sich folgendes:
Ausgleichsgerade 1:
m = {unparams_1_V[0]}
b = {unparams_1_V[1]}
==> lambda = {-1*unparams_1_V[0]}
Halbwertszeit: {T1_V}
Abweichung: {abweichung(224.5,noms(T1_V))}

Ausgleichsgerade 2:
m = {unparams_2_V[0]}
b = {unparams_2_V[1]}
==> lambda = {-1*unparams_2_V[0]}
Halbwertszeit: {T2_V}
Abweichung: {abweichung(224.5,noms(T2_V))}

Für das Rhodium:
Ausgleichsgerade 1:
m = {unparams_1_R[0]}
b = {unparams_1_R[1]}
==> lambda = {-1*unparams_1_R[0]}
Halbwertszeit: {T1_R}
Abweichung: {abweichung(260.4,noms(T1_R))}

Ausgleichsgerade 2:
m = {unparams_2_R[0]}
b = {unparams_2_R[1]}
==> lambda = {-1*unparams_2_R[0]}
Halbwertszeit: {T2_R}
Abweichung: {abweichung(42.3,noms(T2_R))}


""")
