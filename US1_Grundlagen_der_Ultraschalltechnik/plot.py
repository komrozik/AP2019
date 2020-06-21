import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import *
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import scipy.constants as const

#Alles in Richtiger Reihenfolge. X[0] entspricht dem kleinen ZYlinder, X[1] dem mittleren usw.
L=np.array([40.04,80.04,120.07]) #cm - Länge der Zylinder
I0=np.array([1.33,1.35,1.34]) #V - I0 zu den Zylindern
I=np.array([1.33,0.7,0.11])#V - I nach Reflexion
t=np.array([30,59.8,88.9])*10**(-6)/2#s - Laufzeit der Refelexion


td=np.array([16.4,31.3,45.1])#us - Laufzeit druchschall-Verfahren
#Dämpfung
def absorb(I,I0,L):
    i=0
    alpha=[]
    while i<len(L):
        alpha.append(-1/(2*L[i])*np.log(I[i]/I0[i]))
        i=i+1
    return alpha

alpha=absorb(I,I0,L)


#Schalgeschwindigkeitsbestimmtung
def f(x,m,n):
    return m*x+n

params,cov = curve_fit(f,L,t*10**6)
errors = np.sqrt(np.diag(cov))
unparams = unp.uarray(params,errors)
unparams_V = unparams
xplot=np.linspace(0,120)

plt.plot(L,t*10**6,"xb",label="Messwerte")
plt.plot(xplot,f(xplot,*params),"--k",label="Ausgleichsgerade")
plt.xlabel("Länge L der Zylinder$\;/\;cm$")
plt.ylabel("Laufzeit t$\;/\;\mu s$")
plt.legend(loc="best")
plt.savefig("plots/laufzeit.pdf")

c_w=343#m/s - SChallgeschwindigkeit Wasser
d=1/2*c_w*params[1]*10**(-6) #Dicke der Wasserschicht

geschwEcho=2*L*0.01/((td-params[1])*10**(-6))
#Schallbestimmung Durchschall-Verfahren

geschw=2*L*0.01/(td*10**(-6))



#Auge
ta=np.array([11.5,17,70.6])*10**(-6)/2#s

def distance(t,c):
    return 1/2*c*t

iris=distance(ta[0],343)
linse=distance(ta[1]-ta[0],2500)

#Ausgabe
print(f"""
||AUF. 1
    alpha: {alpha}

||AUF. 2
    Ausgleichsgerade
    Steigung m={params[0]}
    Abschnitt n={params[1]}
    Dicke Wasser: {d}cm
    Schallgeschwindigkeit c = {geschwEcho}

||AUF. 3
    Schallgeschwindigkeit c = {geschw}

||AUF. 4 
    Iris: {iris}m (tiefe)
    Linse: {linse}m (dicke)
""")
