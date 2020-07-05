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
L=np.array([40.04,80.04,120.07])*0.001 #m - Länge der Zylinder
I0=np.array([1.33,1.35,1.34]) #V - I0 zu den Zylindern
I=np.array([1.33,0.7,0.11])#V - I nach Reflexion
t=np.array([30,59.8,88.9])*10**(-6)/2#s - Laufzeit der Refelexion
td=np.array([16.4,31.3,45.1])*10**(-6)#s - Laufzeit druchschall-Verfahren


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
xplot=np.linspace(0,0.15)

plt.plot(L,t*10**6,"xb",label="Messwerte")
plt.plot(xplot,f(xplot,*params),"--k",label="Ausgleichsgerade")
plt.xlabel("Zylinderlänge $L\;/\;$m")
plt.ylabel("Laufzeit $t\;/\;\mu$s")
plt.legend(loc="best")
plt.savefig("plots/laufzeit_echo.pdf")
plt.close()

c_w=343#m/s - Schallgeschwindigkeit Wasser
d=1/2*c_w*params[1]*10**(-6) #Dicke der Wasserschicht

#Schallbestimmung Durchschall-Verfahren
paramsd,cov = curve_fit(f,L,td*10**6)
errorsd = np.sqrt(np.diag(cov))
unparamsd = unp.uarray(paramsd,errorsd)
unparamsd_V = unparams
xplot=np.linspace(0,0.15)

plt.plot(L,td*10**6,"xb",label="Messwerte")
plt.plot(xplot,f(xplot,*params),"--k",label="Ausgleichsgerade")
plt.xlabel("Zylinderlänge $L\;/\;$m")
plt.ylabel("Laufzeit $t\;/\;\mu$s")
plt.legend(loc="best")
plt.savefig("plots/laufzeit_durch.pdf")



#Auge
ta=np.array([11.5,17,70.6])*10**(-6)/2#s
c1=1532 #Kammerflüssigkeit 
c2=2500 #Linse
c3=1410 #Glaskörper

s1=0.5*c1*ta[0]
s2=0.5*c2*(ta[1]-ta[0])+s1
s3=0.5*c3*(ta[2]-ta[1])+s2

#Ausgabe
print(f"""
||AUF. 1
    alpha: {alpha}

||AUF. 2
    Ausgleichsgerade
    Steigung m={unparams[0]} us/m
    Abschnitt n={unparams[1]} us
    Dicke Wasser: {d}m
    Schallgeschwinigkeit über Gerade = kehrwert v=1/m={1/unparams[0]*10**(6)} m/s

||AUF. 3
    Ausgleichsgerade
    Steigung m={unparamsd[0]} us/m
    Abschnitt n={unparamsd[1]} us
    Schallgeschwinigkeit über Gerade = kehrwert v=1/m={1/unparamsd[0]*10**(6)} m/s

||AUF. 4 
    Iris: {s1}m
    Linse: {s2}m
    Glasörper:{s3}m 
""")
