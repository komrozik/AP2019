import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit
from uncertainties import ufloat


#Werte allgemein:
R1=30.3#Ohm
R2=271.6#Ohm
L=3.5*10**(-3)#H
C=5.00*10**(-9)#F

R1_e=ufloat(30.3,0.01)#Ohm          #mit Fehler (da Plot nicht mit Fehlern geht)
R2_e=ufloat(271.6,0.2)#Ohm
L_e=ufloat(3.5,0.01)*10**(-3)#H
C_e=ufloat(5.00,0.2)*10**(-9)#F

A=4 #y=A*e^{-R/(2L)*t} der Startwert aus den Daten für die Theoriekurve


#Aufgabe a)
#----------------------------------------------------------------------------------------------------------------------------
#Werte der Einhüllenden (UC,t) wurden aus dem Bild "Bild_5a.jpg" ausgelesen und in
#dataa.txt gespeichert. Nun soll mit den Werten eine Ausgleichsgerade gemacht werden.
#,eine e-Funktion. 
#Der Dämpfungswiderstand Reff und die Ablinkdauer Tex sollen daraus berechnet werden.

def funktiona(t,k):
    return 4*np.exp(-k*t)


UC,t=np.genfromtxt("dataa.txt",unpack=True)
UC=UC*2 #Volt - Umrechnung da Oszilator auf 2V/Div.
t=t*20*10**(-6) #s - Umrechnung da Oszillator auf 20us/Div.

#Messwerte plotten
plt.plot(t,UC,"rx",label="Messdaten")

#Messwerte fitten
line=np.linspace(0,t[len(t)-1])
params, cov= curve_fit(funktiona,t,UC)
errors = np.sqrt(np.diag(cov))
unparams = unp.uarray(params,errors)
plt.plot(line,funktiona(line,*params),"g",label='Ausgleichs e-Funktion')

werte_params=f"""
    Berechnete Werte aus Messwert-Fit für Bausteine\\
    k=R/2L: {unparams[0]}\
    Tex=1/k {1/unparams[0]}\
    Reff: {unparams[0]*2*L}\
    L: {L_e}\\
    Theorie\
    k=R/2L: {R1_e/(2*L_e)}\n
    Rap: {np.sqrt(4*L/C)}
     """
print(werte_params)

#Theoriekurve plotten
plt.plot(t,funktiona(t,R1/(2*L)),"b--",label="Theoriekurve")
plt.legend()
plt.xlabel("$t\;/\;\mu s$")
plt.ylabel("$U_C\;/\;V$")
plt.savefig("bilder/plota.pdf",bbox_inches='tight')
plt.close()
#----------------------------------------------------------------------------------------------------------------------------










#Aufgabe c)
#----------------------------------------------------------------------
w,U=np.genfromtxt("datac.txt",unpack=True)

plt.plot(w,U,"r+",label="Messwerte")
plt.ylabel("$U\;/\;V$")
plt.xlabel("$w\,/\,kHz$")
plt.legend()
plt.savefig("bilder/plotc.pdf",bbox_inches='tight')
plt.show()