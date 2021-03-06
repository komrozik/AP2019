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

A=4-(0.7*2)#V #y=A*e^{-R/(2L)*t} der Startwert aus den Daten für die Theoriekurve


#Aufgabe a)
#----------------------------------------------------------------------------------------------------------------------------
#Werte der Einhüllenden (UC,t) wurden aus dem Bild "Bild_5a.jpg" ausgelesen und in
#dataa.txt gespeichert. Nun soll mit den Werten eine Ausgleichsgerade gemacht werden.
#,eine e-Funktion. 
#Der Dämpfungswiderstand Reff und die Ablinkdauer Tex sollen daraus berechnet werden.

#Idee zur Fehler behebeung. Die funktiona konvergiert gegen 0 für t gegen unendlich.
#in dem Theomodruck ist es aber nicht 0 sonder 1.4V. Wenn man alle messwerte UC-1.4 rechnet
#und auch die KOnstante A -1.4 rechnet passt der Plot perfekt ABER Reff wird noch gößer...
#Problem weiterhin THEORIEKURVE...
#U=R*I - da I=Aexp(R/(2L)*t) musste man eigenlich das ganz noc mit R1 multiplizieren
# wird dann aber noch unpassender.



def funktiona(t,k):
    return (4-(0.7*2))*np.exp(-k*t)
def funktiona_theo(t,k):
    return (4-(0.7*2))*np.exp(-k*t)


UC,t=np.genfromtxt("dataa.txt",unpack=True)
UC=UC*2-(0.7*2) #Volt - Umrechnung da Oszilator auf 2V/Div.
t=t*20*10**(-6) #s - Umrechnung da Oszillator auf 20us/Div.

#Messwerte plotten
tp=t*10**6
plt.plot(tp,UC,"rx",label="Messdaten")

#Messwerte fitten
line=np.linspace(0,tp[len(tp)-1])
params, cov= curve_fit(funktiona,tp,UC)
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
     """                                                                    #Rap mit fehlern
print(werte_params)

#Theoriekurve plotten
plt.plot(tp,funktiona_theo(t,R2/(2*L)),"b--",label="Theoriekurve")
plt.legend()
plt.xlabel("$t\;/\;\mu s$")
plt.ylabel("$U_C\;/\;V$")
plt.savefig("bilder/plota.pdf",bbox_inches='tight')
plt.close()
#----------------------------------------------------------------------------------------------------------------------------










#Aufgabe c) - NAchgetragen werden muss ein Theoriekurve. Die aber immer zu einer Geraden wird
#----------------------------------------------------------------------
line_w=np.linspace(25,45)
line_W=line_w*1000
U_0=1

def U_funktion(w):
    return 1.5/(np.sqrt((1-L*C*w**2)**2+(w*R2*C)**2))

line_w= np.linspace(15,55)
line_W=line_w*1000

w,U=np.genfromtxt("datac.txt",unpack=True)
plt.plot(line_w,U_funktion(line_W*6.6),label="Theoriekurve")
plt.plot(w,U,"r+",label="Messwerte")
plt.ylabel("$U\;/\;V$")
plt.xlabel("$w\,/\,kHz$")
plt.legend()
plt.savefig("bilder/plotc.pdf",bbox_inches='tight')
plt.close()

#Impedanzkurve
line_w= np.linspace(5,60)
line_W=line_w*1000
plt.plot(line_w,np.sqrt(R2**2+(line_W*L-1/(line_W*C))**2),"r-",label="Theoriewerte")
plt.ylabel("$Z\;/\;$Ohm")
plt.xlabel("$w\,/\,kHz$")
plt.legend()
plt.savefig("bilder/plotZ.pdf",bbox_inches='tight')
plt.close()
#----------------------------------------------------------------------
















#Aufgabe d)
#----------------------------------------------------------------------
w,a,b=np.genfromtxt("datad.txt",unpack=True)
phi=a/b*np.pi
print("w    &   a   &   b   &   phi\n")
for x in range(0,len(w)-1):
    print(f"{w[x]} & {a[x]} & {b[x]} & {phi[x].round(2)} \\\\")

plt.plot(w,phi,"rx",label="Phasenverschiebung")
plt.xlabel("$w\;/\;kHz$")
plt.ylabel("$\Delta \phi \;/\;$rad")
plt.legend(loc="best")
plt.savefig("bilder/plotph.pdf",bbox_inches='tight')
plt.show()
#----------------------------------------------------------------------
