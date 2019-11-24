import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit


def Mittelwert(y):
    return (sum(y))/len(y)


#Allgemeine Werte:
M_k = 512.2                                     #Kugelmasse in g
M_kf = unp.uarray(M_k,0.0004*M_k)
d_k = 50.76                                     #Kugeldurchmesser in mm
d_kf = unp.uarray(d_k,0.00007*d_k)
J_kh = 22.5                                     #Trägheitsmoment der Kugelhalterung in g*cm^2 
N = 390                                         #Windungszahl Helmholtzspule
r_h = 78                                        #Radius der Helmholtzspule in mm
I_max = 1.4                                     #Maximalstrom an der Helmholtzspule in A
d_d = np.array([0.19,0.19,0.19,0.2,0.2])        #Durchmesser des Drahts in mm
d_df = unp.uarray(d_d,0.01)
d_dm = Mittelwert(d_df)                         #Mittelwert des Drahtdurchmessers

T=np.genfromtxt("data_T.txt", unpack = True)    #Werte der Messung ohne Magnetfeld
T_fehler = unp.uarray(T,0.1)
T_m = Mittelwert(T_fehler)                      #Mittelwert der Periodendauer ohne Magnet

#Längenberechnung
L_1 = np.array([61.2,61.0])
L_1f = unp.uarray(L_1,0.1)
L_2 = np.array([5.4,5.3])
L_2f = unp.uarray(L_2,0.1)
L_s = np.array([2.3,2.3])
L_sf = unp.uarray(L_s,0.1)
L_g = np.array([68.6,68.6])
L_gf = unp.uarray(L_g,0.1)
L1 = Mittelwert(L_1f)+Mittelwert(L_2f)
L2 = Mittelwert(L_gf)-Mittelwert(L_sf)
L = Mittelwert([L1,L2])                         #finale Länge

E= unp.uarray(21.00*10**10,0.05*10**10)         #Elastizitätsmodul

G = 16/5 * np.pi * (M_kf*L*(d_kf/2)**2)/((T_m**2)*(d_dm/2)**4)+8*np.pi*(J_kh*L)/((T_m**2)*(d_dm/2)**4)  #Schubmodul

my = (E/(2*G))-1                                #gemäß Formel (2)

Q = E/(3*(1-2*my))                              #gemäß Formel (3)

#Mit Magneten:

data = np.genfromtxt("data.txt", unpack = True)

print (data[0])
print("G:")
print(G)
#print("Mittelwert Periodendauer")
#print(Mittelwert(T_fehler))
#print("Durchmesser Draht")
#print(d_d)
#print(d_df)
#print(d_dm)