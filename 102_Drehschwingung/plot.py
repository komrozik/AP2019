import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit


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

T=np.genfromtxt("data_T.txt", unpack = True) #Werte der Messung ohne Magnetfeld
T_fehler = unp.uarray(T,0.1)

def Mittelwert(y):
    return (sum(y))/len(y)

print(T)
print("Mittelwert")
print(Mittelwert(T_fehler))
print("Durchmesser Draht")
print(d_d)
print(d_df)
print(Mittelwert(d_df))