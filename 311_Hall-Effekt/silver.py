import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

#Funktion für den linearen Fit.
def f(x,a,b):
    return a*x+b

#Konstanten
e_0   =   -1.602176634     * 10**(-19)#C
m_0 =   9.1093837015    * 10**(-31) #kg
h   =   6.62607015      * 10**(-34)#J*s

##----------Magnetfeld----------
#Funktion um das B-Feld zu bekanntem Strom zu berechnen.
I,B=np.genfromtxt("data2.txt", unpack = True)
B_params,B_cov = curve_fit(f,I,B)
errors = np.sqrt(np.diag(B_cov))
B_params_err = unp.uarray(B_params,errors)
B_err = B_params_err[0]*I+B_params_err[1]

#B-Feld für verschiedene Stromstärken ausgeben:
#print(f"B Feld: {B_err}")
magnetfeld= f"""
Magnetfeld\n
-----------------------------\n
Strom & Magnetfeld\n
{I[0]} & {B_err[0]}\\\\
{I[1]} & {B_err[1]}\\\\
{I[2]} & {B_err[2]}\\\\
{I[3]} & {B_err[3]}\\\\
{I[4]} & {B_err[4]}\\\\
{I[5]} & {B_err[5]}\\\\
{I[6]} & {B_err[6]}\\\\
{I[7]} & {B_err[7]}\\\\
{I[8]} & {B_err[8]}\\\\
{I[9]} & {B_err[9]}\\\\
{I[10]} & {B_err[10]}\\\\
-----------------------------\n
"""
print(magnetfeld)
## --------Magnetfeld fertig--------

R,l,d=np.genfromtxt("data10.txt", unpack = True)
rho = (R*np.pi*(1/2*unp.uarray(0.218,0.01)*10**(-3))**2)/(l*10**(-2)) #fake Dicke  :(
r = np.sqrt(0.016*10**(-6)*(l*10**(-2))/(R*np.pi**2)) #dicke die es sein muss für lit wert, wir haben nicht gemessen
print(f"Spezifischer Wiederstand von Silber: {rho*10**(6)} Mikro Ohm") 

##------Hall Spannung und Strom berechnen -------------

#6) Data 7 Silber| I_b = Stromstärke des B Felds, U_hall_b = Hallspannung, I_d = Durchflussstrom(konstant 10 A)
I_b,U_hall_b,I_d = np.genfromtxt("data7.txt", unpack = True)

B = f(I_b,*B_params)
params,cov = curve_fit(f,B,U_hall_b)
errors = np.sqrt(np.diag(cov))
U_hall = unp.uarray(params[0],errors[0])*B+unp.uarray(params[1],errors[1])


#7) Data 8 Silber|  I_d = Durchflussstrom,U_hall_d = Hallspannung,I_b = Stromstärke des B Felds(konstant 5 A)
I_b,U_hall_d,I_d = np.genfromtxt("data8.txt", unpack = True)

params,cov = curve_fit(f,I_d,U_hall_d)
errors = np.sqrt(np.diag(cov))

I_err = (U_hall-unp.uarray(params[1],errors[1]))/(unp.uarray(params[0],errors[0]))

##----------------------------------------


n =  (1)/(e_0*U_hall)*(B_err*I_err)/(d)
#print(f"n :{n}")
E_F = (h**2)/(2*m_0)*(((3)/(8*np.pi)*n)**2)**(1/3)
#print(f"E_F :{E_F}")
tau = 2*(m_0)/(e_0**2)*(1)/(n*rho)
#print(f"Tau :{tau}")
v_drift = -(n*e_0)/(1)
#print(f"Drift v :{v_drift}")
v_total = ((2*E_F)/(m_0))**(1/2)
#print(f"Total v :{v_total}")
v_delta = 2* v_drift
#print(f"Delta v :{v_delta}")
l = tau*v_total
#print(f"l:{l}")
mu = -(v_delta*m_0)/(v_drift*tau*e_0)
#print(f"My: {mu}")
tabelle1= f"""
Tabelle 1\n
-----------------------------\n
Hall Spannung & Magnetfeld & Ladungsträger pro Volumen & mittlere Flugzeit & Driftgeschwindigkeit\n
{U_hall[0]}  & {B_err[0]}  & {n[0]}  & {tau[0]}  & {v_drift[0]}  \\\\
{U_hall[1]}  & {B_err[1]}  & {n[1]}  & {tau[1]}  & {v_drift[1]}  \\\\
{U_hall[2]}  & {B_err[2]}  & {n[2]}  & {tau[2]}  & {v_drift[2]}  \\\\
{U_hall[3]}  & {B_err[3]}  & {n[3]}  & {tau[3]}  & {v_drift[3]}  \\\\
{U_hall[4]}  & {B_err[4]}  & {n[4]}  & {tau[4]}  & {v_drift[4]}  \\\\
{U_hall[5]}  & {B_err[5]}  & {n[5]}  & {tau[5]}  & {v_drift[5]}  \\\\
{U_hall[6]}  & {B_err[6]}  & {n[6]}  & {tau[6]}  & {v_drift[6]}  \\\\
{U_hall[7]}  & {B_err[7]}  & {n[7]}  & {tau[7]}  & {v_drift[7]}  \\\\
{U_hall[8]}  & {B_err[8]}  & {n[8]}  & {tau[8]}  & {v_drift[8]}  \\\\
{U_hall[9]}  & {B_err[9]}  & {n[9]}  & {tau[9]}  & {v_drift[9]}  \\\\
{U_hall[10]} & {B_err[10]} & {n[10]} & {tau[10]} & {v_drift[10]} \\\\
-----------------------------\n
"""
print(tabelle1)

tabelle2= f"""
Tabelle 2\n
-----------------------------\n
Hall Spannung  & Totalgeschwindigkeit & mittlere freie Weglänge & Beweglichkeit  \n
{U_hall[0]}  & {v_total[0]}  & {l[0]}  & {mu[0]}  \\\\
{U_hall[1]}  & {v_total[1]}  & {l[1]}  & {mu[1]}  \\\\
{U_hall[2]}  & {v_total[2]}  & {l[2]}  & {mu[2]}  \\\\
{U_hall[3]}  & {v_total[3]}  & {l[3]}  & {mu[3]}  \\\\
{U_hall[4]}  & {v_total[4]}  & {l[4]}  & {mu[4]}  \\\\
{U_hall[5]}  & {v_total[5]}  & {l[5]}  & {mu[5]}  \\\\
{U_hall[6]}  & {v_total[6]}  & {l[6]}  & {mu[6]}  \\\\
{U_hall[7]}  & {v_total[7]}  & {l[7]}  & {mu[7]}  \\\\
{U_hall[8]}  & {v_total[8]}  & {l[8]}  & {mu[8]}  \\\\
{U_hall[9]}  & {v_total[9]}  & {l[9]}  & {mu[9]}  \\\\
{U_hall[10]} & {v_total[10]} & {l[10]} & {mu[10]} \\\\
-----------------------------\n
"""
print(tabelle2)