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
rho = (R*(np.pi*unp.uarray(69,1)*10**(-6))**2)/(l*10**(-2)) #fake Dicke  :(
r = np.sqrt(0.016*10**(-6)*(l*10**(-2))/(R*np.pi**2)) #dicke die es sein muss für lit wert, wir haben nicht gemessen
print(f"Spezifischer Wiederstand von Silber: {rho*10**(6)} Mikro Ohm") 

U_hall = 1
B_err = 1
I_err = 1 

n = - (1)/(e_0*U_hall)*(B_err*I_err)/(d)
E_F = (h**2)/(2*m_0)*(((3)/(8*np.pi)*n)**2)**(1/3)
tau = 2*(m_0)/(e_0**2)*(1)/(n*rho)
v_drift = -(n*e_0)/(1)
v_total = np.sqrt((2*E_F)/(m_0))
v_delta = 2* v_drift
l = tau*v_total
mu = -(v_delta*m_0)/(v_drift*tau*e_0)