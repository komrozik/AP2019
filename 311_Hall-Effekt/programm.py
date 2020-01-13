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

#Konstantin
e_0=-1.602*10**(-19)

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

##---------Kupfer---------
R_k,l_k,d_k=np.genfromtxt("data9.txt", unpack = True)
rho_k = (R_k*(np.pi*unp.uarray(30,1)*10**(-6))**2)/(l_k*10**(-2))
r_k = np.sqrt(0.018*10**(-6)*(l_k*10**(-2))/(R_k*np.pi**2)) #dicke die es sein muss für lit wert, wir haben nicht gemessen
print(f"Spezifischer Wiederstand von Kupfer: {rho_k*10**(6)} Mikro Ohm")

#Anzahl der Elektronen pro Volumeneinheit | n
#Data 3 Kupfer | I_b_k = Stromstärke des B Felds, U_hall_k = Hallspannung, I_d_k = Durchflussstrom(konstant 10 A)
I_b_k,U_hall_k,I_d_k = np.genfromtxt("data3.txt", unpack = True)
B_k = f(I_b_k,*B_params)
params_Bk,cov_Bk = curve_fit(f,U_hall_k,B_k)
errors_Bk = np.sqrt(np.diag(cov_Bk))
B_k_err = unp.uarray(params_Bk[0],errors_Bk[0])*U_hall_k+unp.uarray(params_Bk[1],errors_Bk[1])

#Data 4 Kupfer |  I_d_k = Durchflussstrom,U_hall_k = Hallspannung,I_b_k = Stromstärke des B Felds(konstant 5 A)
I_b_k,U_hall_k,I_d_k = np.genfromtxt("data4.txt", unpack = True)
params_Idk,cov_Idk = curve_fit(f,U_hall_k,I_b_k)
errors_Idk = np.sqrt(np.diag(cov_Idk))
I_d_k_err = unp.uarray(params_Idk[0],errors_Idk[0])*U_hall_k+unp.uarray(params_Bk[1],errors_Bk[1])


n = -(1)/(e_0*U_hall_k)*(I_d_k_err*B_k_err)/(d_k*10**-6)
print(f"Die Anzahl der Elektronen im Leiter: {n}")

##---------Silber--------- 
R_s,l_s,d_s=np.genfromtxt("data10.txt", unpack = True)
rho_s = (R_s*(np.pi*unp.uarray(69,1)*10**(-6))**2)/(l_s*10**(-2))
r_s = np.sqrt(0.016*10**(-6)*(l_s*10**(-2))/(R_s*np.pi**2)) #dicke die es sein muss für lit wert, wir haben nicht gemessen
print(f"Spezifischer Wiederstand von Silber: {rho_s*10**(6)} Mikro Ohm") 

##---------Zink----------- 
rho_z = 6*10**(-8) #mikro Ohm Litereaturwert
print(f"Literaturwert Spezifischer Wiederstand von Zink: {rho_z*10**(6)} Mikro Ohm")