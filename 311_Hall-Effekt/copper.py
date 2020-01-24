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

#######################################################
#Korrektur
#######################################################

##---------Kupfer---------
R,l,d=np.genfromtxt("data9.txt", unpack = True)
rho = (R*np.pi*(1/2*unp.uarray(0.218,0.01)*10**(-3))**2)/(l*10**(-2))
r = np.sqrt(0.018*10**(-6)*(l*10**(-2))/(R*np.pi**2)) #dicke die es sein muss für lit wert, wir haben nicht gemessen
print(f"Spezifischer Wiederstand von Kupfer: {rho*10**(6)} Mikro Ohm meter")
rho = 0.018*10**(-6)
print(f"Spezifischer Literaturwiederstand : {rho*10**(6)} Mikro Ohm meter")

#______________Durchflussstrom konstant__________________
#2) Data 3 Kupfer | I_b = Stromstärke des B Felds, U_hall = Hallspannung, I_d = Durchflussstrom(konstant 10 A)
I_b,U_hall,I_d = np.genfromtxt("data3.txt", unpack = True)

#i=0
# while(i<=len(B)-1):
#     print(f"{B[i]} & {I[i]} \\\\ \n")
#     i=i+1

B = f(I_b,*B_params)
params,cov = curve_fit(f,B,U_hall)
errors = np.sqrt(np.diag(cov))
print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n")
print(f"HIER BEGINNEN DIE PARAMETER aus dem B-Feld Plot:\n")
print(f"""Die Parameter für den \"U_hall in Abh. von B\" Plot sind:\n
        a: {params[0]*10**6} +- {errors[0]*10**6}
        b: {params[1]*10**3} +- {errors[1]*10**3}]""")
x_plot = np.linspace(f(0,*B_params),f(5,*B_params))
plt.plot(x_plot,f(x_plot,*params),label = "linearer Fit")
plt.plot(B,U_hall,"rx",label="Messwerte")
plt.xlabel(f"Magnetfeld $B \;\;mT$")
plt.ylabel(f"Hall Spannung $U \;\; mV$")
plt.legend()
plt.savefig('build/plot3.pdf',bbox_inches='tight')
#plt.show()
plt.close()

n = (I_d[0])/(e_0*d*(10**(-6))*unp.uarray(params[0],errors[0]))
print(f"Der Parameter n in 1/m^3 ist: {n}")

E_F = (h**2)/(2*m_0) * (3/(8*np.pi)*n)**(2/3)
#E_F = E_F *(6.242*10**18) #in eV
print(f"Wert für E_F: {E_F}")

v_total = ((2*E_F)/(m_0))**(1/2)
print(f"Total v :{v_total}")

tau = 2*(m_0)/(e_0**2)*(1)/(n*rho)
print(f"Tau :{tau}")

l = tau*v_total
print(f"l:{l}")

v_drift = -1/(n*e_0*10**(-6))
print(f"Drift v :{v_drift}")


mu = (2*m_0)/(e_0**2*n*tau)*10**(6)
print(f"My: {mu}")

sigma = 1/2*(e_0**2)/(m_0)*n*tau * 10**(-6)
print(f"Sigma: {sigma}")





#_______________________B-Feld konstant___________________________________
#3) Data 4 Kupfer |  I_d = Durchflussstrom,U_hall = Hallspannung,I_b = Stromstärke des B Felds(konstant 5 A)
I_b,U_hall,I_d = np.genfromtxt("data4.txt", unpack = True)

B = f(I_b,*B_params)
params,cov = curve_fit(f,I_d,U_hall)
errors = np.sqrt(np.diag(cov))
print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n")
print(f"HIER BEGINNEN DIE PARAMETER aus dem I-Feld Plot:\n")
print(f"""Die Parameter für den \"U_hall in Abh. von B\" Plot sind:\n
        a: {params[0]*10**6} +- {errors[0]*10**6}
        b: {params[1]*10**3} +- {errors[1]*10**3}]""")
x_plot = np.linspace(0,10)
plt.plot(x_plot,f(x_plot,*params),label = "linearer Fit")
plt.plot(I_d,U_hall,"rx",label="Messwerte")
plt.xlabel(f"Durchflussstrom $I_d \;\; A$")
plt.ylabel(f"Hall Spannung $U \;\; mV$")
plt.legend()
plt.savefig('build/plot4.pdf',bbox_inches='tight')
#plt.show()
plt.close()

n = (B[0])/(e_0*d*(10**(-6))*unp.uarray(params[0],errors[0]))
print(f"Der Parameter n in 1/m^3 ist: {n}")

E_F = (h**2)/(2*m_0) * (3/(8*np.pi)*n)**(2/3)
#E_F = E_F *(6.242*10**18) #in eV
print(f"Wert für E_F: {E_F}")

v_total = ((2*E_F)/(m_0))**(1/2)
print(f"Total v :{v_total}")

tau = 2*(m_0)/(e_0**2)*(1)/(n*rho)
print(f"Tau :{tau}")

l = tau*v_total
print(f"l:{l}")

v_drift = -1/(n*e_0*10**(-6))
print(f"Drift v :{v_drift}")


mu = (2*m_0)/(e_0**2*n*tau)*10**(6)
print(f"My: {mu}")


# tabelle1= f"""
# Tabelle 1\n
# -----------------------------\n
# Hall Spannung & Magnetfeld & Ladungsträger pro Volumen & mittlere Flugzeit & Driftgeschwindigkeit\n
# {U_hall[0]}  & {B_err[0]}  & {n[0]}  & {tau[0]}  & {v_drift[0]}  \\\\
# {U_hall[1]}  & {B_err[1]}  & {n[1]}  & {tau[1]}  & {v_drift[1]}  \\\\
# {U_hall[2]}  & {B_err[2]}  & {n[2]}  & {tau[2]}  & {v_drift[2]}  \\\\
# {U_hall[3]}  & {B_err[3]}  & {n[3]}  & {tau[3]}  & {v_drift[3]}  \\\\
# {U_hall[4]}  & {B_err[4]}  & {n[4]}  & {tau[4]}  & {v_drift[4]}  \\\\
# {U_hall[5]}  & {B_err[5]}  & {n[5]}  & {tau[5]}  & {v_drift[5]}  \\\\
# {U_hall[6]}  & {B_err[6]}  & {n[6]}  & {tau[6]}  & {v_drift[6]}  \\\\
# {U_hall[7]}  & {B_err[7]}  & {n[7]}  & {tau[7]}  & {v_drift[7]}  \\\\
# {U_hall[8]}  & {B_err[8]}  & {n[8]}  & {tau[8]}  & {v_drift[8]}  \\\\
# {U_hall[9]}  & {B_err[9]}  & {n[9]}  & {tau[9]}  & {v_drift[9]}  \\\\
# {U_hall[10]} & {B_err[10]} & {n[10]} & {tau[10]} & {v_drift[10]} \\\\
# -----------------------------\n
# """
# print(tabelle1)

# tabelle2= f"""
# Tabelle 2\n
# -----------------------------\n
# Hall Spannung  & Totalgeschwindigkeit & mittlere freie Weglänge & Beweglichkeit  \n
# {U_hall[0]}  & {v_total[0]}  & {l[0]}  & {mu[0]}  \\\\
# {U_hall[1]}  & {v_total[1]}  & {l[1]}  & {mu[1]}  \\\\
# {U_hall[2]}  & {v_total[2]}  & {l[2]}  & {mu[2]}  \\\\
# {U_hall[3]}  & {v_total[3]}  & {l[3]}  & {mu[3]}  \\\\
# {U_hall[4]}  & {v_total[4]}  & {l[4]}  & {mu[4]}  \\\\
# {U_hall[5]}  & {v_total[5]}  & {l[5]}  & {mu[5]}  \\\\
# {U_hall[6]}  & {v_total[6]}  & {l[6]}  & {mu[6]}  \\\\
# {U_hall[7]}  & {v_total[7]}  & {l[7]}  & {mu[7]}  \\\\
# {U_hall[8]}  & {v_total[8]}  & {l[8]}  & {mu[8]}  \\\\
# {U_hall[9]}  & {v_total[9]}  & {l[9]}  & {mu[9]}  \\\\
# {U_hall[10]} & {v_total[10]} & {l[10]} & {mu[10]} \\\\
# -----------------------------\n
# """
# print(tabelle2)