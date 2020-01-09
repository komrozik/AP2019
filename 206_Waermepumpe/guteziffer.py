import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

t,p1,p2,T1,T2,N=np.genfromtxt("data.txt",unpack=True)
p1_0=4.5 
p1 += 1
p2 += 1
t*=60#sec
T1=T1+273.15 #umrechnung in kelvin
T2=T2+273.15 #umrechnung in kelvin
C_Cu=750
C_w=12570
R=8.314

def function(x,A,B,C):
    return A*x**2+B*x+C
def ableitung_function(x,A,B):
    return 2*A*x+B

videal=T1/(T1-T2)
print(f"videal:\n{videal}\n")
Nm=sum(N)/19

params1, covariance_matrix1=curve_fit(function,t,T1)
params2, covariance_matrix2=curve_fit(function,t,T2)
errors1 = np.sqrt(np.diag(covariance_matrix1))
errors2 = np.sqrt(np.diag(covariance_matrix2))
unparams1 = unp.uarray(params1,errors1)
unparams2 = unp.uarray(params2,errors2)

T1t=ableitung_function(t,unparams1[0],unparams1[1])
print(f"T1t:\n{T1t}\n")

vreal=(C_w+C_Cu)*T1t/Nm
print(f"vreal:\n{vreal}\n")

guetetabelle= f"""
Gütetabelle\n
-----------------------------\n
Zeit & videal & vreal\n
{t[1]} & {videal[1]} & {vreal[1]} \\\\\n
{t[7]} & {videal[4]} & {vreal[4]} \\\\\n
{t[13]} & {videal[13]} & {vreal[13]} \\\\\n
{t[18]} & {videal[18]} & {vreal[18]} \\\\\n
-----------------------------\n
"""

print(guetetabelle)

def functionL(x,a,b):
    return -a*x+b


#-----------------------------
#Dampfdruckkurve

params_L, cov_L = curve_fit(functionL,1/T1,np.log(p1/p1_0))
errors_L = np.sqrt(np.diag(cov_L))
unparams_L = unp.uarray(params_L,errors_L)

A =0.9
x_plot = np.linspace(1/T1[0],1/T1[18])
plt.plot(x_plot,functionL(x_plot,*params_L),label='Fit')


plt.plot(1/T1,np.log(p1/p1_0),"rx",label="Dampfdruck")
plt.ylabel(f"ln(p/p_0)")
plt.xlabel(f"$1/T$ in $1/K$")
plt.legend()
plt.savefig("build/plot_L.pdf",bbox_inches='tight')
#plt.show()
L_berechnet = params_L[0]*R
print(f"Verdampfungswärme: {L_berechnet}")
plt.close()

#---------------------
#Massendurchsatz
dmt = (vreal*Nm)/L_berechnet

massendurchsatz= f"""
Massendurchsatz\n
-----------------------------\n
Zeit & massendurchsatz\n
{t[1]} & {dmt[1]}\\\\\n
{t[7]} & {dmt[4]}\\\\\n
{t[13]} & {dmt[13]} \\\\\n
{t[18]} & {dmt[18]} \\\\\n
-----------------------------\n
"""
print(massendurchsatz)

# Kompressorleistung

rho0 = 5.51
T0 = 273.15
p0 = 1 
kappa = 1.14

rho = (rho0*T0)/(p0)*p1/T1

N_mech = 1/(kappa-1)*(p1*(p2/p1)**(1/kappa)-p2)*1/rho*dmt
print(f"Mechanisch{N_mech}")