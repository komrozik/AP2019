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
p1 += 1
p2 += 1
t*=60#sec
T1=T1+273.15 #umrechnung in kelvin
T2=T2+273.15 #umrechnung in kelvin
C_Cu=750
C_w=12570

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