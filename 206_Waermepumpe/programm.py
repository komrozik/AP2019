#INFOs siehe commit


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

def functionL(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d

#Güteziffer
def guetez(C_w,C_Cu,Tt,N):
    #C_w = Kapazität Wasser
    #C_Cu = Kapazität Kupfer
    #Tt = dT/dt
    Qt=(C_w+C_Cu)*Tt
    vreal=[0]*4
    i=0
    while i<=3:
        vreal[i]=Qt[i]/N[i]
        i=i+1
    return vreal
def mittelwert(data):
    return sum(data)/len(data)

#Temp Plot
plt.plot(t,T1,"b+",label="T1-Verlauf")
plt.xlabel("Zeit $t\;/\;s$")
plt.ylabel("Temperatur $T\;/\;°C$")

plt.plot(t,T2,"g+",label="T2-Verlauf")
plt.grid(True)

#Temp curvefit
params1, covariance_matrix1=curve_fit(function,t,T1)
params2, covariance_matrix2=curve_fit(function,t,T2)
plt.plot(t,function(t,*params1), "b-",label="Ausgleichsgerade T1")
plt.plot(t,function(t,*params2), "g-",label="Ausgleichsgerade T2")
errors1 = np.sqrt(np.diag(covariance_matrix1))
errors2 = np.sqrt(np.diag(covariance_matrix2))
unparams1 = unp.uarray(params1,errors1)
unparams2 = unp.uarray(params2,errors2)

plt.legend(loc="best")
plt.savefig("build/plot_temp.pdf",bbox_inches='tight')
#plt.show()
plt.close()


#p1 Plot
plt.plot(t,p1,"r+",label="r+")
plt.xlabel("Zeit $t\;/\;min$")
plt.ylabel("Druck $p1\;/\;\mathrm{bar}$")
#plt.show()
plt.close()

#p2 Plot
plt.plot(t,p2,"r+",label="r+")
plt.xlabel("Zeit $t\;/\;min$")
plt.ylabel("Druck $p2\;/\;\mathrm{bar}$")
#plt.show()
plt.close()


#e)
#(p,T) Kurven
plt.plot(T1,p1,"b+",label="$(T_1,p_1)$ Dampfdruck Kurve")
#plt.plot(T2,p2,"r+",label="$(T_2,p_2)$ Dampfdruck Kurve") Monometerfehler p1
#curve_fit
paramsL1, covariance_matrixL1=curve_fit(functionL,T1,p1)
errorsL1 = np.sqrt(np.diag(covariance_matrixL1))
unparamsL1 = unp.uarray(paramsL1,errorsL1)
plt.plot(T1,functionL(T1,*paramsL1), "b-",label="Ausgleichskurve")

#plt.xlim(20+273,55+273)
#plt.ylim(np.log(5),np.log(13))
plt.xlabel("Temperatur $T_1\;/\;K$")
plt.ylabel("$p_1$")
#plt.show()
plt.close()
#Ĺ berechnen mit
#(T/(functionL(T,a,b,c,d)) * ( (R*T/2) + np.sqrt(( R*T/2 )**2 + A*(functionL(T,a,b,c,d)) ) ) (3*a*T**2+2*b*T+c))
a=paramsL1[0]
b=paramsL1[1]
c=paramsL1[2]
d=paramsL1[3]
R=8.314

# Aberechnen
A=1

LT1=(3*a*T1**3+2*b*T1**2+c*T1/(functionL(T1,a,b,c,d))*((R*T1/2)+np.sqrt(R*T1**2/2+A*(functionL(T1,a,b,c,d)))))
print(LT1)


plt.plot(T1,LT1)
plt.xlabel("Temperatur $T_1\;/\;K$")
plt.ylabel("$L\;/\;J/mol$")
#plt.show()
plt.close()


#------------------------------------------------------------------------------
#OUTPUTS
#Ausgleichsgerade Temp plot
print(f"Ausgleichsgerade T1:\nA:{unparams1[0]}\nB:{unparams1[1]}\nC:{unparams1[2]}\n")
print(f"Ausgleichsgerade T2:\nA:{unparams2[0]}\nB:{unparams2[1]}\nC:{unparams2[2]}\n")


T1t=[ableitung_function(t[3],unparams1[0],unparams1[1]),ableitung_function(t[7],unparams1[0],unparams1[1]),ableitung_function(t[13],unparams1[0],unparams1[1]),ableitung_function(t[17],unparams1[0],unparams1[1])]
print(f"Differentialquotient T1:\n{t[3]} & {T1[3]} & {T1t[0]} & \\\\")
print(f"{t[7]} & {T1[7]} & {T1t[1]} & \\\\")
print(f"{t[13]} & {T1[13]} & {T1t[2]} & \\\\")
print(f"{t[17]} & {T1[17]} & {T1t[3]} & \\\\")
print(f"Mittelwert: {mittelwert(T1t)}\n")

T2t=[ableitung_function(t[3],unparams2[0],unparams2[1]),ableitung_function(t[7],unparams2[0],unparams2[1]),ableitung_function(t[13],unparams2[0],unparams2[1]),ableitung_function(t[17],unparams2[0],unparams2[1])]
print(f"Differentialquotient T2:\n{t[3]} & {T2[3]} & {T2t[0]} & \\\\")
print(f"{t[7]} & {T2[7]} & {T2t[1]} & \\\\")
print(f"{t[13]} & {T2[13]} & {T2t[2]} & \\\\")
print(f"{t[17]} & {T2[17]} & {T2t[3]} & \\\\")
print(f"Mittelwert: {mittelwert(T2t)}\n")


#d)
#videal mit T1/(T1-T2)
videal=[0]*18
i=0
while(i<=17):
    videal[i]=T1[i]/(T1[i]-T2[i])
    i=i+1

#vreal
Nv=[N[3],N[7],N[13],N[17]]
vreal=guetez(C_w,C_Cu,T1t,Nv)
print(f"videal: {videal}\n")
print(f"vreal: {vreal}")
#print(f"T1:{T1}")

#Dampfdruck Kurve für L werte ax^3+bx^2+cx+d"
params_L, covariance_matrix_L = curve_fit(functionL,T1,p1)
errors_L = np.sqrt(np.diag(covariance_matrix_L))
unparams_L = unp.uarray(params_L,errors_L)
plt.plot(T1,p1,"kx",label="Dampfdruck")
A =0.9
L_berechnet = T1/(functionL(T1,*params_L)) * ( (R*T1/2) + np.sqrt(( R*T1/2 )**2 + A*(functionL(T1,*params_L)) ) )* (3*params_L[0]*T1**2+2*params_L[1]*T1+params_L[2])

x_plot = np.linspace(T1[0],T1[18])
plt.plot(x_plot,functionL(x_plot,*params_L),label='Fit')
plt.ylabel(f"p")
plt.xlabel(f"T in K")
plt.legend()
plt.savefig("build/plot_b.pdf",bbox_inches='tight')
#plt.show()
plt.close()

#Massendurchsatz:

Qt=(C_w+C_Cu)*T1t
L_array = [L_berechnet[3],L_berechnet[7],L_berechnet[13],L_berechnet[17]]
dm = [0]*4
i=0
while i<=3:
    dm[i]=Qt[i]/N[i]
    i=i+1

print(f"Das ist L: {L_array}")
print(f"Das ist der Massendurchsatz: {dm}")

p2_array = [p2[3],p2[7],p2[13],p2[17]]
p1_array = [p1[3],p1[7],p1[13],p1[17]]
rho0 = 5.51  # in kg/m^3
T0 = 273.15  # in kelvin
p0 = 1  # in bar
kappa = 1.14  # dim. los
d_rho = ((rho0*T0)/p0) * p1_array/T1
Nmech = 1/(kappa - 1) * (p2_array * (p1_array/p2_array)**(1/kappa) - p1_array) * 1/d_rho * dm *1e3 *1e-1 # in W, 1e-1 wegen bar nach Pascal
print(f"Nmech: {Nmech}")