import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import sympy as sym
from scipy import integrate
from scipy.optimize import curve_fit

t,p1,p2,T1,T2,N=np.genfromtxt("data.txt",unpack=True)
t*=60#sec
T1=T1+273.15 #umrechnung in kelvin
T2=T2+273.15 #umrechnung in kelvin

def function(x,A,B,C):
    return A*x**2+B*x+C
def ableitung_function(x,A,B):
    return 2*A*x+B

#Güteziffer
def guetez(C_w,C_Cu,Tt,):
    #C_w = Kapazität Wasser
    #C_Cu = Kapazität Kupfer
    #Tt = dT/dt
    Qt=(C_w+C_Cu)*Tt
    return Qt/N
def mittelwert(data):
    return sum(data)/len(data)

#Temp Plot
plt.plot(t,T1,"b+",label="T1-Verlauf")
plt.xlabel("Zeit $t\;/\;s$")
plt.ylabel("Temperatur $T\;/\;°C$")

plt.plot(t,T2,"g+",label="T2-Verlauf")
plt.grid(True)

#Temp curvefit
params1, covarianc_matrix1=curve_fit(function,t,T1)
params2, covarianc_matrix2=curve_fit(function,t,T2)
plt.plot(t,function(t,*params1), "b-",label="Ausgleichsgerade T1")
plt.plot(t,function(t,*params2), "g-",label="Ausgleichsgerade T2")

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

#OUTPUTS
print(f"Ausgleichsgerade T1:\nA:{params1[0]}\nB:{params1[1]}\nC:{params1[2]}\n")
print(f"Ausgleichsgerade T2:\nA:{params2[0]}\nB:{params2[1]}\nC:{params2[2]}\n")


T1t=[ableitung_function(t[3],params1[0],params1[1]),ableitung_function(t[7],params1[0],params1[1]),ableitung_function(t[13],params1[0],params1[1]),ableitung_function(t[17],params1[0],params1[1])]
print(f"Differentialquotient T1:\n{t[3]} & {T1[3]} & {T1t[0]} & \\\\")
print(f"{t[7]} & {T1[7]} & {T1t[1]} & \\\\")
print(f"{t[13]} & {T1[13]} & {T1t[2]} & \\\\")
print(f"{t[17]} & {T1[17]} & {T1t[3]} & \\\\")
print(f"Mittelwert: {mittelwert(T1t)}\n")

T2t=[ableitung_function(t[3],params2[0],params2[1]),ableitung_function(t[7],params2[0],params2[1]),ableitung_function(t[13],params2[0],params2[1]),ableitung_function(t[17],params2[0],params2[1])]
print(f"Differentialquotient T2:\n{t[3]} & {T2[3]} & {T2t[0]} & \\\\")
print(f"{t[7]} & {T2[7]} & {T2t[1]} & \\\\")
print(f"{t[13]} & {T2[13]} & {T2t[2]} & \\\\")
print(f"{t[17]} & {T2[17]} & {T2t[3]} & \\\\")
print(f"Mittelwert: {mittelwert(T2t)}\n")

videal=[0]*18
i=0
while(i<=17):
    videal[i]=T1[i]/(T1[i]-T1[i+1])
    i=i+1
print(videal)