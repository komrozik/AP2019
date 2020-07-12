import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import *
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import scipy.constants as const

U1,d1=np.genfromtxt("data/data1.dat",unpack=True)
U2,d2=np.genfromtxt("data/data2.txt",unpack=True)
U3,d3=np.genfromtxt("data/data3.txt",unpack=True)
U4,d4=np.genfromtxt("data/data4.txt",unpack=True)
U5,d5=np.genfromtxt("data/data5.txt",unpack=True)
U6,d6=np.genfromtxt("data/data6.txt",unpack=True)
U7,d7=np.genfromtxt("data/data7.txt",unpack=True)
f=2.54/4#cm pro Kästchen

def fu(x,m,n):
    return m*x+n

params1,cov1 = curve_fit(fu,U1,d1)
params2,cov2 = curve_fit(fu,U2,d2)
params3,cov3 = curve_fit(fu,U3,d3)
params4,cov4 = curve_fit(fu,U4,d4)
params5,cov5 = curve_fit(fu,U5,d5)
params6,cov6 = curve_fit(fu,U6,d6)
params7,cov7 = curve_fit(fu,U7,d7)
x=np.linspace(-40,30)


plt.plot(U1,d1*f,"ok",label="$U_B=210$V")
plt.plot(U2,d2*f,"ob",label="$U_B=250$V")
plt.plot(U3,d3*f,"oy",label="$U_B=300$V")
plt.plot(U4,d4*f,"og",label="$U_B=350$V")
plt.plot(U5,d5*f,"or",label="$U_B=400$V")
plt.plot(U6,d6*f,"om",label="$U_B=450$V")
plt.plot(U7,d7*f,"ok",label="$U_B=500$V")
plt.xlabel("$U_d\;/\;$V")
plt.ylabel("$D\;/\;$cm")


plt.plot(x,fu(x,*params1),"--k")
plt.plot(x,fu(x,*params2),"--b")
plt.plot(x,fu(x,*params3),"--y")
plt.plot(x,fu(x,*params4),"--g")
plt.plot(x,fu(x,*params5),"--r")
plt.plot(x,fu(x,*params6),"--m")
plt.plot(x,fu(x,*params7),"--k")
plt.legend(loc="best")
plt.ylim(-3,3)
plt.savefig("plots/all.pdf")
plt.close()

#Empfindlichkeits Array
E=[round(params1[0],4),round(params2[0],4),round(params3[0],4),round(params4[0],4),round(params5[0],4),round(params6[0],4),round(params7[0],4)]
Eg=[params1[0],params2[0],params3[0],params4[0],params5[0],params6[0],params7[0]]
UB=[1/210,1/250,1/300,1/350,1/400,1/450,1/500]
x2=np.linspace(UB[0],UB[-1])

def fu2(x,m,n):
    return m*x+n
paramsE,covE = curve_fit(fu2,UB,Eg)
errorsE = np.sqrt(np.diag(covE))
unparamsE = unp.uarray(paramsE,errorsE)

plt.plot(x2,fu2(x2,*paramsE),"--r",label="Ausgleichsgerade")
plt.plot(UB,Eg,"xb",label="Messwerte")
plt.ylabel("$D/U_d\;/\;$cm/V")
plt.xlabel("$1/U_B\;/\;$1/V")
plt.legend(loc="best")
plt.savefig("plots/all2.pdf")
plt.show()

p=1.9#cm
L=14.3#cm
d=0.38#cm







#Ausgabe
print(f"""
    Empfindlichkeit
    210V: {params1}
    250V: {params2}
    300V: {params3}
    350V: {params4}
    400V: {params5}
    450V: {params6}
    500V: {params7}

    E: {E}

    Ausgleichsgerade2: {unparamsE}
    abs. abweichung: {35.75-unparamsE[0]}
""")
