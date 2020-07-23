import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import *
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import scipy.constants as const

g1,b1 = np.genfromtxt("data/data1.txt",unpack=True)
b1 = b1 - g1
g1 = g1 - 30
g1 = g1*10
b1 = b1*10
g2,b2 = np.genfromtxt("data/data2.txt",unpack=True)
b2 = b2 - g2
g2 = g2 - 30
g2 = g2*10
b2 = b2*10
e,g3_1,g3_2 = np.genfromtxt("data/data3.txt",unpack=True)
d = g3_2 - g3_1
e = e -30
d = d*10
e = e*10
A,b4,V = np.genfromtxt("data/data4.txt",unpack=True)
b4 = b4 - A
A = A - 30

f1 = (g1*b1)/(g1+b1)
f1 = (np.sum(f1))/(len(f1))
delta_f1 = 100-f1
deltarel_f1 = delta_f1/100

zero1 = np.zeros(12)

def f_1(x,m,n):
    return ((b1)/(g1))*x+b1

plt.plot([0,g1[0]],[b1[0],0],"-")
plt.plot([0,g1[1]],[b1[1],0],"-")
plt.plot([0,g1[2]],[b1[2],0],"-")
plt.plot([0,g1[3]],[b1[3],0],"-")
plt.plot([0,g1[4]],[b1[4],0],"-")
plt.plot([0,g1[5]],[b1[5],0],"-")
plt.plot([0,g1[6]],[b1[6],0],"-")
plt.plot([0,g1[7]],[b1[7],0],"-")
plt.plot([0,g1[8]],[b1[8],0],"-")
plt.plot([0,g1[9]],[b1[9],0],"-")
plt.plot([0,g1[10]],[b1[10],0],"-")
plt.plot([0,g1[11]],[b1[11],0],"-")
plt.plot(zero1,b1,"xr",label="b Werte")
plt.plot(g1,zero1,"xg",label="g Werte")
plt.plot
plt.xlabel("g s")
plt.ylabel("b s")
plt.legend(loc="best")
plt.savefig("plots/plot1.pdf")
plt.close()

f2 = (g2*b2)/(g2+b2)
f2 = (np.sum(f2))/(len(f2))
delta_f2 = 50-f2
deltarel_f2 = delta_f2/50


plt.plot([0,g2[0]],[b2[0],0],"-")
plt.plot([0,g2[1]],[b2[1],0],"-")
plt.plot([0,g2[2]],[b2[2],0],"-")
plt.plot([0,g2[3]],[b2[3],0],"-")
plt.plot([0,g2[4]],[b2[4],0],"-")
plt.plot([0,g2[5]],[b2[5],0],"-")
plt.plot([0,g2[6]],[b2[6],0],"-")
plt.plot([0,g2[7]],[b2[7],0],"-")
plt.plot([0,g2[8]],[b2[8],0],"-")
plt.plot([0,g2[9]],[b2[9],0],"-")
plt.plot([0,g2[10]],[b2[10],0],"-")
plt.plot([0,g2[11]],[b2[11],0],"-")
plt.plot(zero1,b2,"xr",label="b Werte")
plt.plot(g2,zero1,"xg",label="g Werte")
plt.plot
plt.xlabel("g s")
plt.ylabel("b s")
plt.legend(loc="best")
plt.savefig("plots/plot2.pdf")
plt.close()

print("Bessel")

f3 = (e**2-d**2)/(4*e)
# f3 = (np.sum(f3))/(len(f3))
delta_f3 = 100-(np.sum(f3))/(len(f3))
deltarel_f3 = delta_f3/100


print("Abbe")

plt.plot(A,(1+1/V),label="g Auftragung")
plt.plot
plt.xlabel("g s")
plt.ylabel("b s")
plt.legend(loc="best")
plt.savefig("plots/plot3.pdf")
plt.close()

plt.plot(A,(1+V),label="b Auftragung")
plt.plot
plt.xlabel("g s")
plt.ylabel("b s")
plt.legend(loc="best")
plt.savefig("plots/plot4.pdf")
plt.close()


#Ausgabe
print(f"""
Brennweite 1: {f1}
Fehler absolut: {delta_f1}
Fehler relativ: {deltarel_f1}
Brennweite 2: {f2}
Fehler absolut: {delta_f2}
Fehler relativ: {deltarel_f2}
Brennweite 3: {(np.sum(f3))/(len(f3))}
Fehler absolut: {delta_f3}
Fehler relativ: {deltarel_f3}
Bei bessel:
e ist : {e}
d ist : {d}
f ist : {f3}
""")
