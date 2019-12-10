import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import sympy as sym
from scipy import integrate
from scipy.optimize import curve_fit

t,p1,p2,T1,T2,P=np.genfromtxt("data.txt",unpack=True)

#T1 Plot
plt.plot(t,T1,"b+",label="T1-Verlauf")
plt.xlabel("Zeit $t\;/\;min$")
plt.ylabel("Temperatur $T1\;/\;°C$")
plt.show()
plt.close()
#T2 Plot
plt.plot(t,T2,"b+",label="T2-Verlauf")
plt.xlabel("Zeit $t\;/\;min$")
plt.ylabel("Temperatur $T2\;/\;°C$")
plt.show()
plt.close()

#p1 Plot
plt.plot(t,p1,"r+",label="r+")
plt.xlabel("Zeit $t\;/\;min$")
plt.ylabel("Druck $p1\;/\;\mathrm{bar}$")
plt.show()
plt.close()

#p2 Plot
plt.plot(t,p2,"r+",label="r+")
plt.xlabel("Zeit $t\;/\;min$")
plt.ylabel("Druck $p2\;/\;\mathrm{bar}$")
plt.show()
plt.close()