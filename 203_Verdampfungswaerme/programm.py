import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
#np.seterr(divide='ignore', invalid='ignore')
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit


#IMPORT DATA
T,p=np.genfromtxt("data1.txt", unpack = True)
print(f"p:{p}\n")
print(f"T:{T}")


plt.plot(1/T,np.log(p),label="ln(p)=-m$\cdot$1/T")#numpy log=ln und log!=log
plt.xlabel("ln($p$)")
plt.legend()
plt.ylabel("$1/T$")

plt.show()