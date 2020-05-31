import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import *
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import scipy.constants as const

tv,Nv=np.genfromtxt("data/Vanadium.dat",unpack=True)
Nv=unp.uarray(Nv,np.sqrt(Nv))

def function(t,lam,N0): 
    return exp(-lam*t)+N0

popv,covv=curve_fit(function,tv,noms(Nv))
print(popv)

plt.plot(log(tv),noms(Nv),"xb",label="Messwerte")
plt.errorbar(log(tv),noms(Nv),yerr=stds(Nv),)
plt.plot(log(tv),exp(popv[0])+log(tv)+popv[1],"--r",label="Ausgleichsgerade")
plt.xlabel("ln(t)")
plt.ylabel("$N(t)\;/\;Imp/s$")
plt.legend(loc="best")
plt.show()



#Ausgabe
print(f"""

""")
