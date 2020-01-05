import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit




#5) LED | d=distanc, V=Spannung
d,V=np.genfromtxt("data3.txt", unpack = True)
d=d-18.6
plt.plot(d,V,"rx",label="Messwerte")
plt.ylabel(f"Spannung $U \;/\;V$")
plt.xlabel(f"Entfernung$\;/\;cm$")
plt.savefig('build/plotLED.pdf',bbox_inches='tight')
#plt.show()