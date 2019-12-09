import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
#np.seterr(divide='ignore', invalid='ignore')
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

#IMPORT

data1=np.genfromtxt("data1.txt", unpack = True)
data1[1] = data1[1]*10**(-4)
data2=np.genfromtxt("data2.txt", unpack = True)
#   a)Diagramm des Drucks und der Temperatur

# plt.plot(1/data1[0],np.log(data1[1]),'kx',label="Dampfdruck")
# plt.yscale('log')
# plt.ylabel(f"log(p) in milli Bar")
# plt.xlabel(f"1/T")
# plt.legend()
# plt.show()
# plt.savefig("build/plot_a.pdf",bbox_inches='tight')
# plt.close()


# Definitionen für den curvefit

def f(x,a,b):
    return a*x+b

def pol(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d
####################################################################
# Niedrigdruck
####################################################################
# Bestimmung der Dampfdruckkurve von 30 bis 100 mbar (Wasser)

params_a, covariance_matrix_a = curve_fit(f,1/data1[0],np.log(data1[1]))

plt.plot(1/data1[0],np.log(data1[1]),'kx',label="Dampfdruck")

x_plot = np.linspace(0.01,0.034)
plt.plot(x_plot,f(x_plot,*params_a),label='Fit')
errors_a = np.sqrt(np.diag(covariance_matrix_a))
#plt.yscale('log')
plt.ylabel(f"log(p) in Bar")
plt.xlabel(f"1/T")
plt.legend()
plt.savefig("build/plot_a.pdf",bbox_inches='tight')
plt.show()
plt.close()
print(errors_a)
print(f"Parameter der Ausgleichskurve für die Messung unter 1 Bar:")
print(f"Steigung:{params_a[0]}+- {errors_a[0]}")
print(f"y-Achsenabschnitt:{params_a[1]}+-{errors_a[1]}")

# Berechnungen für den Bereich unter 1 Bar
unparams_a = unp.uarray(params_a,errors_a)
L_a = -unparams_a[0]*8.314462
print(f"L für unter 1 Bar: {L_a}")

####################################################################
# Hochdruck
####################################################################
# Bestimmung der Dampfdruckkurve von 1 bis 15 bar    (Wasser)

params_b, covariance_matrix_b = curve_fit(pol,data2[0],data2[1])

plt.plot(data2[0],data2[1],"kx",label="Dampfdruck")

x_plot = np.linspace(100,200)
plt.plot(x_plot,pol(x_plot,*params_b),label='Fit')
plt.ylabel(f"log(p) in Bar")
plt.xlabel(f"1/T")
plt.legend()
plt.savefig("build/plot_b.pdf",bbox_inches='tight')
#plt.show()
plt.close()
