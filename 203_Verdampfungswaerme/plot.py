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
data1[0] = data1[0]+273
data2=np.genfromtxt("data2.txt", unpack = True)
data2[0] = data2[0]+273
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

R = 8.314462
####################################################################
# Niedrigdruck
####################################################################
# Bestimmung der Dampfdruckkurve von 30 bis 100 mbar (Wasser)

params_a, covariance_matrix_a = curve_fit(f,1/data1[0],np.log(data1[1]))

plt.plot(1/data1[0],np.log(data1[1]),'kx',label="Dampfdruck")

x_plot = np.linspace(0.00267,0.00331)
plt.plot(x_plot,f(x_plot,*params_a),label='Fit')
errors_a = np.sqrt(np.diag(covariance_matrix_a))
#plt.yscale('log')
plt.ylabel(f"log(p)")
plt.xlabel(f"1/T in 1/K")
plt.legend()
plt.savefig("build/plot_a.pdf",bbox_inches='tight')
#plt.show()
plt.close()
print(errors_a)
print(f"Parameter der Ausgleichskurve für die Messung unter 1 Bar:")
print(f"Steigung:{params_a[0]}+- {errors_a[0]}")
print(f"y-Achsenabschnitt:{params_a[1]}+-{errors_a[1]}")

# Berechnungen für den Bereich unter 1 Bar
unparams_a = unp.uarray(params_a,errors_a)
L = -unparams_a[0]*R
print(f"L für unter 1 Bar: {L}")
L_a = R*373
print(f"La für Niedrigdruck {L_a}")
L_i = L-L_a
print(f"Li für Niedrigdruck {L_i}")
L_im = L_i / (6.022 * 10**23)
L_im = L_im * (6.242 * 10**18)
print(f"Li pro Molekül {L_im}")
####################################################################
# Hochdruck
####################################################################
# Bestimmung der Dampfdruckkurve von 1 bis 15 bar    (Wasser)

params_b, covariance_matrix_b = curve_fit(pol,data2[0],data2[1])
errors_b = np.sqrt(np.diag(covariance_matrix_b))
unparams_b = unp.uarray(params_b,errors_b)
plt.plot(data2[0],data2[1],"kx",label="Dampfdruck")

x_plot = np.linspace(100+273,200+273)
plt.plot(x_plot,pol(x_plot,*params_b),label='Fit')
plt.ylabel(f"log(p)")
plt.xlabel(f"T in K")
plt.legend()
plt.savefig("build/plot_b.pdf",bbox_inches='tight')
#plt.show()
plt.close()

print(f"Die Parameter im Hochdruckbereich sind:")
print(f"Parameter :{unparams_b}")
print(f"Parameter a: {unparams_b[0]}")
print(f"Parameter b: {unparams_b[1]}")
print(f"Parameter c: {unparams_b[2]}")
print(f"Parameter d: {unparams_b[3]}")

A = 0.9 

# def L_berechnet(T,a,b,c,d):
#     return(T/(pol(T,a,b,c,d)) * ( (R*T/2) + np.sqrt(( R*T/2 )**2 + A*(pol(T,a,b,c,d)) ) ) (3*a*T**2+2*b*T+c))

L_berechnetp = data2[0]/(pol(data2[0],*params_b)) * ( (R*data2[0]/2) + np.sqrt(( R*data2[0]/2 )**2 + A*(pol(data2[0],*params_b)) ) )* (3*params_b[0]*data2[0]**2+2*params_b[1]*data2[0]+params_b[2])
L_berechnetn = data2[0]/(pol(data2[0],*params_b)) * ( (R*data2[0]/2) - np.sqrt(( R*data2[0]/2 )**2 + A*(pol(data2[0],*params_b)) ) )* (3*params_b[0]*data2[0]**2+2*params_b[1]*data2[0]+params_b[2])
plt.plot(data2[0],L_berechnetp,"rx",label='Wurzel addiert')
#plt.plot(data2[0],L_berechnetn,label='Wurzel subtrahiert')
plt.ylabel(f"L in Joule/mol")
plt.xlabel(f"T in K")
plt.legend()
plt.savefig("build/plot_c.pdf",bbox_inches='tight')
#plt.show()
plt.close()
print(f"Werte:{L_berechnetp} ")