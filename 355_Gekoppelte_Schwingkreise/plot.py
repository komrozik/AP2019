import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit
from uncertainties import ufloat


#Schöne Darstellung für Schwebung in der Theorie
#----------------------------------------------------------------------
a=40
b=45
c=0.25
def l(a,b,c,t):
    return c*np.cos(0.5*2*a*t)*np.cos(0.5*(a-b)*t)
def n(a,b,c,t):
    return c*np.sin(0.5*2*a*t)*np.sin(0.5*(a-b)*t)
def g(a,b,c,t):
    return c*np.cos(0.5*(a-b)*t)

t=np.linspace(0,2*np.pi,10000)
plt.plot(t,l(a,b,c,t),"k",label="System 1 mit $I_1(t)$")
plt.plot(t,n(a,b,c,t),label="System 2 mit $I_2(t)$")
plt.plot(t,g(a,b,c,t),"r--",label="$Schwebung im System 1$")
plt.plot(t,-g(a,b,c,t),"r--")
plt.ylim(-c-0.25,c+0.25)
plt.xlim(0,2*np.pi)
plt.yticks([-c,0,c],["$-I_0$",0,"$I_0$"])
plt.xlabel("t")
plt.ylabel("I(t)")
plt.xticks([])
plt.legend()
plt.savefig("build/schwebung.pdf",bbox_inches='tight')
