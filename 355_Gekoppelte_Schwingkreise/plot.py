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
plt.plot(t,g(a,b,c,t),"r--",label="Schwebung im System 1")
plt.plot(t,-g(a,b,c,t),"r--")
plt.ylim(-c-0.25,c+0.25)
plt.xlim(0,2*np.pi)
plt.yticks([-c,0,c],["$-I_0$",0,"$I_0$"])
plt.xlabel("t")
plt.ylabel("I(t)")
plt.xticks([0],[0])
plt.legend()
plt.savefig("build/schwebung1.pdf",bbox_inches='tight')
plt.close()

plt.plot(t,n(a,b,c,t),label="System 2 mit $I_2(t)$")
plt.plot(t,g(a,b,c,t),"r--",label="Schwebung im System 1")
plt.plot(t,-g(a,b,c,t),"r--")
plt.ylim(-c-0.25,c+0.25)
plt.xlim(0,2*np.pi)
plt.yticks([-c,0,c],["$-I_0$",0,"$I_0$"])
plt.xlabel("t")
plt.ylabel("I(t)")
plt.xticks([0],[0])
plt.legend()
plt.savefig("build/schwebung2.pdf",bbox_inches='tight')
plt.close()

#----------------------------------------------------------------------

L = 32.351 #mH
L = L*10**(-3) # in H
C = 0.8015 #nF
C = C*10**(-9) # in Farad
CSp = 0.037 #nF
CSp = CSp*10**(-9) # in Farad

C_k,n_max,n_min = np.genfromtxt("dataa.txt",unpack = True)

C_k = C_k*10**(-9) # in Farad
C_k = unp.uarray(C_k,C_k*0.03)

#Theoretische Werte für die Frequenzen
nu_p_t = 1/(2 * np.pi * (L*(C+CSp))**(1/2))
nu_p_t = np.array([nu_p_t,nu_p_t,nu_p_t,nu_p_t,nu_p_t,nu_p_t,nu_p_t,nu_p_t])
nu_m_t = 1/(2 * np.pi * (L * ( (1/C+2/C_k)**(-1)+ CSp) )**(1/2))
print(f"Frequenz positiv {nu_p_t}")
print(f"Frequenz negativ {nu_m_t}")

#Verhältnis der Frequenzen:
print(f"Verhältnis der Frequenzen experimentell, 1 durch : {n_max}")

n_theorie = (nu_m_t + nu_p_t) / (2 * (nu_m_t - nu_p_t))
print(f"Verhältnis der Frequenzen theoretisch, 1 durch : {n_theorie}")

# Abweichung des Experiments von der Theorie:
a_n = np.abs(n_theorie-n_max)/n_theorie
print(f"Abweichung von der theorie: {a_n}\n")

# Abweichung des Experiments Resonanzfrequenz (nu plus) von der Theorie:
nu_p = np.array([30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5])
nu_p = nu_p * 1000
a_p = np.abs(nu_p_t - nu_p) / nu_p_t
print(f"Abweichung der Resonanzfrequenz von der theorie: {a_p}\n")

# Abweichung des Experiments gegenphasige Schwingungsfrequenz (nu minus) von der Theorie:
nu_m = np.array([47.2,40.3,37.2,35.9,34.9,34.0,33.4,32.9]) * 10**3
a_m = np.abs(nu_m_t - nu_m) / nu_m_t
print(f"Abweichung der Frequenz der Fundamentalschwingung der gegenphasigen Schwingung von der theorie: {a_m}\n")




R= 48
C_k,nu_p,U_G_p,U2_p = np.genfromtxt("dataw0N.txt",unpack = True)
C_k,nu_m,U_G_m,U2_m = np.genfromtxt("datawpiN.txt",unpack = True)
nu_p = nu_p * 1000
nu_m = nu_m *1000
C_k = C_k*10**(-9)
F = 15/(2.3*5)
I_G_p = F * U_G_p*5
I_2_p = U2_p /(2*R)
I_G_m = F*U_G_m
I_2_m = U2_m /(2*R)
I_k = I_G_p - I_G_m

# def I_theo_p(U,w,k):
#     return U*w*k
def I_theo_m(U,w,k):
    return U * 1/( (4 * w**2 * k**2 * R**2 * (w * L - 1/w * (1/C + 1/k) )**2 + (1/(w*k) - w*k*(w * L - 1/w * (1/C + 1/k) )**2 +w*R**2*k)**2 )**(1/2) )

x = np.linspace(nu_m[0], nu_m[7],10000)

plt.plot(nu_m, I_2_m, "x", label = r"Strom bei $\nu^-$ Messwerten")
plt.plot(x, I_theo_m(2.4*5,2*np.pi*x,C_k[0]), "-", label = r"$C_k = 1.01$nF")
plt.plot(x, I_theo_m(2.4*5,2*np.pi*x,C_k[1]), "-", label = r"$C_k = 2.03$nF")
plt.plot(x, I_theo_m(2.4*5,2*np.pi*x,C_k[2]), "-", label = r"$C_k = 3.00$nF")
plt.plot(x, I_theo_m(2.4*5,2*np.pi*x,C_k[3]), "-", label = r"$C_k = 4.00$nF")
plt.plot(x, I_theo_m(2.4*5,2*np.pi*x,C_k[4]), "-", label = r"$C_k = 5.02$nF")
plt.plot(x, I_theo_m(2.4*5,2*np.pi*x,C_k[5]), "-", label = r"$C_k = 6.37$nF")
plt.plot(x, I_theo_m(2.4*5,2*np.pi*x,C_k[6]), "-", label = r"$C_k = 8.00$nF")
plt.plot(x, I_theo_m(2.4*5,2*np.pi*x,C_k[7]), "-", label = r"$C_k = 9.99$nF")
plt.xlabel(r"$\nu$ / Hz")
plt.ylabel(r"$I_2$ / A")
plt.legend(loc = "best")
plt.savefig("build/Stromverlauf.pdf")
plt.show()
plt.close()

I_theoriewerte = np.array([I_theo_m(2.4*5,2*np.pi*nu_m[0],C_k[0]),
I_theo_m(2.4*5,2*np.pi*nu_m[1],C_k[1]),
I_theo_m(2.4*5,2*np.pi*nu_m[2],C_k[2]),
I_theo_m(2.4*5,2*np.pi*nu_m[3],C_k[3]),
I_theo_m(2.4*5,2*np.pi*nu_m[4],C_k[4]),
I_theo_m(2.4*5,2*np.pi*nu_m[5],C_k[5]),
I_theo_m(2.4*5,2*np.pi*nu_m[6],C_k[6]),
I_theo_m(2.4*5,2*np.pi*nu_m[7],C_k[7])])
I_theoriewerte = I_theoriewerte
print(f"Theoriewerte I: {I_theoriewerte}")

print(f"Experimentwerte: {I_2_m}")

a_I = np.abs(I_theoriewerte - I_2_m)/I_theoriewerte
print(f"Abweichung: {a_I}")


# plt.plot(C_k, I_G_p, "x", label = r"Strom bei $C_k$ Messwerten")
# plt.xlabel(r"$C_k$ / F")
# plt.ylabel(r"$I_2$ / A")
# plt.legend(loc = "best")
# plt.savefig("build/Stromverlauf2.pdf")
# plt.show()
# plt.close()

# plt.plot(C_k, I_2_p, "x", label = r"Strom bei $C_k$ Messwerten")
# plt.xlabel(r"$C_k$ / F")
# plt.ylabel(r"$I_2$ / A")
# plt.legend(loc = "best")
# plt.savefig("build/Stromverlauf3.pdf")
# plt.show()
# plt.close()

# plt.plot(C_k, I_k, "x", label = r"Strom bei $C_k$ Messwerten")
# plt.xlabel(r"$C_k$ / F")
# plt.ylabel(r"$I_2$ / A")
# plt.legend(loc = "best")
# plt.savefig("build/Stromverlauf4.pdf")
# plt.show()
# plt.close()

# plt.plot(C_k, I_G_m, "x", label = r"Strom bei $C_k$ Messwerten")
# plt.xlabel(r"$C_k$ / F")
# plt.ylabel(r"$I_2$ / A")
# plt.legend(loc = "best")
# plt.savefig("build/Stromverlauf5.pdf")
# plt.show()
# plt.close()

# plt.plot(C_k, I_2_m, "x", label = r"Strom bei $C_k$ Messwerten")
# plt.xlabel(r"$C_k$ / F")
# plt.ylabel(r"$I_2$ / A")
# plt.legend(loc = "best")
# plt.savefig("build/Stromverlauf6.pdf")
# plt.show()
# plt.close()

# U_Ampl_p = np.array([1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4])
# U_Ampl_m = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
# R = 48
# U_1 = 2.4
# U_2 = 2.4
# x = np.array([0, 1.3 * 10**(-8)])

# I_1t = np.array([U_1 / (2*R), U_1 / (2*R)])
# I_2t = np.array([U_2 / (2*R), U_2 / (2*R)])
# I_1 = U_Ampl_p / R
# I_2 = U_Ampl_m / R

# plt.plot(unp.nominal_values(C_k), I_1, "x", label = r"Strom bei $\nu^+$ Messwerten")
# plt.plot(unp.nominal_values(C_k), I_2, "x", label = r"Strom bei $\nu^-$ Messwerten")
# plt.plot(x, I_1t, "-", label = r"Strom bei $\nu^+$ Theoriewerten")
# plt.plot(x, I_2t, "-", label = r"Strom bei $\nu^-$ Theoriewerten")
# plt.xlabel(r"$C_k$ / nF")
# plt.ylabel(r"$I$ / A")
# plt.legend(loc = "best")
# plt.savefig("Stromverlauf.pdf")
# plt.show()
# plt.clf()


y = np.array([0, 1.3 * 10**(-8)])
Ck_array = np.linspace(0.1 *10**(-8),1.3 * 10**(-8),1000)
def nu_m_function(k):
   return 1/(2 * np.pi * (L * ( (1/C+2/k)**(-1)+ CSp) )**(1/2))

def nu_p_function(k):
   return 1/(2 * np.pi * (L*(C+CSp))**(1/2)) *(k/k)

plt.plot(unp.nominal_values(C_k), nu_p, "x", label = r"$\nu^+$ Messwerte")
plt.plot(unp.nominal_values(C_k), nu_m, "x", label = r"$\nu^-$ Messwerte")
plt.plot(Ck_array, nu_p_function(Ck_array), "-", label = r"$\nu^+$ Theoriekurve")
plt.plot(Ck_array, nu_m_function(Ck_array), "-", label = r"$\nu^-$ Theoriekurve")
plt.xlabel(r"$C_k$ / F")
plt.ylabel(r"$\nu$ / kHz")
plt.legend(loc = "best")
plt.savefig("build/Frequenzverlauf.pdf")
plt.show()