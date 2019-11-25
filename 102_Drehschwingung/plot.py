import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit


def Mittelwert(y):
    return (sum(y))/len(y)


#Allgemeine Werte:
M_k = 512.2                                     #Kugelmasse in g
M_kf = unp.uarray(M_k,0.0004*M_k)
M_kf = M_kf* 0.001                              #Umrechnung
d_k = 50.76                                     #Kugeldurchmesser in mm
d_kf = unp.uarray(d_k,0.00007*d_k)
d_kf = d_kf*0.001                               #Umrechnung
J_kh = 22.5                                     #Trägheitsmoment der Kugelhalterung in g*cm^2
J_kh = J_kh * 0.0000001                         #Umrechnung
N = 390                                         #Windungszahl Helmholtzspule
r_h = 78                                        #Radius der Helmholtzspule in mm
r_h = 78*0.001                                  #Umrechnung
d_h = r_h                                       #Abstatnd der Helmholtzspulen
I_max = 1.4                                     #Maximalstrom an der Helmholtzspule in A
d_d = np.array([0.19,0.19,0.19,0.2,0.2])        #Durchmesser des Drahts in mm
d_df = unp.uarray(d_d,0.01)
d_df = d_df * 0.001                             #Umrechnung
d_dm = Mittelwert(d_df)                         #Mittelwert des Drahtdurchmessers

T=np.genfromtxt("data_T.txt", unpack = True)    #Werte der Messung ohne Magnetfeld
T_fehler = unp.uarray(T,0.1)
T_m = Mittelwert(T_fehler)                      #Mittelwert der Periodendauer ohne Magnet

#Längenberechnung
L_1 = np.array([61.2,61.0])
L_1f = unp.uarray(L_1,0.1)
L_2 = np.array([5.4,5.3])
L_2f = unp.uarray(L_2,0.1)
L_s = np.array([2.3,2.3])
L_sf = unp.uarray(L_s,0.1)
L_g = np.array([68.6,68.6])
L_gf = unp.uarray(L_g,0.1)
L1 = Mittelwert(L_1f)+Mittelwert(L_2f)
L2 = Mittelwert(L_gf)-Mittelwert(L_sf)
L = Mittelwert([L1,L2])                         #finale Länge
# print("Länge Draht [cm]: ")
# print(L)
L = L * 0.01                                    #Umrechnung

J_k = (2/5)*M_k*(d_kf/2)**2                     #Trägheitsmoment der Kugel
# print("Trägheitsmoment der Kugel:")
# print(J_k)
# print("Trägheitsmoment der Halterung:")
# print(J_kh)
# print("Gesamtträgheitsmoment")
# print(J_kh + J_k)

D = ((J_k+J_kh)*4*np.pi**2)/T_m**2              #D

E= unp.uarray(21.00*10**10,0.05*10**10)         #Elastizitätsmodul

G = 16/5 * np.pi * (M_kf*L*(d_kf/2)**2)/((T_m**2)*(d_dm/2)**4)+8*np.pi*(J_kh*L)/((T_m**2)*(d_dm/2)**4)  #Schubmodul
my = (E/(2*G))-1                                  #gemäß Formel (2)
#print("my: ")
#print(my)

Q = E/(3*(1-2*my))                              #gemäß Formel (3)
#print("Q: ")
#print(Q)

#Mit Magneten:

data = np.genfromtxt("data.txt", unpack = True)

data_array= np.array([Mittelwert(data[0]),Mittelwert(data[1]),Mittelwert(data[2]),Mittelwert(data[3]),Mittelwert(data[4]),Mittelwert(data[5]),Mittelwert(data[6]),Mittelwert(data[7]),Mittelwert(data[8]),Mittelwert(data[9])])

mag = unp.uarray(1.25663706212*10**(-6),1.5*10**(-10))  #magnetische Feldkonstante
I = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
B = ((mag*I)/2)*(r_h**2/(r_h**2+d_h**2)**(3/2))

m_01 = Mittelwert(((J_k+J_kh)*4*np.pi**2)/(data[0]**2*B[0])-D/B[0])
print(f"m_01 {m_01}")
m_02 = Mittelwert(((J_k+J_kh)*4*np.pi**2)/(data[1]**2*B[1])-D/B[1])
print(f"m_02 {m_02}")
m_03 = Mittelwert(((J_k+J_kh)*4*np.pi**2)/(data[2]**2*B[2])-D/B[2])
print(f"m_03 {m_03}")
m_04 = Mittelwert(((J_k+J_kh)*4*np.pi**2)/(data[3]**2*B[3])-D/B[3])
print(f"m_04 {m_04}")
m_05 = Mittelwert(((J_k+J_kh)*4*np.pi**2)/(data[4]**2*B[4])-D/B[4])
print(f"m_05 {m_05}")
m_06 = Mittelwert(((J_k+J_kh)*4*np.pi**2)/(data[5]**2*B[5])-D/B[5])
print(f"m_06 {m_06}")
m_07 = Mittelwert(((J_k+J_kh)*4*np.pi**2)/(data[6]**2*B[6])-D/B[6])
print(f"m_07 {m_07}")
m_08 = Mittelwert(((J_k+J_kh)*4*np.pi**2)/(data[7]**2*B[7])-D/B[7])
print(f"m_08 {m_08}")
m_09 = Mittelwert(((J_k+J_kh)*4*np.pi**2)/(data[8]**2*B[8])-D/B[8])
print(f"m_09 {m_09}")
m_10 = Mittelwert(((J_k+J_kh)*4*np.pi**2)/(data[9]**2*B[9])-D/B[9])
print(f"m_10 {m_10}")
m_f = np.array([m_01,m_02,m_03,m_04,m_05,m_06,m_07,m_08,m_09,m_10])
m_mittel = Mittelwert(m_f)

print(f"m_mittel: {m_mittel}")
#print (data[0])
print("G:")
print(G)
print("Mittelwert Periodendauer")
print(Mittelwert(T_fehler))
print("Durchmesser Draht")
print(d_d)
print(d_df)
print(d_dm)



#Mittelwerte von T bei verschieden A (0.1-1.0)
T_m01=Mittelwert(data[0])
print(f"T_m01: {T_m01}")

T_m02=Mittelwert(data[1])
print(f"T_m02: {T_m02}")

T_m03=Mittelwert(data[2])
print(f"T_m03: {T_m03}")

T_m04=Mittelwert(data[3])
print(f"T_m04: {T_m04}")

T_m05=Mittelwert(data[4])
print(f"T_m05: {T_m05}")

T_m06=Mittelwert(data[5])
print(f"T_m06: {T_m06}")

T_m07=Mittelwert(data[6])
print(f"T_m07: {T_m07}")

T_m08=Mittelwert(data[7])
print(f"T_m08: {T_m08}")

T_m09=Mittelwert(data[8])
print(f"T_m09: {T_m09}")

T_m10=Mittelwert(data[9])
print(f"T_m10: {T_m10}")


#Plot:
def f(x,a):
    return a*x
#x = np.linspace(0,1)
#plt.plot(x,x**2)
#plt.show()
print(data_array)
plt.plot(unp.nominal_values(B),1/(data_array)**2, 'rx')
params, covariance_matrix = curve_fit(f,unp.nominal_values(B),1/(data_array)**2)
x_plot = np.linspace(0,0.000003)
plt.plot(x_plot,f(x_plot,*params))
#plt.errorbar(unp.nominal_values(B[0][1]),unp.nominal_values(data[0]),yerr=unp.std_devs(data[0]),fmt='rx')
plt.savefig('plot.pdf',bbox_inches='tight')
print(m_mittel)
print(params)
#plt.show()
#plt.close()