import numpy as np


A1 = 4.985
dA1 = 0.03
A2 = 0.52
dA2 = 0.01
A3 = 0.29
dA3 = 0.01
A4 = 0.835
dA4 = 0.02
A5 = 0.45
dA5 = 0.01
A6 = 0.33
dA6 = 0.01
A7 = 0.63
dA7 = 0.02
A8 = 1.235
dA8 = 0.02
A9 = 4.515
dA9 = 0.03

B1 = 1.09
dB1 = 0.02
B2 = 0.59
dB2 = 0.01
B3 = 0.255
dB3 = 0.01
B4 = 0.485
dB4 = 0.1
B5 = 1.210
dB5 = 0.02
B6 = 1.320
dB6 = 0.02
B7 = 0.930
dB7 = 0.02
B8 = 0.810
dB8 = 0.02

C1 = 3.48
dC1 = 0.03

D1 = 3.48
dD1 = 0.03
D2 = 1.5
dD2 = 0.02
D3 = 1.03
dD3 = 0.02
D4 = 0.5
dD4 = 0.01
D5 = 0.21
dD5 = 0.01

#Volumen berechnen:

#Quader Volumen:
QV = A1*A9*B5
QA = B5*(0.5* A6)**2*np.pi+B5*(0.5*A3)**2*np.pi+B2*(0.5*D3)**2*np.pi+(B5-B2)*(0.5*A2)**2*np.pi+(B5-B3)*(0.5*A5)**2*np.pi+1/3*np.pi*B3*((0.5*A4)**2+(0.5*A4*0.5*A5)+(0.5*A5)**2)+B4*(0.5*A8)**2*np.pi+(B5-B4)*(0.5*A7)**2*np.pi
QG= QV - QA
print("Quader Volumen:")
print(QG)
#Zylinder Volumen:
ZV = B6*(0.5*D1)**2*np.pi
ZA = (D5*B7*(D1-D2))+(D4*B8*(D1-D2))+B1*(0.5*D2)**2*np.pi+(B6-B1)*(0.5*A7)**2*np.pi
ZG=ZV-ZA
print("Zylinder Volumen:")
print(ZG)
print("Gesamt Volumen:")
print(QG+ZG)

#Fehler:

#Quader Fehler:

#Zykinder Fehler:
ZF= np.sqrt((((B6-B1)*A7*np.pi*dA7)**2)+((0.5*D2)**2*np.pi*dB1)**2+(-(0.5*A7)**2*np.pi*dB1)**2+((0.5*D1)**2*np.pi*dB6)**2+((0.5*A7)**2*np.pi*dB1)**2+((0.5*D1)**2*np.pi*dB6)**2+((0.5*A7)**2*np.pi*dB6)**2+(D5*(D1-D2)*dB7)**2+(B6*D1*np.pi*dD1)**2+(D5*B7*dD1)**2+(D4*B8*dD1)**2+(-1*D5+B7*dD2)**2+(-1*D4*B8*dD2)**2+(B1*D2*np.pi*dD2)**2+(B1*D2*np.pi*dD2)**2+(B8*(D1-D2)*dD4)**2+(B7*(D1-D2)*dD5)**2)
QF= np.sqrt(((A9*B5)*dA1)**2+((B5-B2)*0.5*A2*np.pi*dA2)**2+((B5*0.5*A3)*dA3)**2+(1/3*np.pi*B3*(0.5*A4+0.25*A5)*dA4)**2+((B5-B3)*0.5*A5*np.pi+1/3*np.pi*B3*(0.5*A5+0.25*A4)*dA5)**2+((B5*0.5*A6*np.pi)*dA6)**2+(((B5-B4)*0.5*A7)*dA7)**2+((B4*0.5*A8*np.pi)*dA8)**2+((A1*B5)*dA9)**2+(((0.5*D3)**2*np.pi+(-(0.5*A2)**2*np.pi))*dB2)**2+((0.5*A8)**2*np.pi*dB4)**2+((A1*A9+(0.5*A6)**2*np.pi+(0.5*A3)**2*np.pi+(0.5*A2)**2*np.pi+(0.5*A5)**2*np.pi+(0.5*A7)**2*np.pi)*dB5)**2+((B2*0.5*D3)*dD3)**2)
GF= np.sqrt(QF**2+ZF**2)
print("Gesamtfehler:")
print(GF)
