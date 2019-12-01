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

#IMPORT DATA
data_raw=np.genfromtxt("data.txt", unpack = True)
data=data_raw/5 #für alle außer für Schwebung
l1=72*0.01#m
l2=80*0.01#m

#Frei schwingende Pendel 1 & 2 mit Schwingungsdauer T
T_P1 = data[0] #T_1
T_P2 = data[1] #T_2 
M_P1 = Mittelwert(T_P1)
M_P2 = Mittelwert(T_P2)
print(f"Frei schwingende Pendel - Mittelwerte\nPendel1 (72cm): {M_P1}\nPedel2 (72cm): {M_P2}")
print(f"Werte Pendel1 {data[0]}\nWerte Pendel2: {data[1]}\n")

#Gleichsinnige Schwingung mit Pendellängen l1 und l2
T1_p = data[2] #T_+ 72cm
T2_p = data[3] #T_+ 80cm
M1_p = Mittelwert(T1_p)
M2_p = Mittelwert(T2_p)
wg1_p= 2*np.pi/M1_p #gemessen Frequenz mit wg=2pi/T
wg2_p= 2*np.pi/M2_p
print(f"Gleichsinnige Schwingung - Mittelwert\nPendel (72cm): {M1_p}\nPendel (80cm): {M2_p}")
print(f"Werte Pendel1 {data[2]}\nWerte Pendel2: {data[3]}\n")

#Gegenphasige Schwingung mit Pendellänge l1 und l2
T1_m = data[7] #T_- 72cm
T2_m = data[4] #T_- 80cm
M1_m = Mittelwert(T1_m)
M2_m = Mittelwert(T2_m)
wg1_m= 2*np.pi/M1_m #Frequenz mit w=1/T
wg2_m= 2*np.pi/M2_m
print(f"Gegenphasige Schwingung - Mittelwert\nPendel (72cm): {M1_m}\nPednel (80cm): {M2_m}"+'\x1b[6;30;42m' + ' Komisch!' + '\x1b[0m')
print(f"Werte Pendel1 {data[7]}\nWerte Pendel2: {data[4]}\n")
#!!! Längeres Pendel schneller --> Widerspruch?

#Gemessene Scwebungsdauer und Schwingungsdauer mit l1 und l2
T1_schwi = data[8] #T_s_schwingung 72cm
T1_schwe = data_raw[9] #T_s_schwebung 72cm
M1_schwi = Mittelwert(T1_schwi)
M1_schwe = Mittelwert(T1_schwe)
wg1_schwi= 2*np.pi/M1_schwi #Frequenz mit w=1/T
print(f"Schwebung/Schwingung 72cm - Mittelwert\nSchwingung: {M1_schwi}\nSchwebung: {M1_schwe}")
print(f"Schwingung {data[8]}\nSchwebung: {data_raw[9]}\n")

T2_schwi = data[5] #T_s_schwingung 80cm
T2_schwe = data_raw[6] #T_s_schwebung 80cm
M2_schwi = Mittelwert(T2_schwi)
M2_schwe = Mittelwert(T2_schwe)
wg2_schwi= 2*np.pi/M2_schwi
print(f"Schwebung/Schwingung 80cm - Mittelwert\nSchwingung: {M2_schwi}\nSchwebung: {M2_schwe}")
print(f"Schwingung {data[5]}\nSchwebung: {data_raw[6]}\n\n")
#------------------------------------------------------------------------------------------------------------------------------------------------------------

#Kopplungskonstante K
DK1=(T1_p**2-T1_m**2)/(T1_p**2+T1_m**2) #über Mittelwerte
K1=Mittelwert(DK1)
K2=(M2_p**2-M2_m**2)/(M2_p**2+M2_m**2)
print(f"Kopplungskonstante K (72cm): {K1}\nKopplungskonstante K (80cm): {K2}\n")

#Frequenzen
#Gegensinnig
wb1_m=np.sqrt((9.81+2*K1)/l1) #berechnete Frquenz mit wb=sqrt(g/l+2K/l)
wb2_m=np.sqrt((9.81+2*K2)/l2)
print(f"Gegensinnig Frequenzen:\nGemessen (72cm):{wg1_m}\nBerechnet (72cm):{wb1_m}\nGemessen (80cm):{wg2_m}\nBerechnet (80cm):{wb2_m}\n")
#Gleichsinnig
wb1_p= np.sqrt(9.81/l1) #berechnete Frequenz mit wb=sqrt(g/l)
wb2_p= np.sqrt(9.81/l2)
print(f"Gleichsinnig Frequenzen:\nGemessen (72cm):{wg1_p}\nBerechnet (72cm):{wb1_p}\nGemessen (80cm):{wg2_p}\nBerechnet (80cm):{wb2_p}\n")


#Schwebungsdauern TS
#TS1=(M1_p*M1_m)/(M1_p-M1_m) #über Mittelwert
TS1=(T1_p*T1_m)/(T1_p-T1_m)

#TS2=(M2_p*M2_m)/(M2_p-M2_m) #M2_m=1.77
TS2=(T2_p*T2_m)/(T2_p-T2_m)


print(f"Schwebungsdauer\nTS (72cm, berechnet): {Mittelwert(TS1)}\nTS (80cm, berechnet): {Mittelwert(TS2)}")
print(f"TS (72cm, gemessen): {M1_schwe}\nTS (80cm, gemessen): {M2_schwe}\n")
#print(f"test für 72cm: {Mittelwert(test1)}\ntest für 80cm {Mittelwert(test2)}")


#Gekoppelt
wb1_schwi= wb1_p-wb1_m
wb2_schwi= wb2_p-wb2_m
print(f"Gekoppelt Frequenzen:\nGemessen (72cm):{wg1_schwi}\nBerechnet (72cm):{wb1_schwi}\nGemessen (80cm):{wg2_schwi}\nBerechnet (80cm):{wb2_schwi}\n")


Kgtest = (wg1_m**2-wg1_p**2)/(wg1_m**2+wg1_p**2)
Kbtest = (wb1_m**2-wb1_p**2)/(wb1_m**2+wb1_p**2)
print(f"K1: gemessen {Kgtest}, berechnet {Kbtest}")

#--------------------------------------------------------------------------------
#PLOTS
l=np.linspace(0,2)
plt.plot(l,l**2)
plt.show()