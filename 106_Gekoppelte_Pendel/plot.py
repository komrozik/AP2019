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

l1=72#cm
l2=80#cm

#Frei schwingende Pendel 1 & 2 mit Schwingungsdauer T
T_P1 = data[0] #T_1
T_P2 = data[1]#T_2 
M_P1 = Mittelwert(T_P1)
M_P2 = Mittelwert(T_P2)
print(f"Frei schwingende Pendel - Mittelwerte\nPendel1 (72cm): {M_P1}\nPedel2 (72cm): {M_P2}\n")

#Gleichsinnige Schwingung mit Pendellängen l1 und l2
T1_p = data[2] #T_+ 72cm
T2_p = data[3] #T_+ 80cm
M1_p = Mittelwert(T1_p)
M2_p = Mittelwert(T2_p)
print(f"Gleichsinnige Schwingung - Mittelwert\nPendel (72cm): {M1_p}\nPendel (80cm): {M2_p}\n")

#Gegenphasige Schwingung mit Pendellänge l1 und l2
T1_m = data[7] #T_- 72cm
T2_m = data[4] #T_- 80cm
M1_m = Mittelwert(T1_m)
M2_m = Mittelwert(T2_m)
print(f"Gegenphasige Schwingung - Mittelwert\nPendel (72cm): {M1_m}\nPednel (80cm): {M2_m}"+'\x1b[6;30;42m' + ' Komisch!' + '\x1b[0m' + "\n")
#!!! Längeres Pendel schneller --> Widerspruch?

#Gemessene Scwebungsdauer und Schwingungsdauer mit l1 und l2
T1_schwi = data[8] #T_s_schwingung 72cm
T1_schwe = data_raw[9] #T_s_schwebung 72cm
M1_schwi = Mittelwert(T1_schwi)
M1_schwe = Mittelwert(T1_schwe)
print(f"Schwebung/Schwingung 72cm - Mittelwert\nSchwingung: {M1_schwi}\nSchwebung: {M1_schwe}\n")

T2_schwi = data[5] #T_s_schwingung 80cm
T2_schwe = data_raw[6] #T_s_schwebung 80cm
M2_schwi = Mittelwert(T2_schwi)
M2_schwe = Mittelwert(T2_schwe)
print(f"Schwebung/Schwingung 80cm - Mittelwert\nSchwingung: {M2_schwi}\nSchwebung: {M2_schwe}\n\n")
#------------------------------------------------------------------------------------------------------------------------------------------------------------

#Kopplungskonstante K
K1=(M1_p**2-M1_m**2)/(M1_p**2+M1_m**2) #über Mittelwerte
K2=(M2_p**2-M2_m**2)/(M2_p**2+M2_m**2)
print(f"Kopplungskonstante K (72cm): {K1}\nKopplungskonstante K (80cm): {K1}\n")

#SChwebungsdauern TS
TS1=(M1_p*M1_m)/(M1_p-M1_m) #über Mittelwert
TS2=(M2_p*M2_m)/(M2_p-M2_m)
print(f"Schwebungsdauer\nTS (72cm, berechnet): {TS1}\nTS (80cm, berechnet): {TS2}")
print(f"TS (72cm, gemessen): {M1_schwe}\nTS (80cm, gemessen): {M2_schwe}")
