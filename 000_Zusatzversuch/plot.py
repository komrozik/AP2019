import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import * 

def standabw(F):
    return None


L0_z=58  #pm2  mm (aus Zeichnung, anstreben)
L1_z=105 #mm (aus Zeichnung)
L2_z=142 #mm (aus Zeichnung)
LA=[L1_z,L2_z]

#Datenimport:
#-----------------------------------------------------
#Feder 1 (data1.1, data1.2)
D1,L1_0 = np.genfromtxt("data1.1.txt",unpack=True)
D1_mittelwert = sum(D1)/len(D1)
L1_0_mittelwert = sum(L1_0)/len(L1_0)

F12,F11 = np.genfromtxt("data1.2.txt",unpack = True)
F11_mittelwert = sum(F11)/len(F11)
F12_mittelwert = sum(F12)/len(F12)
F1_err= [standabw(F11),standabw(F12)]
F1_err=[0,0]
FA1=[F11_mittelwert,F12_mittelwert]

#Feder 2 (data2.1, data2.2)
D2,L2_0 = np.genfromtxt("data2.1.txt",unpack=True)
D2_mittelwert = sum(D2)/len(D2)
L2_0_mittelwert = sum(L2_0)/len(L2_0)

F22,F21 = np.genfromtxt("data2.2.txt",unpack = True)
F21_mittelwert = sum(F21)/len(F21)
F22_mittelwert = sum(F22)/len(F22)
F2_err= [standabw(F21),standabw(F22)]
F2_err=[0,0]
FA2=[F21_mittelwert,F22_mittelwert]

#Feder 3 (data3.1, data3.2)
D3,L3_0 = np.genfromtxt("data3.1.txt",unpack=True)
D3_mittelwert = sum(D3)/len(D3)
L3_0_mittelwert = sum(L3_0)/len(L3_0)

F32,F31 = np.genfromtxt("data3.2.txt",unpack = True)
F31_mittelwert = sum(F31)/len(F31)
F32_mittelwert = sum(F32)/len(F32)
F3_err= [standabw(F31),standabw(F32)]
F3_err=[0,0]
FA3=[F31_mittelwert,F32_mittelwert]

#Feder 4 (data4.1, data4.2)
D4,L4_0 = np.genfromtxt("data4.1.txt",unpack=True)
D4_mittelwert = sum(D4)/len(D4)
L4_0_mittelwert = sum(L4_0)/len(L4_0)

F42,F41 = np.genfromtxt("data4.2.txt",unpack = True)
F41_mittelwert = sum(F41)/len(F41)
F42_mittelwert = sum(F42)/len(F42)
F4_err= [standabw(F41),standabw(F42)]
F4_err=[0,0]
FA4=[F41_mittelwert,F42_mittelwert]

#Feder 5 (data5.1, data5.2)
D5,L5_0 = np.genfromtxt("data5.1.txt",unpack=True)
D5_mittelwert = sum(D5)/len(D5)
L5_0_mittelwert = sum(L5_0)/len(L5_0)

F52,F51 = np.genfromtxt("data5.2.txt",unpack = True)
F51_mittelwert = sum(F51)/len(F51)
F52_mittelwert = sum(F52)/len(F52)
F5_err= [standabw(F51),standabw(F52)]
F5_err=[0,0]
FA5=[F51_mittelwert,F52_mittelwert]
#-----------------------------------------------------






#Plots:
#Feder 1-3, Dicke wurde varriert

#Feder1
R1,F1_0= polyfit(LA,FA1,1)

x=np.linspace(L1_0_mittelwert,L2_z)
plt.errorbar(LA,FA1,yerr=F1_err,fmt="ok",label=f"Feder 1 (Basis), D={D1_mittelwert}")
plt.plot(x,R1*x+F1_0,"--k")

#Feder2
R2,F2_0= polyfit(LA,FA2,1)

x=np.linspace(L2_0_mittelwert,L2_z)
plt.errorbar(LA,FA2,yerr=F2_err,fmt="og",label=f"Feder 2, D={D2_mittelwert}")
plt.plot(x,R2*x+F2_0,"--g")

#Feder3
R3,F3_0= polyfit(LA,FA3,1)

x=np.linspace(L3_0_mittelwert,L2_z)
plt.errorbar(LA,FA3,yerr=F3_err,fmt="ob",label=f"Feder 3, D={D3_mittelwert}")
plt.plot(x,R3*x+F3_0,"--b")


plt.xlabel("Federweg $L\;/\;$mm")
plt.ylabel("Federkraft $F\;/\;$N")
plt.legend()
plt.savefig("build/D_kraftweg_dia.pdf")
plt.close()




#Feder 1,4,5,Widungung (Längen) varriert
#print Feder1 again
x=np.linspace(L1_0_mittelwert,L2_z)
plt.errorbar(LA,FA1,yerr=F1_err,fmt="ok",label=f"Feder 1 (Basis), L={L1_0_mittelwert}")
plt.plot(x,R1*x+F1_0,"--k")

#Feder4
R4,F4_0= polyfit(LA,FA4,1)

x=np.linspace(L4_0_mittelwert,L2_z)
plt.errorbar(LA,FA4,yerr=F4_err,fmt="oc",label=f"Feder 4, L={L4_0_mittelwert}")
plt.plot(x,R4*x+F4_0,"--c")

#Feder5
R5,F5_0= polyfit(LA,FA5,1)

x=np.linspace(L5_0_mittelwert,L2_z)
plt.errorbar(LA,FA5,yerr=F5_err,fmt="om",label=f"Feder 5, L={L5_0_mittelwert}")
plt.plot(x,R5*x+F5_0,"--m")



plt.xlabel("Federweg $L\;/\;$mm")
plt.ylabel("Federkraft $F\;/\;$N")
plt.legend()
plt.savefig("build/n_kraftweg_dia.pdf")
plt.close()


#federkonstanten R
R=[R1,R2,R3,R4,R5]
print(f"""
    R1: {R1}, F1_0: {F1_0}
    R2: {R2}, F2_0: {F2_0}
    R3: {R3}, F3_0: {F3_0}
    R4: {R4}, F4_0: {F4_0}
    R5: {R5}, F5_0: {F5_0}
""")
#ausgabe Tabel
print(D1[5])
print(f"""
    $D_a$ & {D1[0]} & {D2[0]} & {D3[0]} & {D4[0]} & {D5[0]} \\\\
          & {D1[1]} & {D2[1]} & {D3[1]} & {D4[1]} & {D5[1]} \\\\
          & {D1[2]} & {D2[2]} & {D3[2]} & {D4[2]} & {D5[2]} \\\\
          & {D1[3]} & {D2[3]} & {D3[3]} & {D4[3]} & {D5[3]} \\\\
          & {D1[4]} & {D2[4]} & {D3[4]} & {D4[4]} & {D5[4]} \\\\
          & {D1[5]} &         &         &         &         \\\\
    $D_(mittelwert)$ & {D1_mittelwert} & {D2_mittelwert} & {D3_mittelwert} & {D4_mittelwert} & {D5_mittelwert}\\\\
    $L_a$ & {L1_0[0]} & {L2_0[0]} & {L3_0[0]} & {L4_0[0]} & {L5_0[0]} \\\\
          & {L1_0[1]} & {L2_0[1]} & {L3_0[1]} & {L4_0[1]} & {L5_0[1]} \\\\
          & {L1_0[2]} & {L2_0[2]} & {L3_0[2]} & {L4_0[2]} & {L5_0[2]} \\\\
          & {L1_0[3]} & {L2_0[3]} & {L3_0[3]} & {L4_0[3]} & {L5_0[3]} \\\\
          & {L1_0[4]} & {L2_0[4]} & {L3_0[4]} & {L4_0[4]} & {L5_0[4]} \\\\
          & {L1_0[5]} &         &         &         &         \\\\
    $L_(mittelwert)$ & {L1_0_mittelwert} & {L2_0_mittelwert} & {L3_0_mittelwert} & {L4_0_mittelwert} & {L5_0_mittelwert}\\\\
    $R$ & {R1} & {R2} & {R3} & {R4} & {R5} \\\\
    $F_0$ & {F1_0} & {F2_0} & {F3_0} & {F4_0} & {F5_0} \\\\
""")




