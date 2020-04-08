#Erklärung unter:    git log 148ceab17919e81230b12114c35c81f549f60c7a

import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import * 

def standabw(x,xm):
    return np.sqrt(sum((x-xm)**2)/len(x))
mm_m=1
G=71000*mm_m**2 #N/m^2
d=0.430/mm_m #m aus zeichung 0.43425optimum    
L0_z=58/mm_m  #pm2  mm (aus Zeichnung, anstreben)
L1_z=105/mm_m #mm (aus Zeichnung)
L2_z=142/mm_m #mm (aus Zeichnung)
L0A=[L1_z,L2_z]
xL0A=L0A*mm_m

#c FAKE-Wert abzug an Wicklungen damit Theorie auf Kurve passt...
c=11.755 #ca. LH


#Datenimport:
#-----------------------------------------------------
#Federmassen
M1, M2, M3, M4, M5 =np.genfromtxt("data_m.txt",unpack=True) #g
MA = [x / 5 for x in [M1,M2,M3,M4,M5]] #g


#Feder 1 (data1.1, data1.2) D=3.68
D1,L1_0 = np.genfromtxt("data1.1.txt",unpack=True)
D1_mittelwert = sum(D1)/len(D1)/mm_m
D1_standabw = standabw(D1,D1_mittelwert)/mm_m
L1_0_mittelwert = sum(L1_0)/len(L1_0)/mm_m
L1_standabw = standabw(L1_0,L1_0_mittelwert)/mm_m

F12,F11 = np.genfromtxt("data1.2.txt",unpack = True)
F11_mittelwert = sum(F11)/len(F11)
F12_mittelwert = sum(F12)/len(F12)
F1_err= [standabw(F11,F11_mittelwert),standabw(F12,F12_mittelwert)]
#F1_err=[0,0]
FA1=[F11_mittelwert,F12_mittelwert]

#Feder 2 (data2.1, data2.2)
D2,L2_0 = np.genfromtxt("data2.1.txt",unpack=True)
D2_mittelwert = sum(D2)/len(D2)/mm_m
D2_standabw = standabw(D2,D2_mittelwert)/mm_m
L2_0_mittelwert = sum(L2_0)/len(L2_0)/mm_m
L2_standabw = standabw(L2_0,L2_0_mittelwert)/mm_m

F22,F21 = np.genfromtxt("data2.2.txt",unpack = True)
F21_mittelwert = sum(F21)/len(F21)
F22_mittelwert = sum(F22)/len(F22)
F2_err= [standabw(F21,F21_mittelwert),standabw(F22,F22_mittelwert)]
#F2_err= [0,0]
FA2=[F21_mittelwert,F22_mittelwert]

#Feder 3 (data3.1, data3.2)
D3,L3_0 = np.genfromtxt("data3.1.txt",unpack=True)
D3_mittelwert = sum(D3)/len(D3)/mm_m
D3_standabw = standabw(D3,D3_mittelwert)/mm_m
L3_0_mittelwert = sum(L3_0)/len(L3_0)/mm_m
L3_standabw = standabw(L3_0,L3_0_mittelwert)/mm_m

F32,F31 = np.genfromtxt("data3.2.txt",unpack = True)
F31_mittelwert = sum(F31)/len(F31)
F32_mittelwert = sum(F32)/len(F32)
F3_err= [standabw(F31,F31_mittelwert),standabw(F32,F32_mittelwert)]
#F3_err=[0,0]
FA3=[F31_mittelwert,F32_mittelwert]

#Feder 4 (data4.1, data4.2)
D4,L4_0 = np.genfromtxt("data4.1.txt",unpack=True)
D4_mittelwert = sum(D4)/len(D4)/mm_m
D4_standabw = standabw(D4,D4_mittelwert)/mm_m
L4_0_mittelwert = sum(L4_0)/len(L4_0)/mm_m
L4_standabw = standabw(L4_0,L4_0_mittelwert)/mm_m

F42,F41 = np.genfromtxt("data4.2.txt",unpack = True)
F41_mittelwert = sum(F41)/len(F41)
F42_mittelwert = sum(F42)/len(F42)
F4_err= [standabw(F41,F41_mittelwert),standabw(F42,F42_mittelwert)]
#F4_err=[0,0]
FA4=[F41_mittelwert,F42_mittelwert]

#Feder 5 (data5.1, data5.2)
D5,L5_0 = np.genfromtxt("data5.1.txt",unpack=True)
D5_mittelwert = sum(D5)/len(D5)/mm_m
D5_standabw = standabw(D5,D5_mittelwert)/mm_m
L5_0_mittelwert = sum(L5_0)/len(L5_0)/mm_m
L5_standabw = standabw(L5_0,L5_0_mittelwert)/mm_m

F52,F51 = np.genfromtxt("data5.2.txt",unpack = True)
F51_mittelwert = sum(F51)/len(F51)
F52_mittelwert = sum(F52)/len(F52)
F5_err= [standabw(F51,F51_mittelwert),standabw(F52,F52_mittelwert)]
#F5_err=[0,0]
FA5=[F51_mittelwert,F52_mittelwert]
#-----------------------------------------------------

n=[(L1_0_mittelwert-c)/(d*mm_m),(L4_0_mittelwert-c)/(d*mm_m),(L5_0_mittelwert-c)/(d*mm_m)]





#Plots:
#Feder 1-3, Dicke wurde varriert
def rate1(L,R,F):
    return R*(L-L1_0_mittelwert)+F


paramsR1, covR1 = curve_fit(rate1,xL0A,FA1)
errorsR1 = np.sqrt(np.diag(covR1))
unparamsR1 = unp.uarray(paramsR1,errorsR1) 

paramsR2, covR2 = curve_fit(rate1,L0A,FA2)            
paramsR3, covR3 = curve_fit(rate1,L0A,FA3)           
paramsR4, covR4 = curve_fit(rate1,L0A,FA4)               
paramsR5, covR5 = curve_fit(rate1,L0A,FA5)           

print(f"RATE R1 CURVE FIT R verwendet:{unparamsR1[0]}, F0:{unparamsR1[1]}")

#DISKUSSIONSPLOT - Schnittpunkt und Nullstellen:
R1,F1_0= polyfit(xL0A,FA1,1)
R2,F2_0= polyfit(xL0A,FA2,1)
R3,F3_0= polyfit(xL0A,FA3,1)
R4,F4_0= polyfit(L0A,FA4,1)
R5,F5_0= polyfit(L0A,FA5,1)


F0_true=[F1_0+L1_0_mittelwert*R1,F2_0+L2_0_mittelwert*R2,F3_0+L3_0_mittelwert*R3,F4_0+L4_0_mittelwert*R4,F5_0+L5_0_mittelwert*R5]

#R mit dF und dL
#--------------------------
#R1=(F11_mittelwert-F12_mittelwert)/(L1_z-L2_z)
#R2=(F21_mittelwert-F22_mittelwert)/(L1_z-L2_z)
#R3=(F31_mittelwert-F32_mittelwert)/(L1_z-L2_z)
#R4=(F41_mittelwert-F42_mittelwert)/(L1_z-L2_z)
#R5=(F51_mittelwert-F52_mittelwert)/(L1_z-L2_z)

Fx_0=[F1_0,F2_0,F3_0,F4_0,F5_0]
R=[R1,R2,R3,R4,R5]

#Schnittpunkt xS für diskussionsgraphic
xS_1=(-F2_0+F3_0)/(R2-R3)
xS_2=(-F4_0+F5_0)/(R4-R5)

xN=[-Fx_0[0]/R[0],-Fx_0[3]/R[3],-Fx_0[4]/R[4],]




#Feder1
xL0A=[L1_z*mm_m,L2_z*mm_m]
#R1=paramsR12[0]#
#F1_0=0#
x=np.linspace(L1_0_mittelwert*mm_m,L2_z*mm_m)
#x=np.linspace(xS_2,L2_z*mm_m)#dis
plt.errorbar(xL0A,FA1,yerr=F1_err,fmt="ok",label=f"Feder 1 Basis \n(D:{round(D1_mittelwert*mm_m,2)}$\;$mm, n:{round(n[0],1)}$\;$)")
plt.plot(x,R1*x+F1_0,"--k")

#Feder2
#R2=paramsR22[0]#
#F2_0=0#
x=np.linspace(L2_0_mittelwert*mm_m,L2_z*mm_m)
#x=np.linspace(xS_1,L2_z*mm_m)#dis
plt.errorbar(xL0A,FA2,yerr=F2_err,fmt="og",label=f"Feder 2 \n(D:{round(D2_mittelwert*mm_m,1)}$\;$mm, n:konst)")
plt.plot(x,R2*x+F2_0,"--g")

#Feder3
#R3=paramsR32[0]#
#F3_0=0#
x=np.linspace(L3_0_mittelwert*mm_m,L2_z*mm_m)
#x=np.linspace(xS_1,L2_z*mm_m)#dis
plt.errorbar(xL0A,FA3,yerr=F3_err,fmt="ob",label=f"Feder 3 \n(D:{round(D3_mittelwert*mm_m,1)}$\;$mm, n:konst)")
plt.plot(x,R3*x+F3_0,"--b")

plt.xlabel("Länge der belasteten Feder $L\;/\;$mm")
plt.ylabel("Federkraft $F\;/\;$N")
plt.legend()
plt.savefig("plots/D_kraftweg_dia.pdf")
plt.close()

#True PLOT FOR ds
#---------------------------------------------
#1
xL0A=[L1_z*mm_m-L1_0_mittelwert,L2_z*mm_m-L1_0_mittelwert]
x=np.linspace(0,xL0A[1])
plt.plot(x,R1*x+F0_true[0],"--k")
plt.plot(0,F0_true[0],"ok",label=f"Feder 1 Basis \n(D:{round(D1_mittelwert*mm_m,2)}$\;$mm, n:{round(n[0],1)}$\;$)")
plt.plot(xL0A,[F11_mittelwert,F12_mittelwert],"ok")
#2
xL0A=[L1_z*mm_m-L2_0_mittelwert,L2_z*mm_m-L2_0_mittelwert]
x=np.linspace(0,xL0A[1])
plt.plot(x,R2*x+F0_true[1],"--g")
plt.plot(0,F0_true[1],"og",label=f"Feder 2 \n(D:{round(D2_mittelwert*mm_m,1)}$\;$mm, n:konst)")
plt.plot(xL0A,[F21_mittelwert,F22_mittelwert],"og")
#3
xL0A=[L1_z*mm_m-L3_0_mittelwert,L2_z*mm_m-L3_0_mittelwert]
x=np.linspace(0,xL0A[1])
plt.plot(x,R3*x+F0_true[2],"--b")
plt.plot(0,F0_true[2],"ob",label=f"Feder 3 \n(D:{round(D3_mittelwert*mm_m,1)}$\;$mm, n:konst)")
plt.plot(xL0A,[F31_mittelwert,F32_mittelwert],"ob")


plt.xlabel("Federweg $s\;/\;$mm")
plt.ylabel("Federkraft $F\;/\;$N")
plt.legend()
plt.savefig("plots/f0_123_dia.pdf")
plt.close()


#Feder 1,4,5,Widungung (Längen) varriert
#print Feder1 again
x=np.linspace(L1_0_mittelwert,L2_z)
#x=np.linspace(xN[0],L2_z)#dis
plt.errorbar(L0A,FA1,yerr=F1_err,fmt="ok",label=f"Feder 1 Basis \n(D:{round(D1_mittelwert*mm_m,2)}$\;$mm, n:{round(n[0],1)}$\;$)")
plt.plot(x,R1*x+F1_0,"--k")

#Feder4
#R4=paramsR42[0]#
#F4_0=0#
x=np.linspace(L4_0_mittelwert,L2_z)
#x=np.linspace(xS_2,L2_z)#dis
plt.errorbar(L0A,FA4,yerr=F4_err,fmt="oc",label=f"Feder 4 \n(D:konst, n:{round(n[1],1)}$\;$)")
plt.plot(x,R4*x+F4_0,"--c")

#Feder5
#R5=paramsR52[0]#
#F5_0=0#
x=np.linspace(L5_0_mittelwert,L2_z)
#x=np.linspace(xS_2,L2_z)#dis
plt.errorbar(L0A,FA5,yerr=F5_err,fmt="om",label=f"Feder 5 \n(D:konst, n:{round(n[2],1)}$\;$)")
plt.plot(x,R5*x+F5_0,"--m")

#Schnittpunkte einzeichnen:
#plt.plot(xS_1,R2*xS_1+F2_0,">r",label="Schnittpunkt 1,2,3")#dis
#plt.plot(xS_2,R4*xS_2+F4_0,"<r",label="Schnittpunkt 1,4,5")#dis


plt.xlabel("Länge der belasteten Feder $L\;/\;$mm")
plt.ylabel("Federkraft $F\;/\;$N")
#ax = plt.subplot(111)#dis
#box = ax.get_position()
#plt.legend(loc='center left', bbox_to_anchor=(1,0.5))#dis
plt.legend()
plt.savefig("plots/n_kraftweg_dia.pdf")
#plt.tight_layout()
#plt.savefig("plots/diss_kraft_dia.pdf")#dis
plt.close()


#federkonstanten R und F1_0 was dem Verschiebungswert v entspricht
R=[R1,R2,R3,R4,R5]
print(f"""
    R1: {R1}, F1_0: {F1_0}
    R2: {R2}, F2_0: {F2_0}
    R3: {R3}, F3_0: {F3_0}
    R4: {R4}, F4_0: {F4_0}
    R5: {R5}, F5_0: {F5_0}
""")
#innere Vorspannkraft
print(f"F_0 TRUE:{F0_true}")
#Schnittpunk überprüfung
print(f"xS_1:f({xS_1})={R2*xS_1+F2_0} --> BASIS: f({xS_1})={unparamsR1[0]*xS_1+unparamsR1[1]}")
print(f"xS_2:f({xS_2})={R2*xS_2+F2_0} --> BASIS: f({xS_2})={unparamsR1[0]*xS_2+unparamsR1[1]}")


#ausgabe Tabel
print(f"""
    $M$ \t& {MA[0]} \t& {MA[1]} \t& {MA[2]} \t& {MA[3]} \t& {MA[4]} \t\\\\
    $D_a$ \t& {D1[0]} \t& {D2[0]} \t& {D3[0]} \t& {D4[0]} \t& {D5[0]} \t\\\\
          \t& {D1[1]} \t& {D2[1]} \t& {D3[1]} \t& {D4[1]} \t& {D5[1]} \t\\\\
          \t& {D1[2]} \t& {D2[2]} \t& {D3[2]} \t& {D4[2]} \t& {D5[2]} \t\\\\
          \t& {D1[3]} \t& {D2[3]} \t& {D3[3]} \t& {D4[3]} \t& {D5[3]} \t\\\\
          \t& {D1[4]} \t& {D2[4]} \t& {D3[4]} \t& {D4[4]} \t& {D5[4]} \t\\\\
          \t& {D1[5]} \t&         \t&         \t&         \t&         \t\\\\
    $bar(D_a)$ \t& {D1_mittelwert} \t& {D2_mittelwert} \t& {D3_mittelwert} \t& {D4_mittelwert} \t& {D5_mittelwert}\t\\\\
    $D_(a_S)$ {D1_standabw} \t& {D2_standabw} \t& {D3_standabw} \t& {D4_standabw} \t& {D5_standabw}\t\\\\
    $L_a$ \t& {L1_0[0]} \t& {L2_0[0]} \t& {L3_0[0]} \t& {L4_0[0]} \t& {L5_0[0]} \t\\\\
          \t& {L1_0[1]} \t& {L2_0[1]} \t& {L3_0[1]} \t& {L4_0[1]} \t& {L5_0[1]} \t\\\\
          \t& {L1_0[2]} \t& {L2_0[2]} \t& {L3_0[2]} \t& {L4_0[2]} \t& {L5_0[2]} \t\\\\
          \t& {L1_0[3]} \t& {L2_0[3]} \t& {L3_0[3]} \t& {L4_0[3]} \t& {L5_0[3]} \t\\\\
          \t& {L1_0[4]} \t& {L2_0[4]} \t& {L3_0[4]} \t& {L4_0[4]} \t& {L5_0[4]} \t\\\\
          \t& {L1_0[5]} \t&         \t&         \t&         \t&         \t\\\\
    $bar(L_a)$ \t& {L1_0_mittelwert} \t& {L2_0_mittelwert} \t& {L3_0_mittelwert} \t& {L4_0_mittelwert} \t& {L5_0_mittelwert}\t\\\\
    $L_(a_S)\t& {L1_standabw} \t& {L2_standabw} \t& {L3_standabw} \t& {L4_standabw} \t& {L5_standabw}\t\\\\
    $F1$ bei $L1={L1_z}$ \t& {F11[0]} \t& {F21[0]} \t& {F31[0]} \t& {F41[0]} \t& {F51[0]}\t\\\\
                         \t& {F11[1]} \t& {F21[1]} \t& {F31[1]} \t& {F41[1]} \t& {F51[1]}\t\\\\
                         \t& {F11[2]} \t& {F21[2]} \t& {F31[2]} \t& {F41[2]} \t& {F51[2]}\t\\\\
                         \t& {F11[3]} \t& {F21[3]} \t& {F31[3]} \t& {F41[3]} \t& {F51[3]}\t\\\\
                         \t& {F11[4]} \t& {F21[4]} \t& {F31[4]} \t& {F41[4]} \t& {F51[4]}\t\\\\
    $bar(F1)$ \t& {F11_mittelwert} \t& {F21_mittelwert} \t& {F31_mittelwert} \t& {F41_mittelwert} \t& {F51_mittelwert}\t\\\\
    $F1_S$    \t& {F1_err[0]}      \t& {F2_err[0]}      \t& {F3_err[0]}      \t& {F4_err[0]}    \t& {F5_err[0]} \t\\\\
    $F2_S$    \t& {F1_err[1]}      \t& {F2_err[1]}      \t& {F3_err[1]}      \t& {F4_err[1]}    \t& {F5_err[1]} \t\\\\
    $F2$ bei $L2={L2_z}$ \t& {F12[0]} \t& {F22[0]} \t& {F32[0]} \t& {F42[0]} \t& {F52[0]}\t\\\\
                         \t& {F12[1]} \t& {F22[1]} \t& {F32[1]} \t& {F42[1]} \t& {F52[1]}\t\\\\
                         \t& {F12[2]} \t& {F22[2]} \t& {F32[2]} \t& {F42[2]} \t& {F52[2]}\t\\\\
                         \t& {F12[3]} \t& {F22[3]} \t& {F32[3]} \t& {F42[3]} \t& {F52[3]}\t\\\\
                         \t& {F12[4]} \t& {F22[4]} \t& {F32[4]} \t& {F42[4]} \t& {F52[4]}\t\\\\
    $bar(F1)$ \t& {F12_mittelwert} \t& {F22_mittelwert} \t& {F32_mittelwert} \t& {F42_mittelwert} \t& {F52_mittelwert}\\\\
    $R$ \t& {R1} \t& {R2} \t& {R3} \t& {R4} \t& {R5} \\\\
    $v0$ \t& {F1_0} \t& {F2_0} \t& {F3_0} \t& {F4_0} \t& {F5_0} \\\\
    $F0$ \t& {F0_true[0]} \t& {F0_true[1]} \t& {F0_true[2]} \t& {F0_true[3]} \t& {F0_true[4]} \\\\
""")


#Varriere Federdicke D
#plot Federkonstante-Federdicke Diagramm für Feder 1-3
xd=[D1_mittelwert*mm_m,D2_mittelwert*mm_m,D3_mittelwert*mm_m,D4_mittelwert*mm_m,D5_mittelwert*mm_m]
xm=MA
y=[R1/mm_m,R2/mm_m,R3/mm_m,R4/mm_m,R5/mm_m]
Rn=[R1/mm_m,R4/mm_m,R5/mm_m]
x_run=np.linspace(D2_mittelwert-0.5,D3_mittelwert+0.5,1000)
def funkt_D(x,k):
    return k*(1/(x-d)**3)

def theorie_D(D):
    n=(L1_0_mittelwert-c)/d
    print(L1_0_mittelwert/d)
    return (G*d**4)/(8*n*((D-d)**3))

paramsD, covD = curve_fit(funkt_D,xd[0:3],y[0:3])
errorsD = np.sqrt(np.diag(covD))
unparamsD = unp.uarray(paramsD,errorsD)


k=[G/(8*L1_0_mittelwert),G/(8*L2_0_mittelwert),G/(8*L3_0_mittelwert)]
print(k)

plt.plot(x_run,theorie_D(x_run),"--r",label="Theoriekurve nach (1)")
plt.plot(x_run,funkt_D(x_run,*paramsD),"--",label="Fit mit D^(-3)")
plt.plot(xd[0],R[0],"ok",label=f"Feder 1 Basis \n(D:{round(D1_mittelwert,2)}$\;$mm, n:{round(n[0],1)}$\;$)")
plt.plot(xd[1],R[1],"og",label=f"Feder 2 \n(D:{round(D2_mittelwert,1)}$\;$mm, n:konst)")
plt.plot(xd[2],R[2],"ob",label=f"Feder 3 \n(D:{round(D3_mittelwert,1)}$\;$mm, n:konst)")
plt.xlabel("Federdicke $D\;/\;$mm")
plt.ylabel("Federkonstante $R\;/\;$ N/mm")
plt.legend()
plt.savefig("plots/dicke_konstante_dia.pdf")
plt.close()
print(f"params D^(-3) k_D: {unparamsD}")

#Varriere Federwindungszahl n
#plot Federkonstante-Windungszahl Diagramm Für Feder 1,4,5
n=[(L1_0_mittelwert-c)/(d*mm_m),(L4_0_mittelwert-c)/(d*mm_m),(L5_0_mittelwert-c)/(d*mm_m)]
xn_run=np.linspace(n[2]-20,n[1]+20)
print(f"nwirk:{n},n2:{(L2_0_mittelwert-c)/(d*mm_m)},n3:{(L3_0_mittelwert-c)/(d*mm_m)}")
#Für n wird erstmal Lx_0/d angenommen                                    FALSCH

def funkt_n(n,k):                                           #mit +b wäre besser
    return k*1/n
def theorie_n(n):
    D=3.68-d/mm_m#mm
    return (G*d**4)/(8*D**3*n)

paramsn, covn = curve_fit(funkt_n,n,Rn)
errorsn = np.sqrt(np.diag(covn))
unparamsn = unp.uarray(paramsn,errorsn)
print(f"Params 1/n k_n={unparamsn}")

plt.plot(xn_run,theorie_n(xn_run),"--r",label="Theoriekurve nach (1)")
plt.plot(xn_run,funkt_n(xn_run,*paramsn),"--",label="Fit mit 1/n")
plt.plot(n[0],R[0],"ok",label=f"Feder 1 Basis \n(D:{round(D1_mittelwert,2)}mm ,n:{round(n[0],1)})")
plt.plot(n[1],R[3],"oc",label=f"Feder 4 \n(D:konst,n:{round(n[1],1)})")
plt.plot(n[2],R[4],"om",label=f"Feder 5 \n(D:konst,n:{round(n[2],1)})")
plt.xlabel("wirkende Windungszahl $n_{wirk}$")
plt.ylabel("Federkonstante $R\;/\;$N/mm")
plt.legend()
plt.savefig("plots/n_konstante_dia.pdf")
plt.close()
#Massen gegenüber Federkonstante
plt.plot(xm[0],y[0],"ok",label=f"Feder 1 Basis \n(D:{round(D1_mittelwert,2)}$\;$mm, n:{round(n[0],1)}$\;$)")
plt.plot(xm[1],y[1],"o",label=f"Feder 2 \n(D:{round(D2_mittelwert,1)}$\;$mm, n:konst)")
plt.plot(xm[2],y[2],"o",label=f"Feder 3 \n(D:{round(D3_mittelwert,1)}$\;$mm, n:konst)")
plt.plot(xm[3],y[3],"o",label=f"Feder 4 \n(D:konst, L:{round(n[1],1)}$\;$)")
plt.plot(xm[4],y[4],"o",label=f"Feder 5 \n(D:konst, L:{round(n[2],1)}$\;$)")
plt.xlabel("Federmasse $m\;/\;g$")
plt.ylabel("Federkonstante $R\;/\;$ N/m")
plt.legend()
plt.savefig("plots/masse_konstante_dia.pdf")
plt.close()

#Diskussion-Vergleich mit schnöring Federberechnung
D=3.700
nreal=110.714
L0=59.577
L1=105
L2=142
Ln=154.135
L=[L0,L1,L2,Ln]
F0=1.440
F1=5.000
F1err=0.250
F2=7.900
Fn=8.851
F2err=0.300
F=[F0,F1,F2]
R=0.078
x=np.linspace(0,Ln)


plt.plot(x,R*x+F0-R*L0,"k",label=f"Schnöring Federberechnung (Idealfeder) \n $D={D}, n={nreal}$")
#plt.plot(0,F0,"hk")
plt.errorbar(L1,F1,yerr=F1err,fmt="ok",label=f"Soll-Federkraft F1,F2 mit Abweichung\n$\Delta F1={F1err}, \Delta F2={F2err}$")
plt.errorbar(L2,F2,yerr=F2err,fmt="ok")

plt.plot(x,R1*x+F0_true[0]-R1*L1_0_mittelwert,"--b",label=f"Messung, Berechnung (Ist-Feder/Basisfeder) \n $D={round(D1_mittelwert,2)}, n={round(n[0],1)}$")
#plt.plot(0,F0_true[0],"ob")
plt.errorbar(L1_z,F11_mittelwert,yerr=0.021,fmt="ob",label="(gemessene) Ist-Federkraft F1,F2\n$\Delta F1=0.021, \Delta F2=0.03$")
plt.errorbar(L2_z,F12_mittelwert,yerr=0.03,fmt="ob")
plt.legend(loc="best")
plt.xlabel("Länge der belasteten Feder $L\;/\;$mm")
plt.ylabel("Federkraft $F\;/\;$N")


#D=3.600
#n=121.562
#L0=64.796
#L1=105
#L2=142
#Ln=157.697
#L=[L0,L1,L2,Ln]
#F0=1.849
#F1=5.000
#F1err=0.250
#F2=7.900
#Fn=9.130
#F2err=0.300
#F=[F0,F1,F2]
#R=0.078
#x=np.linspace(0,Ln-L0)


#plt.plot(x,R*x+F0,"b",label=f"Schnöring Federberechnung \n $D={D}, L={L0}$")
#plt.plot(0,F0,"hb")
#plt.errorbar(L1-L0,F1,yerr=F1err,fmt="hb")
#plt.errorbar(L2-L0,F2,yerr=F2err,fmt="hb")
#plt.plot(Ln-L0,Fn,"hb")

#plt.plot(x,R2*x+F0_true[1],"--b",label=f"Messung, Berechnung Feder 2 \n $D={round(D2_mittelwert,2)}, L={round(L2_0_mittelwert,2)}$")
#plt.plot(0,F0_true[1],"hb")
#plt.plot(L1_z-L2_0_mittelwert,F21_mittelwert,"hb")
#plt.plot(L2_z-L2_0_mittelwert,F22_mittelwert,"hb")
#plt.legend()
#plt.xlabel("Federweg $s\;/\;$mm")
#plt.ylabel("Federkraft $F\;/\;$N")
plt.xlim(L1-5,L2_z+5)
plt.ylim(F11_mittelwert-1,F12_mittelwert+1)
plt.savefig("plots/schnö_12_dia.pdf")
plt.close()



plt.plot(D1_mittelwert,M1/5,"ok",label="Feder 1")
plt.plot(D2_mittelwert,M2/5,"ob",label="Feder 2")
plt.plot(D3_mittelwert,M3/5,"oy",label="Feder 3")
plt.xlabel("Federausßendruchmesser $D_a\;/\;$mm")
plt.ylabel("Masse $m\;/\;$g")
plt.legend()
plt.savefig("plots/m_D_dia.pdf")
plt.close()

n=[(L1_0_mittelwert-c)/(d*mm_m),(L4_0_mittelwert-c)/(d*mm_m),(L5_0_mittelwert-c)/(d*mm_m)]
plt.plot(n[0],M1/5,"ok",label="Feder 1")
plt.plot(n[1],M4/5,"og",label="Feder 4")
plt.plot(n[2],M5/5,"or",label="Feder 5")
plt.xlabel("Wicklungzahl $n$")
plt.ylabel("Masse $m\;/\;$g")
plt.legend()
plt.savefig("plots/m_n_dia.pdf")
plt.close()



