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
d=0.430/mm_m #m aus zeichung    
L0_z=58/mm_m  #pm2  mm (aus Zeichnung, anstreben)
L1_z=105/mm_m #mm (aus Zeichnung)
L2_z=142/mm_m #mm (aus Zeichnung)
L0A=[L1_z,L2_z]
xL0A=L0A*mm_m


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






#Plots:
#Feder 1-3, Dicke wurde varriert
def rate1(L,R,F):
    return R*L+F

def rate2(L,R):
    return R*L

paramsR1, covR1 = curve_fit(rate1,xL0A,FA1)
errorsR1 = np.sqrt(np.diag(covR1))
unparamsR1 = unp.uarray(paramsR1,errorsR1) 
print(f"POLYFIT, RATE R1 CURVE FIT R verwendet:{unparamsR1[0]}, F0:{unparamsR1[1]}")

paramsR12, covR12 = curve_fit(rate2,xL0A,FA1)#           PROBLEMATIK: R beschrieben für Feder1
#paramsR22, covR22 = curve_fit(rate2,L0A,FA2)#           (1) Bestimmung mit rate1: R=0.08 aber F=-4.2, was ist hierbei F?
#paramsR32, covR32 = curve_fit(rate2,L0A,FA3)#           (2) Bestimmung mit rate2: R=0.05
#paramsR42, covR42 = curve_fit(rate2,L0A,FA4)#               Schnöring R=0.078   Voher dieser Wert? Klassisch F/s=R=0.05
#paramsR52, covR52 = curve_fit(rate2,L0A,FA5)#           --> Theoriekruven passen aber besser auf (2)


errorsR12 = np.sqrt(np.diag(covR12))
unparamsR12 = unp.uarray(paramsR12,errorsR12) 
print(f"RATE R2 CURVE FIT R:{unparamsR12[0]}")

#DISKUSSIONSPLOT - Schnittpunkt und Nullstellen:
R1,F1_0= polyfit(xL0A,FA1,1)
R2,F2_0= polyfit(xL0A,FA2,1)
R3,F3_0= polyfit(xL0A,FA3,1)
R4,F4_0= polyfit(L0A,FA4,1)
R5,F5_0= polyfit(L0A,FA5,1)


F0_true=[F1_0+L1_0_mittelwert*R1,F2_0+L2_0_mittelwert*R2,F3_0+L3_0_mittelwert*R3,F4_0+L4_0_mittelwert*R4,F5_0+L5_0_mittelwert*R5]
print(f"F_0 TRUE:{F0_true}")

#R mit dF und dL
#--------------------------
R1=(F11_mittelwert-F12_mittelwert)/(L1_z-L2_z)
R2=(F21_mittelwert-F22_mittelwert)/(L1_z-L2_z)
R3=(F31_mittelwert-F32_mittelwert)/(L1_z-L2_z)
R4=(F41_mittelwert-F42_mittelwert)/(L1_z-L2_z)
R5=(F51_mittelwert-F52_mittelwert)/(L1_z-L2_z)

Fx_0=[F1_0,F2_0,F3_0,F4_0,F5_0]
R=[R1,R2,R3,R4,R5]

xS_1=(-F2_0+F3_0)/(R2-R3)
xS_2=(-F4_0+F5_0)/(R4-R5)

xN=[-Fx_0[0]/R[0],-Fx_0[3]/R[3],-Fx_0[4]/R[4],]
print(f"xS_1:f({xS_1})={R2*xS_1+F2_0} --> BASIS: f({xS_1})={unparamsR1[0]*xS_1+unparamsR1[1]}")
print(f"xS_2:f({xS_2})={R2*xS_2+F2_0} --> BASIS: f({xS_2})={unparamsR1[0]*xS_2+unparamsR1[1]}")




#Feder1
xL0A=[L1_z*mm_m,L2_z*mm_m]
#R1=paramsR12[0]#
#F1_0=0#
x=np.linspace(L1_0_mittelwert*mm_m,L2_z*mm_m)
#x=np.linspace(xS_2,L2_z*mm_m)#dis
plt.errorbar(xL0A,FA1,yerr=F1_err,fmt="ok",label=f"Feder 1 Basis \n(D:{round(D1_mittelwert*mm_m,2)}$\;$mm, L:{round(L1_0_mittelwert*mm_m,1)}$\;$mm)")
plt.plot(x,R1*x+F1_0,"--k")

#Feder2
#R2=paramsR22[0]#
#F2_0=0#
x=np.linspace(L2_0_mittelwert*mm_m,L2_z*mm_m)
#x=np.linspace(xS_1,L2_z*mm_m)#dis
plt.errorbar(xL0A,FA2,yerr=F2_err,fmt="og",label=f"Feder 2 \n(D:{round(D2_mittelwert*mm_m,1)}$\;$mm, L:konst)")
plt.plot(x,R2*x+F2_0,"--g")

#Feder3
#R3=paramsR32[0]#
#F3_0=0#
x=np.linspace(L3_0_mittelwert*mm_m,L2_z*mm_m)
#x=np.linspace(xS_1,L2_z*mm_m)#dis
plt.errorbar(xL0A,FA3,yerr=F3_err,fmt="ob",label=f"Feder 3 \n(D:{round(D3_mittelwert*mm_m,1)}$\;$mm, L:konst)")
plt.plot(x,R3*x+F3_0,"--b")

plt.xlabel("Federweg $L\;/\;$mm")
plt.ylabel("Federkraft $F\;/\;$N")
plt.legend()
plt.savefig("build/D_kraftweg_dia.pdf")
plt.show()
plt.close()

#True PLOT FOR ds
#---------------------------------------------
#1
xL0A=[L1_z*mm_m-L1_0_mittelwert,L2_z*mm_m-L1_0_mittelwert]
x=np.linspace(0,xL0A[1])
plt.plot(x,R1*x+F0_true[0],"--k")
plt.plot(0,F0_true[0],"ok",label=f"Feder 1 Basis \n(D:{round(D1_mittelwert*mm_m,2)}$\;$mm, L:{round(L1_0_mittelwert*mm_m,1)}$\;$mm)")
plt.plot(xL0A,[F11_mittelwert,F12_mittelwert],"ok")
#2
xL0A=[L1_z*mm_m-L2_0_mittelwert,L2_z*mm_m-L2_0_mittelwert]
x=np.linspace(0,xL0A[1])
plt.plot(x,R2*x+F0_true[1],"--g")
plt.plot(0,F0_true[1],"og",label=f"Feder 2 \n(D:{round(D2_mittelwert*mm_m,1)}$\;$mm, L:konst)")
plt.plot(xL0A,[F21_mittelwert,F22_mittelwert],"og")
#3
xL0A=[L1_z*mm_m-L3_0_mittelwert,L2_z*mm_m-L3_0_mittelwert]
x=np.linspace(0,xL0A[1])
plt.plot(x,R3*x+F0_true[2],"--b")
plt.plot(0,F0_true[2],"ob",label=f"Feder 3 \n(D:{round(D3_mittelwert*mm_m,1)}$\;$mm, L:konst)")
plt.plot(xL0A,[F31_mittelwert,F32_mittelwert],"ob")


plt.xlabel("Federweg $\Delta L\;/\;$mm")
plt.ylabel("Federkraft $F\;/\;$N")
plt.legend()
plt.savefig("build/f0_123_dia.pdf")
plt.close()



#Feder 1,4,5,Widungung (Längen) varriert
#print Feder1 again
x=np.linspace(L1_0_mittelwert,L2_z)
#x=np.linspace(xN[0],L2_z)#dis
plt.errorbar(L0A,FA1,yerr=F1_err,fmt="ok",label=f"Feder 1 Basis \n(D:{round(D1_mittelwert*mm_m,2)}$\;$mm, L:{round(L1_0_mittelwert*mm_m,1)}$\;$mm)")
plt.plot(x,R1*x+F1_0,"--k")

#Feder4
#R4=paramsR42[0]#
#F4_0=0#
x=np.linspace(L4_0_mittelwert,L2_z)
#x=np.linspace(xS_2,L2_z)#dis
plt.errorbar(L0A,FA4,yerr=F4_err,fmt="oc",label=f"Feder 4 \n(D:konst, L:{round(L4_0_mittelwert*mm_m,1)}$\;$mm)")
plt.plot(x,R4*x+F4_0,"--c")

#Feder5
#R5=paramsR52[0]#
#F5_0=0#
x=np.linspace(L5_0_mittelwert,L2_z)
#x=np.linspace(xS_2,L2_z)#dis
plt.errorbar(L0A,FA5,yerr=F5_err,fmt="om",label=f"Feder 5 \n(D:konst, L:{round(L5_0_mittelwert*mm_m,1)}$\;$mm)")
plt.plot(x,R5*x+F5_0,"--m")

#Schnittpunkte einzeichnen:
#plt.plot(xS_1,R2*xS_1+F2_0,">r",label="Schnittpunkt 1,2,3")#dis
#plt.plot(xS_2,R4*xS_2+F4_0,"<r",label="Schnittpunkt 1,4,5")#dis


plt.xlabel("Federweg $L\;/\;$m")
plt.ylabel("Federkraft $F\;/\;$N")
#ax = plt.subplot(111)#dis
#box = ax.get_position()
#plt.legend(loc='center left', bbox_to_anchor=(1,0.5))#dis
plt.legend()
plt.savefig("build/n_kraftweg_dia.pdf")
#plt.tight_layout()
#plt.savefig("build/diss_kraft_dia.pdf")#dis
plt.show()
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
    $M$ & {MA[0]} & {MA[1]} & {MA[2]} & {MA[3]} & {MA[4]} \\\\
    $D_a$ & {D1[0]} & {D2[0]} & {D3[0]} & {D4[0]} & {D5[0]} \\\\
          & {D1[1]} & {D2[1]} & {D3[1]} & {D4[1]} & {D5[1]} \\\\
          & {D1[2]} & {D2[2]} & {D3[2]} & {D4[2]} & {D5[2]} \\\\
          & {D1[3]} & {D2[3]} & {D3[3]} & {D4[3]} & {D5[3]} \\\\
          & {D1[4]} & {D2[4]} & {D3[4]} & {D4[4]} & {D5[4]} \\\\
          & {D1[5]} &         &         &         &         \\\\
    $bar(D_a)$ & {D1_mittelwert} & {D2_mittelwert} & {D3_mittelwert} & {D4_mittelwert} & {D5_mittelwert}\\\\
    $D_(a_S)$ {D1_standabw} $ {D2_standabw} $ {D3_standabw} $ {D4_standabw} $ {D5_standabw}\\\\
    $L_a$ & {L1_0[0]} & {L2_0[0]} & {L3_0[0]} & {L4_0[0]} & {L5_0[0]} \\\\
          & {L1_0[1]} & {L2_0[1]} & {L3_0[1]} & {L4_0[1]} & {L5_0[1]} \\\\
          & {L1_0[2]} & {L2_0[2]} & {L3_0[2]} & {L4_0[2]} & {L5_0[2]} \\\\
          & {L1_0[3]} & {L2_0[3]} & {L3_0[3]} & {L4_0[3]} & {L5_0[3]} \\\\
          & {L1_0[4]} & {L2_0[4]} & {L3_0[4]} & {L4_0[4]} & {L5_0[4]} \\\\
          & {L1_0[5]} &         &         &         &         \\\\
    $bar(L_a)$ & {L1_0_mittelwert} & {L2_0_mittelwert} & {L3_0_mittelwert} & {L4_0_mittelwert} & {L5_0_mittelwert}\\\\
    $L_(a_S)& {L1_standabw} & {L2_standabw} & {L3_standabw} & {L4_standabw} & {L5_standabw}\\\\
    $F1$ bei $L1={L1_z}$ & {F11[0]} & {F21[0]} & {F31[0]} & {F41[0]} & {F51[0]}\\\\
                         & {F11[1]} & {F21[1]} & {F31[1]} & {F41[1]} & {F51[1]}\\\\
                         & {F11[2]} & {F21[2]} & {F31[2]} & {F41[2]} & {F51[2]}\\\\
                         & {F11[3]} & {F21[3]} & {F31[3]} & {F41[3]} & {F51[3]}\\\\
                         & {F11[4]} & {F21[4]} & {F31[4]} & {F41[4]} & {F51[4]}\\\\
    $bar(F1)$ & {F11_mittelwert} & {F21_mittelwert} & {F31_mittelwert} & {F41_mittelwert} & {F51_mittelwert}\\\\
    $F2$ bei $L2={L2_z}$ & {F12[0]} & {F22[0]} & {F32[0]} & {F42[0]} & {F52[0]}\\\\
                         & {F12[1]} & {F22[1]} & {F32[1]} & {F42[1]} & {F52[1]}\\\\
                         & {F12[2]} & {F22[2]} & {F32[2]} & {F42[2]} & {F52[2]}\\\\
                         & {F12[3]} & {F22[3]} & {F32[3]} & {F42[3]} & {F52[3]}\\\\
                         & {F12[4]} & {F22[4]} & {F32[4]} & {F42[4]} & {F52[4]}\\\\
    $bar(F1)$ & {F12_mittelwert} & {F22_mittelwert} & {F32_mittelwert} & {F42_mittelwert} & {F52_mittelwert}\\\\
    $R$ & {R1} & {R2} & {R3} & {R4} & {R5} \\\\
    $F_0$ & {F1_0} & {F2_0} & {F3_0} & {F4_0} & {F5_0} \\\\
""")


#Varriere Federdicke D
#plot Federkonstante-Federdicke Diagramm für Feder 1-3
xd=[D1_mittelwert*mm_m,D2_mittelwert*mm_m,D3_mittelwert*mm_m,D4_mittelwert*mm_m,D5_mittelwert*mm_m]
xm=MA
y=[R1/mm_m,R2/mm_m,R3/mm_m,R4/mm_m,R5/mm_m]
Rn=[R1/mm_m,R4/mm_m,R5/mm_m]
Rn=Rn
x_run=np.linspace(D2_mittelwert-0.5,D3_mittelwert+0.5,1000)
def funkt_D(x,k):
    return k*(1/(x**3)) 

def theorie_D(D):
    n=L1_0_mittelwert/d
    return (G*d**4)/(8*n*D**3)

paramsD, covD = curve_fit(funkt_D,xd[0:3],y[0:3])
errorsD = np.sqrt(np.diag(covD))
unparamsD = unp.uarray(paramsD,errorsD)

plt.plot(x_run,theorie_D(x_run),"--r",label="Theoriekurve nach (1)")
plt.plot(x_run,funkt_D(x_run,*paramsD),"--",label="Fit mit D^(-3)")
plt.plot(xd[0],y[0],"ok",label=f"Feder 1 Basis \n(D:{round(D1_mittelwert,2)}$\;$mm, L:{round(L1_0_mittelwert,1)}$\;$mm)")
plt.plot(xd[1],y[1],"o",label=f"Feder 2 \n(D:{round(D2_mittelwert,1)}$\;$mm, L:konst)")
plt.plot(xd[2],y[2],"o",label=f"Feder 3 \n(D:{round(D3_mittelwert,1)}$\;$mm, L:konst)")
plt.xlabel("Federdicke $D_a\;/\;$mm")
plt.ylabel("Federkonstante $R\;/\;$ N/mm")
plt.legend()
plt.savefig("build/dicke_konstante_dia.pdf")
plt.close()
print(f"params D^(-3) k_D: {unparamsD}")

#Varriere Federwindungszahl n
#plot Federkonstante-Windungszahl Diagramm Für Feder 1,4,5
n=[L1_0_mittelwert/(d*mm_m),L4_0_mittelwert/(d*mm_m),L5_0_mittelwert/(d*mm_m)]
xn_run=np.linspace(n[2]-20,n[1]+20)
#Für n wird erstmal Lx_0/d angenommen                                    FALSCH

def funkt_n(n,k):                                           #mit +b wäre besser
    return k*1/n
def theorie_n(n):
    D=3.68/mm_m#mm
    return (G*d**4)/(8*D**3*n)

paramsn, covn = curve_fit(funkt_n,n,Rn)
errorsn = np.sqrt(np.diag(covn))
unparamsn = unp.uarray(paramsn,errorsn)
print(f"Params 1/n k_n={unparamsn}")

plt.plot(xn_run,theorie_n(xn_run),"--r",label="Theoriekurve nach (1)")
plt.plot(xn_run,funkt_n(xn_run,*paramsn),"--",label="Fit mit 1/n")
plt.plot(n[0],Rn[0],"ok",label=f"Feder 1 Basis \n(D:{round(D1_mittelwert,2)}mm ,L:{round(L1_0_mittelwert,1)}mm)")
plt.plot(n[1],Rn[1],"o",label=f"Feder 4 \n(D:konst,L:{round(L4_0_mittelwert,1)}mm)")
plt.plot(n[2],Rn[2],"o",label=f"Feder 5 \n(D:konst,L:{round(L5_0_mittelwert,1)}mm)")
plt.xlabel("Windungszahl $n$")
plt.ylabel("Federkonstante $R\;/\;$N/m")
plt.legend()
plt.savefig("build/n_konstante_dia.pdf")
plt.close()
#Massen gegenüber Federkonstante
plt.plot(xm[0],y[0],"ok",label=f"Feder 1 Basis \n(D:{round(D1_mittelwert,2)}$\;$mm, L:{round(L1_0_mittelwert,1)}$\;$mm)")
plt.plot(xm[1],y[1],"o",label=f"Feder 2 \n(D:{round(D2_mittelwert,1)}$\;$mm, L:konst)")
plt.plot(xm[2],y[2],"o",label=f"Feder 3 \n(D:{round(D3_mittelwert,1)}$\;$mm, L:konst)")
plt.plot(xm[3],y[3],"o",label=f"Feder 4 \n(D:konst, L:{round(L4_0_mittelwert,1)}$\;$mm)")
plt.plot(xm[4],y[4],"o",label=f"Feder 5 \n(D:konst, L:{round(L5_0_mittelwert,1)}$\;$mm)")
plt.xlabel("Federmasse $m\;/\;g$")
plt.ylabel("Federkonstante $R\;/\;$ N/m")
plt.legend()
plt.savefig("build/masse_konstante_dia.pdf")
plt.close()



