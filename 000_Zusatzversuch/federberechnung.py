import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import * 

def standabw(x,xm):
    return np.sqrt(sum((x-xm)**2)/len(x))

#DATAIMPORT
M1, M2, M3, M4, M5 =np.genfromtxt("data_m.txt",unpack=True) #g

#Feder 1 (data1.1, data1.2) D=3.68
D1,L1_0 = np.genfromtxt("data1.1.txt",unpack=True)
D1_mittelwert = sum(D1)/len(D1)
D1_standabw = standabw(D1,D1_mittelwert)
L1_0_mittelwert = sum(L1_0)/len(L1_0)
L1_standabw = standabw(L1_0,L1_0_mittelwert)
F12,F11 = np.genfromtxt("data1.2.txt",unpack = True)
F11_mittelwert = sum(F11)/len(F11)
F12_mittelwert = sum(F12)/len(F12)
F1_err= [standabw(F11,F11_mittelwert),standabw(F12,F12_mittelwert)]
#F1_err=[0,0]
FA1=[F11_mittelwert,F12_mittelwert]

G=71000
d=0.430#opt
werkstoff="x10CrNi18-8  DIN EN 10270-3"
E= 185000 
p= 7.90
L1=105
L2=142

def ist_calculate(D,L1_0,F11,F22):
    De=sum(D1)/len(D1)
    Di=De-d
    D=(De+Di)/2
    AD=standabw(D1,De)
    Lo=sum(L1_0)/len(L1_0)
    F1=sum(F11)/len(F11)#4.83
    AF1=standabw(F11,F1)#0.021
    F2=sum(F12)/len(F12)#7.97
    AF2=standabw(F12,F2)#0.03
    Fn=0
    R=(F1-F2)/(L1-L2)
    v=F1-R*L1#verschiebungswert
    Fo=v+R*Lo
    test=R*(L1-Lo)+Fo
    Fozul=None
    Sn=(Fn-Fo)/R
    LH=11.755
    Lk=Lo-LH
    Ln=Lo+Sn
    n=(Lo-LH)/d


    to=(8*Di*Fo)/(np.pi*d**3) #vermutlich richtig D=Di
    Rm=336.61
    tozul=0.45*Rm
    ti1=(8*Di*F1)/(np.pi*d**3)
    ti2=(8*Di*F2)/(np.pi*d**3)
    tih=(8*Di*(F2-F1))/(np.pi*d**3)
    tzul=None                    #vermitlich materialkonstante da max für x10CrNi18-8 bei 950 liegt

    k=((Di/d)+0.5)/((Di/d)-0.75)
    tk1=k*ti1
    tk2=k*ti2
    tkh=k*tih
    tkO=None #asu Goodman Diagramm, Materialkonstante?
    tkH=None

    q=((De/d)+0.5)/((De/d)-0.75)

    w=Di/d

    gewicht=M1/5 #kg/1000stk

    print(f"""
    FEDERBERECHNUNG - IST\n
        {werkstoff}
        G={G}, E={E}, p={p}
        --------------------------------------------------------------------
        d \t=\t       \t\t    {round(d,4)}\tmm\t
        De\t=\t       \t\t    {round(De,2)}\tmm\t
        AD\t=\pm\t    \t\t    {round(AD,2)}\tmm\t
        n\t=\t        \t    {round(n,2)}\t\t
        Lo\t=\t       \t\t    {round(Lo,2)}\tmm\t
        Fo\t=\t       \t\t    {round(Fo,2)}\tN\t
        L1\t=\t       \t\t    {round(L1,2)}\t\tmm\t
        F1\t=\t       \t\t    {round(F1,2)}\tN\t
        AF1\t=\pm\t   \t\t    {round(AF1,2)}\tN\t
        L2\t=\t       \t\t    {round(L2,2)}\t\tmm\t
        F2\t=\t       \t\t    {round(F2,2)}\t\tN\t
        AF2\t=\pm\t   \t\t    {round(AF2,2)}\tN\t
        Lk\t=\t       \t\t    {round(Lk,2)}\tmm\t
        Ln\t=\t       \t\t    {round(Ln,2)}\tmm!\t
        Fn\t=\t       \t\t    {round(Fn,2)}\t\tN!\t
        R\t=\t        \t    {round(R,2)}\tN/mm\t\n
        to\t=\t       \t\t    {round(to,2)}\tN/mm^2!\t
        tozul\t=\t    \t\t    {round(tozul,2)}\tN/mm^2!\t
        ti1\t=\t      \t\t    {round(ti1,2)}\tN/mm^2\t
        ti2\t=\t      \t\t    {round(ti2,2)}\tN/mm^2\t
        tih\t=\t      \t\t    {round(tih,2)}\tN/mm^2\t
        \n
        tk1\t=\t      \t\t    {round(tk1,2)}\tN/mm^2\t
        tk2\t=\t      \t\t    {round(tk2,2)}\tN/mm^2\t
        tkh\t=\t      \t\t    {round(tkh,2)}\tN/mm^2\t
        k\t=\t        \t    {round(k,2)}\t\t
        \n
        q\t=\t        \t    {round(q,2)}\t\t
        \n
        w\t=\t        \t    {round(w,2)}\t\t
        2LH\t=\t      \t\t    {round(LH,2)}\tmm\t
        \n
        Gewicht\t:\t  \t\t    {round(gewicht,3)}\tkg/1000 Stück\t
        --------------------------------------------------------------------
    """)


def Soll_calculate(F1,F2,L1,L2,Dm,Lom):
    
    De=sum(D1)/len(D1)
    Di=De-d
    D=(De+Di)/2
    AD=standabw(D1,De)
    Lo=sum(L1_0)/len(L1_0)
    F1=sum(F11)/len(F11)#4.83
    AF1=standabw(F11,F1)#0.021
    F2=sum(F12)/len(F12)#7.97
    AF2=standabw(F12,F2)#0.03
    Fn=0
    R=(F1-F2)/(L1-L2)
    v=F1-R*L1#verschiebungswert
    Fo=v+R*Lo
    test=R*(L1-Lo)+Fo
    Fozul=None
    Sn=(Fn-Fo)/R
    LH=11.755
    Lk=Lo-LH
    Ln=Lo+Sn
    n=(Lo-LH)/d


    to=(8*Di*Fo)/(np.pi*d**3) #vermutlich richtig D=Di
    Rm=336.61
    tozul=0.45*Rm
    ti1=(8*Di*F1)/(np.pi*d**3)
    ti2=(8*Di*F2)/(np.pi*d**3)
    tih=(8*Di*(F2-F1))/(np.pi*d**3)
    tzul=None                    #vermitlich materialkonstante da max für x10CrNi18-8 bei 950 liegt

    k=((Di/d)+0.5)/((Di/d)-0.75)
    tk1=k*ti1
    tk2=k*ti2
    tkh=k*tih
    tkO=None #asu Goodman Diagramm, Materialkonstante?
    tkH=None

    q=((De/d)+0.5)/((De/d)-0.75)

    w=Di/d

    gewicht=M1/5 #kg/1000stk

    print(f"""
    FEDERBERECHNUNG - SOLL\n
        {werkstoff}
        G={G}, E={E}, p={p}
        --------------------------------------------------------------------
        d \t=\t       \t\t    {round(d,2)}\tmm\t
        d (soll) \t=\t    \t\t            \tmm\t
        De\t=\t       \t\t    {round(De,2)}\tmm\t
        AD\t=\pm\t    \t\t    {round(AD,2)}\tmm\t
        n\t=\t        \t\t    \t\t
        Lo\t=\t       \t\t    {round(Lo,2)}\tmm\t
        Fo\t=\t       \t\t    {round(Fo,2)}\tN\t
        Fozul\t=\t    \t\t    {Fozul}\tN\t
        L1\t=\t       \t\t    {round(L1,2)}\tmm\t
        F1\t=\t       \t\t    {round(F1,2)}\tN\t
        AF1\t=\pm\t   \t\t    {round(AF1,2)}\tN\t
        L2\t=\t       \t\t    {round(L2,2)}\tmm\t
        F2\t=\t       \t\t    {round(F2,2)}\tN\t
        AF2\t=\pm\t   \t\t    {round(AF2,2)}\tN\t
        Lk\t=\t       \t\t    {round(Lk,2)}\tmm\t
        Ln\t=\t       \t\t    {round(Ln,2)}\tmm!\t
        Fn\t=\t       \t\t    {round(Fn,2)}\tN!\t
        R\t=\t        \t\t    {round(R,2)}\tN/mm\t\n
        to\t=\t       \t\t    {round(to,2)}\tN/mm^2!\t
        tozul\t=\t    \t\t    {round(tozul,2)}\tN/mm^2!\t
        ti1\t=\t      \t\t    {round(ti1,2)}\tN/mm^2\t
        ti2\t=\t      \t\t    {round(ti2,2)}\tN/mm^2\t
        tih\t=\t      \t\t    {round(tih,2)}\tN/mm^2\t
        tzul\t=\t     \t\t    {round(tzul,2)}\tN/mm^2\t
        \n
        tk1\t=\t      \t\t    {round(tk1,2)}\tN/mm^2\t
        tk2\t=\t      \t\t    {round(tk2,2)}\tN/mm^2\t
        tkh\t=\t      \t\t    {round(tkh,2)}\tN/mm^2\t
        tkO\t=\t      \t\t    {round(tkO,2)}\tN/mm^2\t
        tkH\t=\t      \t\t    {tkH}\tN/mm^2\t
        k\t=\t        \t\t    {round(k,2)}\t\t
        \n
        q\t=\t        \t\t    {round(q,2)}\t\t
        \n
        w\t=\t        \t\t    {round(w,2)}\t\t
        2LH\t=\t      \t\t    {round(LH,2)}\tmm\t
        \n
        Gewicht\t:\t  \t\t    {round(gewicht,3)}\tkg/1000 Stück\t
        --------------------------------------------------------------------
    """)

ist_calculate(D1,L1_0,F11,F12)
Soll_calculate(5,7.9,105,142,3.68,58.78)