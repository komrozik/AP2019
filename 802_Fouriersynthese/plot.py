import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

GK=16 #Genauigkeit Fourier wahrer GK ist GK+1 weil A0 und B0 definiert ist

#FUNKTION 1 - f(x)=x
#-------------------------------------------------------------------------------------------------------------------
#Fouriersynthese
Ak=0    #Da Funktion ungerade ist, es gilt: f(-x)=-f(x)
#w=(2*np.pi*k)/T1
T1=2*np.pi

k=1
Bks = []
Bks.append(0) #Ist in Fourier definiert B0=0
while k<=GK:
    f = lambda x: (1/np.pi*x*np.sin((2*np.pi*k)/T1*x))
    B=integrate.quad(f,-T1/2,T1/2)
    Bks.append(B[0]) #B ist Tuple aus (values, error) mit B[0] wir nur der wert abgegriffen
    k+=1
k=1 # zurücksetzen von Laufvariable für spätere Funktion

#FUNKTION 2 - f(x)= |sin(x)|
#-------------------------------------------------------------------------------------------------------------------
#Fouriersynthese
Bk=0 # Da Funktion gerade ist, es gilt: f(x)=f(-x)
#w=(2*np.pi*k)/T2
T2=np.pi

k=1
Aks = []
A0dx= lambda x: (1/T2*np.abs(np.sin(x)))
A0=integrate.quad(A0dx,-T2/2,T2/2)
Aks.append(A0[0])
while k<=GK:
    f= lambda x: (2/T2*np.abs(np.sin(x))*np.cos((2*np.pi*k)/T2*x))
    A=integrate.quad(f,-T2/2,T2/2)
    Aks.append(A[0])
    k+=1

##PLOT 2 - x
x2=np.linspace(-np.pi,np.pi,1000)
y2=x2
ft_x=0
for i in range(GK):
    ft_x += Bks[i]*np.sin(i*x2)

plt.plot(x2,ft_x,label="Fouriersynthese")
plt.plot(x2,y2,label=r"$x$")
plt.xlim(-np.pi-1,np.pi+1)
plt.ylim(-5,5)
plt.xticks([-np.pi,0,np.pi],[r"$-\pi$","$0$",r"$\pi$"])
plt.xlabel(r"x")
plt.ylabel(r"y")
plt.legend(loc="best")
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot2.pdf')
plt.close()



##PLOT 1 - |sin(x)| 
x1= np.linspace(0,2*np.pi,1000)
y1= abs(np.sin(x1))
ft_abs=0

for i in range(GK):
    ft_abs += Aks[i]*np.cos((2*np.pi*i)/T2*x1)

plt.plot(x1,ft_abs,label="Fouriersynthese")
plt.plot(x1,y1, label=r"$|\sin(x)|$")
plt.xlim(0,2*np.pi+1)
plt.ylim(0,1.25)
plt.xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],[r"$0$",r"$\frac{1}{2}\pi$",r"$\pi$",r"$\frac{3}{2}\pi$",r"$2\pi$"])
plt.yticks([0,0.5,1],[0,0.5,1])
plt.xlabel(r"x")
plt.ylabel(r"y")
plt.legend(loc="best")

plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot1.pdf')
plt.close()


#-------------------------------------------------------------------------------------------------------------------
#Ausgabe
print("Funktion f(x)=x - Bk: ")
print(Bks)

print("Funktion f(x)=|sin(x)| - Ak: ")
print(Aks)



