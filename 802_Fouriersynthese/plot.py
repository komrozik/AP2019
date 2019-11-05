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
#w=(2*np.pi*k)/T
T=2*np.pi

k=1
Bks = []
Bks.append(0) #Ist in Fourier definiert B0=0
while k<=GK:
    f = lambda x: (1/np.pi*x*np.sin((2*np.pi*k)/T*x))
    B=integrate.quad(f,-np.pi,np.pi)
    Bks.append(B[0]) #B ist Tuple aus (values, error) mit B[0] wir nur der wert abgegriffen
    k+=1

#FUNKTION 2 - f(x)= |sin(x)|
#-------------------------------------------------------------------------------------------------------------------
#Fouriersynthese



##PLOT 2 - x
x2=np.linspace(-np.pi,np.pi,1000)
y2=x2
ft=0
for i in range(GK):
    ft += Bks[i]*np.sin(i*x2)

plt.plot(x2,ft,label="Fouriersynthese")
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



