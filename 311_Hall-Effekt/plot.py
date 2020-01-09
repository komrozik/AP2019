import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

def f(x,a,b):
    return a*x+b


#1) Data 2 in reihe geschaltete BFeld Messung | I = Stromstärke, B = Magnetfeld
I,B=np.genfromtxt("data2.txt", unpack = True)

#i=0
# while(i<=len(B)-1):
#     print(f"{B[i]} & {I[i]} \\\\ \n")
#     i=i+1


params,cov = curve_fit(f,B,I)
Bparams,Bcov = curve_fit(f,I,B)
errors = np.sqrt(np.diag(cov))
x_plot = np.linspace(0,1200)

plt.plot(x_plot,f(x_plot,*params),label = "linearer Fit")
plt.plot(B,I,"rx",label="Messwerte")
plt.xlabel(f"Magnetfeld $B \;\;mT$")
plt.ylabel(f"Stromstärke $I \;\; A$")
plt.legend()
plt.savefig('build/plot1.pdf',bbox_inches='tight')
#plt.show()
plt.close()

#2) Data 3 | I_b = Stromstärke des B Felds, U_hall = Hallspannung, I_d = Durchflussstrom(konstant 10 A)
I_b,U_hall,I_d = np.genfromtxt("data3.txt", unpack = True)

#i=0
# while(i<=len(B)-1):
#     print(f"{B[i]} & {I[i]} \\\\ \n")
#     i=i+1

B = f(I_b,*Bparams)
params,cov = curve_fit(f,B,U_hall)
errors = np.sqrt(np.diag(cov))
x_plot = np.linspace(0,1300)

plt.plot(x_plot,f(x_plot,*params),label = "linearer Fit")
plt.plot(B,U_hall,"rx",label="Messwerte")
plt.xlabel(f"Magnetfeld $B \;\;mT$")
plt.ylabel(f"Hall Spannung $U \;\; mV$")
plt.legend()
plt.savefig('build/plot1.pdf',bbox_inches='tight')
#plt.show()
plt.close()

#3) Data 4 |  I_d = Durchflussstrom,U_hall = Hallspannung,I_b = Stromstärke des B Felds(konstant 5 A)
I_b,U_hall,I_d = np.genfromtxt("data4.txt", unpack = True)

#i=0
# while(i<=len(B)-1):
#     print(f"{B[i]} & {I[i]} \\\\ \n")
#     i=i+1

#B = f(I_b,*Bparams)
params,cov = curve_fit(f,I_d,U_hall)
errors = np.sqrt(np.diag(cov))
x_plot = np.linspace(0,10)

plt.plot(x_plot,f(x_plot,*params),label = "linearer Fit")
plt.plot(I_d,U_hall,"rx",label="Messwerte")
plt.xlabel(f"Durchflussstrom $I_d \;\; A$")
plt.ylabel(f"Hall Spannung $U \;\; mV$")
plt.legend()
plt.savefig('build/plot1.pdf',bbox_inches='tight')
#plt.show()
plt.close()

#4) Data 5 | I_b = Stromstärke des B Felds, U_hall = Hallspannung, I_d = Durchflussstrom(konstant 10 A)
I_b,U_hall,I_d = np.genfromtxt("data5.txt", unpack = True)

#i=0
# while(i<=len(B)-1):
#     print(f"{B[i]} & {I[i]} \\\\ \n")
#     i=i+1

B = f(I_b,*Bparams)
params,cov = curve_fit(f,B,U_hall)
errors = np.sqrt(np.diag(cov))
x_plot = np.linspace(0,1300)

plt.plot(x_plot,f(x_plot,*params),label = "linearer Fit")
plt.plot(B,U_hall,"rx",label="Messwerte")
plt.xlabel(f"Magnetfeld $B \;\;mT$")
plt.ylabel(f"Hall Spannung $U \;\; mV$")
plt.legend()
plt.savefig('build/plot1.pdf',bbox_inches='tight')
#plt.show()
plt.close()

#5) Data 6 |  I_d = Durchflussstrom,U_hall = Hallspannung,I_b = Stromstärke des B Felds(konstant 5 A)
I_b,U_hall,I_d = np.genfromtxt("data6.txt", unpack = True)

#i=0
# while(i<=len(B)-1):
#     print(f"{B[i]} & {I[i]} \\\\ \n")
#     i=i+1

#B = f(I_b,*Bparams)
params,cov = curve_fit(f,I_d,U_hall)
errors = np.sqrt(np.diag(cov))
x_plot = np.linspace(0,10)

plt.plot(x_plot,f(x_plot,*params),label = "linearer Fit")
plt.plot(I_d,U_hall,"rx",label="Messwerte")
plt.xlabel(f"Durchflussstrom $I_d \;\; A$")
plt.ylabel(f"Hall Spannung $U \;\; mV$")
plt.legend()
plt.savefig('build/plot1.pdf',bbox_inches='tight')
#plt.show()
plt.close()

#6) Data 7 | I_b = Stromstärke des B Felds, U_hall = Hallspannung, I_d = Durchflussstrom(konstant 10 A)
I_b,U_hall,I_d = np.genfromtxt("data7.txt", unpack = True)

#i=0
# while(i<=len(B)-1):
#     print(f"{B[i]} & {I[i]} \\\\ \n")
#     i=i+1

B = f(I_b,*Bparams)
params,cov = curve_fit(f,B,U_hall)
errors = np.sqrt(np.diag(cov))
x_plot = np.linspace(0,1300)

plt.plot(x_plot,f(x_plot,*params),label = "linearer Fit")
plt.plot(B,U_hall,"rx",label="Messwerte")
plt.xlabel(f"Magnetfeld $B \;\;mT$")
plt.ylabel(f"Hall Spannung $U \;\; mV$")
plt.legend()
plt.savefig('build/plot1.pdf',bbox_inches='tight')
plt.show()
plt.close()

#7) Data 8 |  I_d = Durchflussstrom,U_hall = Hallspannung,I_b = Stromstärke des B Felds(konstant 5 A)
I_b,U_hall,I_d = np.genfromtxt("data4.txt", unpack = True)

#i=0
# while(i<=len(B)-1):
#     print(f"{B[i]} & {I[i]} \\\\ \n")
#     i=i+1

#B = f(I_b,*Bparams)
params,cov = curve_fit(f,I_d,U_hall)
errors = np.sqrt(np.diag(cov))
x_plot = np.linspace(0,10)

plt.plot(x_plot,f(x_plot,*params),label = "linearer Fit")
plt.plot(I_d,U_hall,"rx",label="Messwerte")
plt.xlabel(f"Durchflussstrom $I_d \;\; A$")
plt.ylabel(f"Hall Spannung $U \;\; mV$")
plt.legend()
plt.savefig('build/plot1.pdf',bbox_inches='tight')
#plt.show()
plt.close()

