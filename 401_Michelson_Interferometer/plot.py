import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import *
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import scipy.constants as const

s1,e1,Max1=np.genfromtxt("data/data1.txt",unpack=True)
s1 = s1/100 #Umrechnung in m
e1 = e1/100 #Umrechnung in m
s1 = s1/5.017
e1 = e1/5.017
p2,Max2=np.genfromtxt("data/data2.txt",unpack=True)

print("Bestimmung der Wellenlänge")
d1 = np.sqrt((e1-s1)**2)
lambda1 = 2*d1/(Max1)
lambda1_mittel = sum(lambda1)/size(lambda1)


print("Messung des Brechungsindex")
Max2 = sum(Max2)/size(Max2)
p2 = p2[0]
delta_n = (Max2*lambda1_mittel)/(2*1)
n = 1 + delta_n*(1.0132)/(1-p2)*(293.15)/(273.15)

#Ausgabe
print(f"""
Die Längen: {d1}
Die ausgerechneten Wellenlängen: {lambda1}
Und die mittl. Wellenlänge: {lambda1_mittel} 
Die Brechindexänderung ergibt sich als :{delta_n}
Und damit der Brechungsindex bei {n}
""")
