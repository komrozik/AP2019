import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Datenimport:

D,L_0 = np.genfromtxt("data1.1.txt",unpack=True)
D_mittelwert = sum(D)/len(D)
L_0_mittlewert = sum(L_0)/len(L_0)
F1,F2 = np.genfromtxt("data2.2.txt",unpack = True)
F1_mittelwert = sum(F1)/len(F1)
F2_mittelwert = sum(F2)/len(F2)
