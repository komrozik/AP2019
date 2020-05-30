import numpy as np
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import *
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import scipy.constants as const

U,N=np.genfromtxt("input/Kennlinie.dat",unpack=True)
N=unp.uarray(N,np.sqrt(N))




#Ausgabe
print(f"""

""")
