import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
#k=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] 

#def f(x):
#    (A(k)*np.cos(w(k)*x)+B(k)*sin(w(k)*x))





x= np.linspace(0,2*np.pi,1000)
y= abs(np.sin(x))

plt.plot(x,y, label=r"$|\sin(x)|$")
plt.xlim(0,2*np.pi+1)
plt.ylim(0,1.25)
plt.xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],[r"$0$",r"$\frac{1}{2}\pi$",r"$\pi$",r"$\frac{3}{2}\pi$",r"$2\pi$"])
plt.yticks([0,0.5,1],[0,0.5,1])
plt.xlabel(r"x")
plt.ylabel(r"y")
plt.legend(loc="best")

plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot1.pdf')

x=np.linespace(-np.pi,np.pi,1000)
y=x

plt.plot(x,y,label=r"$x$")
