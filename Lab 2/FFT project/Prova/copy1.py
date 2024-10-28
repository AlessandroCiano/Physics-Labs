import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as spi
from scipy.signal import square

A=1.
T=2.
a0=1
omega=2.*np.pi/T
terms= 10 #num termini sviluppo

t=np.linspace(0,T,10000,endpoint=False)

#coefficienti n-esimi
def bn(n):
    return (2.*A*(1-np.cos(n*np.pi)))/(np.pi* n)

 #somma di tutti i termini
s= a0/2. + sum( bn(k)*np.sin(k*omega*t) for k in range (1,terms+1))

plt.plot(t,s,label="Serie di Fourier [n=%d]" %terms)
plt.xlabel("tempo [t]")
plt.ylabel('y=f(t) [u.a.]')
plt.legend(loc='best',prop={'size':10})
plt.show()