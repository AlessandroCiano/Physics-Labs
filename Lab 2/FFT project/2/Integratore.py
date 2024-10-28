import numpy as np
import matplotlib.pyplot as plt
from math import* #import all function from math


frequency_quadra = 241.7
period = 1/frequency_quadra
amplitude = 1
n = 1000
frequency_taglio = 234


t=np.linspace(0, period*4, 1000,endpoint=False)


bn = lambda n: (2.*(1-np.cos(n*np.pi)))/(np.pi* n)

dphi = lambda n: np.arctan( -frequency_quadra * n / frequency_taglio )

Gk = lambda n: 1 / np.sqrt( 1+ (frequency_quadra*n / frequency_taglio)**2 )

s = sum( bn(k)*Gk(k)*np.sin(2*np.pi*frequency_quadra*k* t + dphi(k)) for k in range (1,n+1))


t = t/period
plt.plot(t,s,label="Serie di Fourier (%d termini di sviluppo)" % (n))
plt.xlabel("Tempo [T]")
plt.ylabel
plt.legend(loc='upper right',prop={'size':6})


# Plot the data
plt.plot(t, s,'g')
plt.show()