import numpy as np
import matplotlib.pyplot as plt
from math import* #import all function from math


frequency_quadra = 200
period = 1/frequency_quadra
n = 1000
frequency_taglio = 4000
dutycycle=0.5 #variabile tra 0 e 1


t=np.linspace(-period*2, period*2, 1000,endpoint=False)

w=2.*np.pi/frequency_quadra #frequenza

bk = lambda k: (2./(np.pi* k))*(1/np.sqrt(1+(frequency_quadra * n/frequency_taglio)**2))

dphi = lambda k: np.arctan( -frequency_quadra * n / frequency_taglio )



s = dutycycle + sum((bk(k)*np.sin(k*np.pi*dutycycle + dphi(k))*np.cos(w*k*t + dphi(k))) for k in range (1,n+1))


t = t/period
plt.plot(t,s,label="Serie di Fourier (%d termini di sviluppo)" % (n))
plt.xlabel("Tempo [T]")
plt.ylabel
plt.legend(loc='upper left',prop={'size':6})


# Plot the data
plt.plot(t, s,'g')
plt.show()