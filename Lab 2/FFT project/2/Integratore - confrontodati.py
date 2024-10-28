import numpy as np
import matplotlib.pyplot as plt
from math import* #import all function from math

# data load
x,y=np.loadtxt('F:\\VERO UTENTE\Desktop\\Uni\\Lab2\\4\\misureArd4.txt',unpack=True)

Dx=np.full(len(x),1)
Dy=np.full(len(y),1)

#noramlizzo asse y ei suoi errori
Dy=Dy/np.max(y)
y=y/np.max(y)
#noramlizzo i tempi x in periodi ei suoi errori idem
Dx=Dx/4120
x=x/4120


# scatter plot with error bars
plt.errorbar(x,y,Dy,Dx,linestyle = '', color = 'black', marker = '.')


frequency_quadra = 242.7
period = 1/frequency_quadra
amplitude = 1
n = 1000
frequency_taglio =230


t=np.linspace(0, period*4, 1000,endpoint=False)


bn = lambda n: (2.*(1-np.cos(n*np.pi)))/(np.pi* n)

dphi = lambda n: np.arctan( -frequency_quadra * n / frequency_taglio )

Gk = lambda n: 1 / np.sqrt( 1+ (frequency_quadra*n / frequency_taglio)**2 )

s = sum( bn(k)*Gk(k)*np.sin(2*np.pi*frequency_quadra*k* t + dphi(k)+np.pi) for k in range (1,n+1))/2+0.5


t = t/period
plt.plot(t,s,label="Serie di Fourier (%d termini di sviluppo)" % (n))
plt.xlabel("Tempo [T]")
plt.ylabel
plt.legend(loc='upper right',prop={'size':6})

# Plot the data
plt.plot(t, s,'g')
plt.show()