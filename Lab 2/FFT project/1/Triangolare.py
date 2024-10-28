import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth #triangolare già pronta
from scipy.integrate import quad #per integrazione
from math import* #import all function from math


T=2. #periodo
w0=2.*np.pi/T #armonica fondamentale

samples=5000 #suddivisioni

x=np.linspace(0,T,samples)

y=sawtooth(2.*np.pi*x/T, 0.5) #triangolare(x) vale 𝑦 =−1, 𝑓𝑜𝑟 − 𝜋 ≤ 𝑥 ≤ 0
#y= +1, 𝑓𝑜𝑟 0 ≤ 𝑥 ≤ 𝜋


n=50 #num iterazioni
sum=0.0

#def arrai che sarnno i coefficienti della serie
An=[0]
Bn=[0]


#func da integrare
fc=lambda x:sawtooth(2.*np.pi*x/T, 0.5)*cos(w0*i*x)  #i :dummy index

fs=lambda x:sawtooth(2.*np.pi*x/T, 0.5)*sin(w0*i*x)


#determino tutti i coeff
for i in range(1, n+1):
    an=quad(fc,0,T)[0]*(2.0/T) #[0] serve perché quad restituisce un tuple, di cui solo il primo elemtno è il valore dell'integrale
    An.append(an)

    bn=quad(fs,0,T)[0]*(2.0/T)
    Bn.append(bn) #putting value in array Bn

for i in range(1, n+1):
    sum=sum+(An[i]*np.cos(w0*i*x)+Bn[i]*np.sin(w0*i*x))



plt.plot(x,sum,'g',label="Serie di Fourier (%d termini di sviluppo)" % (n))
plt.plot(x,y,'r--',label="Funzione modello")

plt.title("fourier series for square wave")
plt.xlabel("tempo [t]")
plt.ylabel('y=f(t) [u.a.]')
plt.legend(loc='upper right',prop={'size':6})

plt.show()