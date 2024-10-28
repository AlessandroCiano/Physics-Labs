import numpy as np
from matplotlib import pyplot as plt

f=40
ft=20
T=0.005
a0=0
omega=2.*np.pi*f
terms= 1000

t=np.linspace(0,T*6,10000,endpoint=False)
n=list(range(1,10000,2))
wn= lambda n: n*omega
fn= lambda n: (n*omega)/(2.*np.pi)
bn= lambda n : 2/(n*np.pi)
gn= lambda n: 1/(np.sqrt(1+((ft/(n*omega)/2.*np.pi))**2))
dphin= lambda n: np.arctan(ft/((n*omega)/2.*np.pi))

s= a0/2. +sum(bn(k)*gn(k)*np.sin(2.*np.pi*fn(k)*t+dphin(k))for k in range(1,terms+1)if k%2!=0)

plt.plot(t,s)
plt.xlabel("tempo [t]")
plt.ylabel('y=w(t) [u.a.]')
plt.legend(loc='best',prop={'size':10})
plt.show()