import numpy

w=14177.693
dw=0.384

tau=0.38445
dtau=0.00296e-3

C=0.47e-6
dC=C=0.47e-6*0.1

##########

pippo=w**2+1/tau**2
dpippo=numpy.sqrt((2*dw)**2+(2*dtau)**2)


Qf=(w*tau)/2
dQf=Qf*numpy.sqrt((dtau/tau)**2+(dw/w)**2)

L=1/(C*pippo)
dL=L*numpy.sqrt(0.1**2+(dpippo/pippo)**2)


print('Qf=', L, '+-', dQf)
print('L=', L ,'+-', dL)