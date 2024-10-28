import pylab
import numpy
from scipy.optimize import curve_fit


# data load
x,dx,Vout,dVout,Vin,dVin=pylab.loadtxt('/home/studenti/datifit/risonanteRLC654542.txt',unpack=True)

y=Vout/Vin
dy=y*numpy.sqrt((dVin/Vin)**2+(dVout/Vout)**2)


# scatter plot with error bars
pylab.errorbar(x,y,dx,dy,linestyle = '', color = 'black', marker = '.')

# bellurie
pylab.rc('font',size=16)
pylab.xlabel('$Tempo$  [us]',fontsize=18)
pylab.ylabel('$Digit voltaggio$ [a.u.]',fontsize=18)
pylab.minorticks_on()
#pylab.xscale('log')
#pylab.yscale('log')

# make the array with initial values
init=(1.05e-3, 320**2, 1.343e-6)

# set the error
sigma=dy
w=1/sigma**2

# define the linear function
# note how parameters are entered
# note the syntax
def ff(x, a, b, c):
    return a*x/numpy.sqrt((1-x**2/b)**2+c*x**2)


# call the routine
pars,covm=curve_fit(ff,x,y,init,sigma,absolute_sigma=False)

# calculate the chisquare for the best-fit funtion
# note the indexing of the pars array elements
chi2 = ((w*(y-ff(x,*pars))**2)).sum()

# determine the ndof
ndof=len(x)-len(init)

# print results on the console
print( 'a=', pars[0],'+-', numpy.sqrt(covm[0,0]))
print( 'b=', pars[1],'+-', numpy.sqrt(covm[1,1]))
print( 'b=', pars[2],'+-', numpy.sqrt(covm[2,2]))
#print(covm)
print ('chi/ndof', chi2/ndof)

# prepare a dummy xx array (with 100 linearly spaced points)
xx=numpy.linspace(numpy.min(x),numpy.max(x),500)


# plot the fitting curve
pylab.plot(xx, ff(xx,*pars), color='red')

# show the plot
pylab.show()