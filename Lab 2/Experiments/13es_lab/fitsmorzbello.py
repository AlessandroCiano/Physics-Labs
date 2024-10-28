import pylab
import numpy
from scipy.optimize import curve_fit

T=5650

# data load
x,dx,y,dy=pylab.loadtxt('/home/studenti/dati_arduino/ferroila7fit.txt',unpack=True)

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
init=(520, 590, 1.57, 2*numpy.pi/T, 3e4)

# set the error
sigma=dy
w=1/sigma**2

# define the linear function
# note how parameters are entered
# note the syntax
def ff(x, A, B, phi, w, tau):
    return A*numpy.exp(-x/tau)*numpy.cos(w*x+phi)+B


# call the routine
pars,covm=curve_fit(ff,x,y,init,sigma,absolute_sigma=False)

# calculate the chisquare for the best-fit funtion
# note the indexing of the pars array elements
chi2 = ((w*(y-ff(x,*pars))**2)).sum()

# determine the ndof
ndof=len(x)-len(init)

# print results on the console
print( 'w=', pars[3]*10e6,'+-', numpy.sqrt(covm[3,3])*10e6)
print( 'tau=', pars[4]/10e3,'+-', numpy.sqrt(covm[4,4])/10e3)
#print(covm)
print ('chi/ndof', chi2/ndof)

# prepare a dummy xx array (with 100 linearly spaced points)
xx=numpy.linspace(numpy.min(x),numpy.max(x),500)


# plot the fitting curve
pylab.plot(xx, ff(xx,*pars), color='red')

# show the plot
pylab.show()