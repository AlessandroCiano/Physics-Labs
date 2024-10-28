import pylab
import numpy
from scipy.optimize import curve_fit

# data load
x=numpy.array([96, 196, 296, 397, 496])
dx=numpy.array([1,1,1,1,1])
y= numpy.array([2.3036, 592, 892, 1192, 1492])
dy=numpy.array([1,2,2,2,2])

# scatter plot with error bars
pylab.errorbar(x,y,dx,dy,linestyle = '', color = 'black', marker = '.')

# bellurie
pylab.rc('font',size=16)
pylab.xlabel('$V_{in}$  [mV]',fontsize=18)
pylab.ylabel('$V_{out}$ [mV]',fontsize=18)
pylab.minorticks_on()
#pylab.xscale('log')
#pylab.yscale('log')

# make the array with initial values
init=(3)

# set the error
sigma=dy
w=1/sigma**2

# define the linear function
# note how parameters are entered
# note the syntax
def ff(x, A):
    return A*x


# call the routine
pars,covm=curve_fit(ff,x,y,init,sigma,absolute_sigma=False)

# calculate the chisquare for the best-fit funtion
# note the indexing of the pars array elements
chi2 = ((w*(y-ff(x,*pars))**2)).sum()

# determine the ndof
ndof=len(x)-1

# print results on the console
print(pars)
print(covm)
print (chi2, ndof)

# prepare a dummy xx array (with 100 linearly spaced points)
xx=numpy.linspace(numpy.min(x),numpy.max(x),500)


# plot the fitting curve
pylab.plot(xx, ff(xx,*pars), color='red')

# show the plot
pylab.show()
