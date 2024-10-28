from matplotlib import pylab
import numpy
from scipy.optimize import curve_fit

# data load
x = numpy.array([104, 204, 305, 405, 507, 607])
Dx = numpy.array([2, 2, 2, 2, 3, 3])
y = numpy.array([1.03, 2.03, 3.03, 4.03, 4.98, 6.1])*1e3
Dy = numpy.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])*1e3

# scatter plot with error bars
pylab.errorbar(x, y, Dy, Dx, linestyle = '', color = 'black', marker = '.')

# bellurie
pylab.rc('font',size=18)
pylab.xlabel('$V_{in}$  [mV]')
pylab.ylabel('$V_{out}$  [mV]')
pylab.minorticks_on()
#pylab.yscale('log')

# AT THE FIRST ATTEMPT COMMENT FROM HERE TO THE END

# define the function (linear, in this example)
def ff(xx, a):
    return a*xx

# define the initial values (STRICTLY NEEDED!!!)
init=(10)

# prepare a dummy xx array (with 2000 linearly spaced points)
xx=numpy.linspace(min(x),max(x),2000)

# plot the fitting curve computed with initial values
# AT THE SECOND ATTEMPT THE FOLLOWING LINE MUST BE COMMENTED
#pylab.plot(xx,ff(xx,*init), color='blue')

# set the error
sigma=Dy
w=1/sigma**2

# call the minimization routine
pars,covm=curve_fit(ff,x,y,init,sigma, absolute_sigma=False)


# calculate the chisquare for the best-fit function
chi2 = ((w*(y-ff(x,*pars))**2)).sum()

# determine the ndof
ndof=len(x)-1

# print results on the console
print('pars:',pars)
print('covm:',covm)
print ('chi2, ndof:',chi2, ndof)


# plot the best fit curve
pylab.plot(xx,ff(xx,*pars), color='red')

# show the plot
pylab.show()