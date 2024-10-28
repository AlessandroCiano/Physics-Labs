from matplotlib import pylab
import numpy
from scipy.optimize import curve_fit

# data load
x, Dx, Vout, DVout, Vin, DVin = pylab.loadtxt(r'dati16.txt', unpack=True)

y=Vout/Vin
Dy=y*numpy.sqrt((DVout/Vout)**2+(DVin/Vin)**2)

# scatter plot with error bars
pylab.errorbar(x,y,Dy,Dx,linestyle = '', color = 'black', marker = '.')

# bellurie
pylab.rc('font',size=18)
pylab.xlabel('$f$  [Hz]')
pylab.ylabel('$V$  [V]')
pylab.minorticks_on()

# AT THE FIRST ATTEMPT COMMENT FROM HERE TO THE END

# define the function (linear, in this example)
def ff(x, A, B, C):
    return A*x/numpy.sqrt((B*x)**2+(1-(x/C)**2)**2)

# define the initial values (STRICTLY NEEDED!!!)
init=(1.66e-4,1.85e-4,2062)

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
ndof=len(x)-len(init)

# print results on the console
print('pars:',pars)
print('covm:',covm)
print ('chi2, ndof:',chi2, ndof)

# plot the best fit curve
pylab.plot(xx,ff(xx,*pars), color='red')

# show the plot
pylab.show()