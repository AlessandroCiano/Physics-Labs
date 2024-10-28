import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit

# data load
x,y=numpy.loadtxt('F:\\VERO UTENTE\\Desktop\\Uni\\Lab2\\4\\carica_cond.txt',unpack=True)

Dx=numpy.full(len(x),1)
Dy=numpy.full(len(y),1)

# scatter plot with error bars
plt.errorbar(x,y,Dy,Dx,linestyle = '', color = 'black', marker = '.')

# bellurie
plt.rc('font',size=16)
plt.xlabel('$\Delta t$  [s]')
plt.ylabel('Pseudo-$V$  [V]')
plt.title('Data plot w numerical fit')
plt.minorticks_on()

# make the array with initial values
# must be adjusted!!!!
init=(800,2900,2600)

# set the error
sigma=Dy
w=1/sigma**2

# define the linear function
# note how parameters are entered
# note the syntax
def ff(x, aa, bb, cc):
    return aa-aa*numpy.exp(-(x-cc)/bb)

# call the routine (NOTE THE ABSOLUTE SIGMA PARAMETER!)
pars,covm=curve_fit(ff,x,y,init,sigma,absolute_sigma=False)

# calculate the chisquare for the best-fit funtion
# note the indexing of the pars array elements
chi2 = ((w*(y-ff(x,pars[0],pars[1],pars[2]))**2)).sum()

# determine the ndof
ndof=len(x)-len(init)

# print results on the console
print(pars)
print(covm)
print (chi2, ndof)

# # print the same in a slightly more readible version (not very useful, yet)
print('a = ', pars[0], '+/-', numpy.sqrt(covm[0,0]))
print('b = ', pars[1], '+/-', numpy.sqrt(covm[1,1]))
print('c = ', pars[1], '+/-', numpy.sqrt(covm[2,2]))

# prepare a dummy xx array (with 1000 linearly spaced points)
xx=numpy.linspace(min(x),max(x),1000)

# plot the fitting curve
plt.plot(xx,ff(xx,*pars), color='red')

# show the plot
plt.show()
