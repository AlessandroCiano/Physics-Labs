from matplotlib import pylab
import numpy
from scipy.optimize import curve_fit

# data load
x=numpy.array([1.00, 2.00, 0.5, 1.5, 1.75, 1.25, 0.75])*1e-9*2
Dx=x*0.05
y= numpy.array([374.3, 441.56, 301.15, 413.81, 427.72, 396.11, 343.96])*1e-6
Dy=numpy.array([2.5,2.5,2.5,2.5,2.5,2.5,2.5])*1e-6

# Create a figure and a set of subplots
fig, (ax1, ax2) = pylab.subplots(2, sharex=True, gridspec_kw={'hspace': 0.3, 'height_ratios': [2.5, 1]}, figsize=(10,8))

# Main plot on ax1
ax1.errorbar(x, y, Dy, Dx, linestyle='', color='black', marker='.')
ax1.set_ylabel('$T$  [s]')
ax1.minorticks_on()

# define the function (linear, in this example)
def ff(xx, a, b):
    return a*numpy.log(xx/b)

# define the initial values (STRICTLY NEEDED!!!)
init=(0.0001, 1e-9*60*1e-3)

# prepare a dummy xx array (with 2000 linearly spaced points)
xx=numpy.linspace(min(x),max(x),2000)

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
print('chi2, ndof:',chi2, ndof)

# plot the best fit curve on ax1
ax1.plot(xx,ff(xx,*pars), color='red')

# Calculate residuals
residuals = y - ff(x,*pars)

# Normalize residuals
normalized_residuals = residuals / Dy

# Residuals plot on ax2
ax2.errorbar(x, normalized_residuals, yerr=1, linestyle='', color='black', marker='.')
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel('$Q$  [C]')
ax2.set_ylabel('Residui normalizzati')
ax2.minorticks_on()

# show the plot
pylab.show()

# save the figure
fig.savefig('fit_guadagnoeresidui.png', dpi=300)
