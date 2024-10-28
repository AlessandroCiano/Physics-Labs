from matplotlib import pylab
import numpy
from scipy.optimize import curve_fit

# data load
x=numpy.array([5, 4, 3])
y= numpy.array([2.3036, 2.0559, 1.5251])*1e6
Dy=numpy.array([0.0016, 0.0013, 0.0007])*1e6

print(y)
print(Dy)


# Create a figure and a set of subplots
fig, (ax1, ax2) = pylab.subplots(2, sharex=True, gridspec_kw={'hspace': 0.3, 'height_ratios': [2.5, 1]}, figsize=(10,8))

# Main plot on ax1
ax1.errorbar(x, y, Dy, linestyle='', color='black', marker='.')
ax1.set_ylabel('$1/\lambda$  [1/m]')
ax1.minorticks_on()

# define the function (linear, in this example)
def ff(xx, a):
    return a*(1/4 - 1/(xx**2))

# define the initial values (STRICTLY NEEDED!!!)
init=(11e7)

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
ndof=len(x)-1

# print results on the console
print('pars:',pars)
print('covm:',covm)
print('chi2, ndof:',chi2, ndof)
print('R = ', pars[0], ' +/- ', numpy.sqrt(covm[0,0]))

# plot the best fit curve on ax1
ax1.plot(xx,ff(xx,*pars), color='red')

# Calculate residuals
residuals = y - ff(x,*pars)

# Normalize residuals
normalized_residuals = residuals / Dy

#calculate and print reduced chi square
chi2red = chi2/ndof
print('chi2red:', chi2red)

# Residuals plot on ax2
ax2.errorbar(x, normalized_residuals, yerr=1, linestyle='', color='black', marker='.')
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel('$n_2$')
ax2.set_ylabel('Residui normalizzati')
ax2.minorticks_on()

# show the plot
pylab.show()

# save the figure
fig.savefig('fit_guadagnoeresidui.png', dpi=300)
