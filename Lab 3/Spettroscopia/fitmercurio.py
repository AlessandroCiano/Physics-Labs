from matplotlib import pylab
import numpy
from scipy.optimize import curve_fit

# data load
x=numpy.array([546.1, 404.6, 407.7, 435.8, 576.9, 579.0, 406*2, 435.8*2, 546.1*2])*1e-9
y= numpy.array([0.65535,  0.48509,  0.48900,  0.52223,  0.69235,  0.69503 ,  0.9708,  1.0460,  1.31072])
Dy=numpy.array([0.00035, 0.00033, 0.00033, 0.00033, 0.00035, 0.00035, 0.0004, 0.0004, 0.00034])

print(len(x))
print(len(y))
print(len(Dy))
# Create a figure and a set of subplots
fig, (ax1, ax2) = pylab.subplots(2, sharex=True, gridspec_kw={'hspace': 0.3, 'height_ratios': [2.5, 1]}, figsize=(10,8))

# Main plot on ax1
ax1.errorbar(x, y, Dy, linestyle='', color='black', marker='.')
ax1.set_ylabel(r'$\sin( \theta_i) - \sin( \theta_{d, m})$')
ax1.minorticks_on()

# define the function (linear, in this example)
def ff(xx, a):
    return xx/a

# define the initial values (STRICTLY NEEDED!!!)
init=(8.3*1e-7)

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
print('d = ', pars[0], ' +/- ', numpy.sqrt(covm[0,0]))

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
ax2.set_xlabel('$m*\lambda$ [m]')
ax2.set_ylabel('Residui normalizzati')
ax2.minorticks_on()

# show the plot
pylab.show()

# save the figure
fig.savefig('fit_guadagnoeresidui.png', dpi=300)
