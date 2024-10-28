import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from uncertainties import *

#calibration fit for determining the I_M-B relation
def linear(x, a, b):
    return a * x + b

I_M = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
B = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
err_I_M = np.sqrt((0.5*1e-2*I_M)**2 + (np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))**2)
err_B = np.sqrt((0.5*1e-2*B)**2 + (np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))**2)
err_eff = err_B
for i in range(3):
    params, covariance = curve_fit(linear, I_M, B, sigma=err_eff)
    errors = np.sqrt(np.diag(covariance))
    err_eff = np.sqrt((linear(I_M, *params) - B)**2 + err_B**2)

print('a =', params[0], '+-', errors[0])
plt.errorbar(I_M, B, yerr=err_B, xerr=err_I_M, fmt='kx', label='Messwerte')
plt.plot(I_M, linear(I_M, *params), 'r-', label='Fit')
plt.xlabel('I_M [mA]')
plt.ylabel('B [G]')
plt.show()





