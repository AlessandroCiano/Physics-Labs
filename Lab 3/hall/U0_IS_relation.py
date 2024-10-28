import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import correlated_values
import pickle

#U_0 I_S relation I_S [mA] and U_0 [V]
I_S = np.array([3, 4, 5, 6, 7])
err_I_S = np.sqrt((0.5*1e-2*I_S)**2 + (np.full(len(I_S), 0.02))**2)
U_0 = np.array([2.155, 2.877, 3.630, 4.37, 5.14])
err_U_0 = np.sqrt((0.5*1e-2*U_0)**2 + (np.full(len(U_0), 0.002))**2)

#make a fit of U_0 versus I_S
def linear(x, a, b):
    return a * x + b

#iterative fit the hall coefficient
err_eff = err_U_0
for i in range(3):
    params, covariance = curve_fit(linear, I_S, U_0, sigma=err_eff)
    errors = np.sqrt(np.diag(covariance))
    err_eff = np.sqrt((linear(I_S, *params) - U_0)**2 + err_U_0**2)

plt.errorbar(I_S, U_0, yerr=err_U_0, xerr=err_I_S, fmt='kx', label='Messwerte')
plt.plot(I_S, linear(I_S, *params), 'r-', label='Fit')
plt.xlabel('I_S [mA]')
plt.ylabel('U_0 [V]')
plt.title('Arbeit macht frei')
plt.show()

Rw = ufloat(params[0], errors[0])
#save Rw with pickle
with open('Rw.pkl', 'wb') as f:
    pickle.dump(Rw, f)


