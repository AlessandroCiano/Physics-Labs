import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from uncertainties import *
import pickle

#calibration fit for determining the I_M-B relation
def linear(x, a, b):
    return a * x + b

I_M = np.array([0, 50, 100, 160, 200, 250, 300, 350, 400, 450, 500, 600, 700])
B = np.array([0.0, -54.0, -108.0, -172.8, -216.4, -271.3, -325.5, -380.2, -434.8, -489.9, -544.9, -656.0, -767.5])
err_I_M = np.sqrt((0.5*1e-2*I_M)**2 + (np.full(len(I_M), 2))**2)
err_B = np.full(len(B), 0.1)
#realize a table with the data and errors
data = np.array([I_M, B, err_I_M, err_B])
#save the table to a file
np.savetxt('magnetic_calibration_fit.txt', data.T, fmt='%1.4e', delimiter=' & ', newline=' \\\\\n')
#fit the data

err_eff = err_B
for i in range(3):
    params, covariance = curve_fit(linear, I_M, B, sigma=err_eff)
    errors = np.sqrt(np.diag(covariance))
    err_eff = np.sqrt((linear(I_M, *params) - B)**2 + err_B**2)


#save the parameters and covariance matrix to a file
with open('magnetic_calibration_fit.pkl', 'wb') as f:
    pickle.dump((params, covariance), f)

    
print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])
plt.errorbar(I_M, B, yerr=err_B, xerr=err_I_M, fmt='kx', label='Messwerte')
plt.plot(I_M, linear(I_M, *params), 'r-', label='Fit')
plt.xlabel('I_M [mA]')
plt.ylabel('B [G]')
#plt.show()


#zero calibration fit. I_S [mA] and U_H [mV]
I_S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
U_H = np.array([0, -0.2, -0.5, -0.8, -1.1, -1.4, -1.8, -2.1, -2.4, -2.7, -3.0])
err_I_S = np.sqrt((0.5*1e-2*I_S)**2 + (np.full(len(I_S), 0.1))**2)
err_U_H = np.sqrt((0.5*1e-2*U_H)**2 + (np.full(len(U_H), 0.1))**2)
err_eff = err_U_H
for i in range(3):
    params, covariance = curve_fit(linear, I_S, U_H, sigma=err_eff)
    errors = np.sqrt(np.diag(covariance))
    err_eff = np.sqrt((linear(I_S, *params) - U_H)**2 + err_U_H**2)



# Save the parameters and covariance matrix using pickle
with open('zero_calibration_fit.pkl', 'wb') as f:
    pickle.dump((params, covariance), f)


print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])
plt.errorbar(I_S, U_H, yerr=err_U_H, xerr=err_I_S, fmt='kx', label='Messwerte')
plt.plot(I_S, linear(I_S, *params), 'r-', label='Fit')
plt.xlabel('I_S [mA]')
plt.ylabel('U_H [mV]')
plt.show()

I_test = ufloat(1.00, np.sqrt((0.5*1e-2*1.00)**2 + (0.02)**2))
V_test = ufloat(717, np.sqrt((0.5*1e-2*717)**2 + (2)**2))
w = ufloat(3.9, 0.1)
Rs = V_test/I_test
pend = ufloat(params[0], errors[0])
print(pend*w/Rs) 





