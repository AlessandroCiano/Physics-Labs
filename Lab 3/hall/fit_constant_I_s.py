import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import correlated_values
import pickle

def linear(x, a, b):
    return a * x + b  

#obtain the fit parameters and covariance matrix of the  zero calibration fit
with open('zero_calibration_fit.pkl', 'rb') as f:
    params, covariance = pickle.load(f)
zero_pars = params
fit_pars_zero = correlated_values(params, covariance)

#obtain the fit parameters and covariance matrix of the magnetic calibration fit
with open('magnetic_calibration_fit.pkl', 'rb') as f:
    params, covariance = pickle.load(f)
mag_pars = params
fit_pars_mag = correlated_values(params, covariance)


#we are fitting the data at fixed I_S, I_M [mA] and U_H [mV] B [G = 1e-4 T]
I_S = ufloat(7.00, np.sqrt((0.5*1e-2*3.00)**2 + (0.02)**2))
I_M = np.array([0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
err_I_M = np.sqrt((0.5*1e-2*I_M)**2 + (np.full(len(I_M), 2))**2)
B = np.zeros(len(I_M))
U_H_m = np.array([-2.1, -16.7, -24.2, -31.6, -38.9, -46.3, -53.7, -61.2, -68.6, -76.2, -83.6, -91.1, -98.6, -106.2])

for i, x in enumerate(I_M):
    I_M_i = ufloat(x, err_I_M[i])
    B_err = linear(I_M_i, *fit_pars_mag)
    B[i]= B_err.nominal_value
    err_I_M[i] = B_err.std_dev

#initialize array U_H_0 of kength len(U_H) andd handle correlations with correlated_values
U_H = np.zeros(len(U_H_m))
err_U_H = np.zeros(len(U_H_m))
for i, x in enumerate(U_H_m):
    u = ufloat(x, 0.2)
    U_H_err = u - linear(I_S, *fit_pars_zero)
    U_H[i]= U_H_err.nominal_value
    err_U_H[i] = U_H_err.std_dev

plt.errorbar(B, U_H, yerr=err_U_H, xerr=err_I_M, fmt='.', label='Experimental data')
#iterative fit the hall coefficient U_H versus B
err_eff = err_U_H
for i in range(3):
    params_0, covariance_0 = curve_fit(linear, B, U_H, sigma=err_eff)
    errors = np.sqrt(np.diag(covariance))
    err_eff = np.sqrt((linear(B, *params) - U_H)**2 + err_U_H**2)

I_S0 = I_S
#we are fitting the data at fixed I_S = , I_M [mA] and U_H [mV]
I_S = ufloat(5.00, np.sqrt((0.5*1e-2*3.00)**2 + (0.02)**2))
I_M = np.array([0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
err_I_M = np.sqrt((0.5*1e-2*I_M)**2 + (np.full(len(I_M), 2))**2)
B = np.zeros(len(I_M))
U_H_m = np.array([-1.5, -12.0, -17.2, -22.5, -27.8, -33.1, -38.4, -43.8, -49.1, -54.4, -59.8, -65.2, -70.6, -75.9])

for i, x in enumerate(I_M):
    I_M_i = ufloat(x, err_I_M[i])
    B_err = linear(I_M_i, *fit_pars_mag)
    B[i]= B_err.nominal_value
    err_I_M[i] = B_err.std_dev

#initialize array U_H_0 of kength len(U_H) andd handle correlations with correlated_values
U_H = np.zeros(len(U_H_m))
err_U_H = np.zeros(len(U_H_m))
for i, x in enumerate(U_H_m):
    u = ufloat(x, 0.2)
    U_H_err = u - linear(I_S, *fit_pars_zero)
    U_H[i]= U_H_err.nominal_value
    err_U_H[i] = U_H_err.std_dev

plt.errorbar(B, U_H, yerr=err_U_H, xerr=err_I_M, fmt='.')
#iterative fit the hall coefficient U_H versus B
err_eff = err_U_H
for i in range(3):
    params_1, covariance_1 = curve_fit(linear, B, U_H, sigma=err_eff)
    errors = np.sqrt(np.diag(covariance))
    err_eff = np.sqrt((linear(B, *params) - U_H)**2 + err_U_H**2)

I_S1 = I_S
#we are fitting the data at fixed I_S, I_M [mA] and U_H [mV]
I_S = ufloat(3.00, np.sqrt((0.5*1e-2*3.00)**2 + (0.02)**2))
I_M = np.array([0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
err_I_M = np.sqrt((0.5*1e-2*I_M)**2 + (np.full(len(I_M), 2))**2)
B = np.zeros(len(I_M))
U_H_m = np.array([-0.8, -7.1, -10.3, -13.4, -16.6, -19.8, -23.0, -26.2, -29.4, -32.6, -35.8, -39.0, -42.2, -45.4])

for i, x in enumerate(I_M):
    I_M_i = ufloat(x, err_I_M[i])
    B_err = linear(I_M_i, *fit_pars_mag)
    B[i]= B_err.nominal_value
    err_I_M[i] = B_err.std_dev

#initialize array U_H_0 of kength len(U_H) andd handle correlations with correlated_values
U_H = np.zeros(len(U_H_m))
err_U_H = np.zeros(len(U_H_m))
for i, x in enumerate(U_H_m):
    u = ufloat(x, 0.2)
    U_H_err = u - linear(I_S, *fit_pars_zero)
    U_H[i]= U_H_err.nominal_value
    err_U_H[i] = U_H_err.std_dev

plt.errorbar(B, U_H, yerr=err_U_H, xerr=err_I_M, fmt='.')
#iterative fit the hall coefficient U_H versus B
err_eff = err_U_H
for i in range(3):
    params_2, covariance_2 = curve_fit(linear, B, U_H, sigma=err_eff)
    errors = np.sqrt(np.diag(covariance))
    err_eff = np.sqrt((linear(B, *params) - U_H)**2 + err_U_H**2)

I_S2 = I_S
#plot the data and the fits in an unique plot
plt.plot(B, linear(B, *params_0), 'r-', label='Fit I_S = 7 mA')
plt.plot(B, linear(B, *params_1), 'g-', label='Fit I_S = 5 mA')
plt.plot(B, linear(B, *params_2), 'b-', label='Fit I_S = 3 mA')
plt.xlabel('B [G]')
plt.ylabel('U_H [mV]')
plt.legend()
plt.show()

#calculate r_h and its error converting B to T
r_h = np.zeros(3)
err_r_h = np.zeros(3)
d = ufloat(1.2, 0.1)*1e-3
pend0 = ufloat(params_0[0], errors[0])
pend1 = ufloat(params_1[0], errors[0])
pend2 = ufloat(params_2[0], errors[0])
r_h0 = pend0*1e4*d/I_S0
r_h1 = pend1*1e4*d/I_S1
r_h2 = pend2*1e4*d/I_S2

r_h[0]= r_h0.nominal_value
r_h[1]= r_h1.nominal_value
r_h[2]= r_h2.nominal_value
err_r_h[0] = r_h0.std_dev
err_r_h[1] = r_h1.std_dev
err_r_h[2] = r_h2.std_dev

print('r_h =', r_h[0], '+-', err_r_h[0])
print('r_h =', r_h[1], '+-', err_r_h[1])
print('r_h =', r_h[2], '+-', err_r_h[2])

from scipy.constants import e
#calculate the charge carrier density
n = 1/(e*r_h)
err_n = err_r_h/(e*r_h**2)
print(n, err_n)

r_h_IScost = r_h
err_r_h_IScost = err_r_h
n_IScost = n
err_n_IScost = err_n

#save data with pickle
with open('rh1.pkl', 'wb') as f:
    pickle.dump((r_h_IScost, err_r_h_IScost, n_IScost, err_n_IScost), f)
