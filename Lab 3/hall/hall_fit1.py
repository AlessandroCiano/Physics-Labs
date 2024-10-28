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


I_M = ufloat(300, np.sqrt((0.5*1e-2*300)**2 + (2)**2))
B = linear(I_M, *fit_pars_mag)




#data at fixes I_M, I_S [mA] and U_H [mV
I_S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
U_H_m = np.array([0, -6.6, -13.2, -19.9, -26.6, -33.2, -39.8, -46.5, -53.1, -59.7, -66.3])
#err_I_S = np.sqrt((0.5*1e-2*I_S)**2 + (np.full(len(I_S), 0.02))**2)
#err_U_H = np.sqrt((0.5*1e-2*U_H_m)**2 + (np.full(len(U_H_m), 0.2))**2)
err_I_S = np.full(len(I_S), 0.001)
err_U_H = np.sqrt((np.full(len(U_H_m), 0.01))**2)
#initialize array U_H_0 of kength len(U_H) andd handle correlations with correlated_values
U_H = np.zeros(len(U_H_m))
for i, x in enumerate(U_H_m):
    u = ufloat(x, err_U_H[i])
    U_H_err = u - linear(I_S[i], *fit_pars_zero)
    U_H[i]= U_H_err.nominal_value
    err_U_H[i] = U_H_err.std_dev

#iterative fit the hall coefficient
err_eff = err_U_H   
for i in range(3):
    params, covariance = curve_fit(linear, I_S, U_H, sigma=err_eff, absolute_sigma=False)
    errors = np.sqrt(np.diag(covariance))
    err_eff = np.sqrt((linear(I_S, *params) - U_H)**2 + err_U_H**2)

print(err_U_H)
# calcola chi^2 ridotto
chi2 = np.sum(((U_H - linear(I_S, *params))**2)/err_eff**2)/(len(U_H)-2)
chi2_red = chi2/(len(U_H)-2)
print('chi2_red = ', chi2_red)


plt.errorbar(I_S, U_H, yerr=err_U_H, xerr=err_I_S, fmt='kx', label='Messwerte')
plt.plot(I_S, linear(I_S, *params), 'r-', label='Fit')
plt.xlabel('I_S [mA]')
plt.ylabel('U_H [mV]')
plt.title('B = ' + str(B))
plt.show()
print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])

#make a plot of residual
residuals = U_H - linear(I_S, *params)
normslized_residuals = residuals/err_U_H
plt.errorbar(I_S, normslized_residuals, yerr=1, fmt='kx', label='Residui normalizzati')
plt.show()

#r_h calculation
r_h = np.zeros(3)
err_r_h = np.zeros(3)
d = ufloat(1.2, 0.1)*1e-3
pend = ufloat(params[0], errors[0])
r_h_300  = pend*d/(B*1e-4)
r_h[0]= r_h_300.nominal_value
err_r_h[0] = r_h_300.std_dev




# do the same for I_M = 500 mA and I_M = 700 mA and label the plot with the relative B strenhth
I_M = ufloat(500, np.sqrt((0.5*1e-2*500)**2 + (2)**2))
B = linear(I_M, *fit_pars_mag)
I_S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
U_H_m = np.array([0, -10.8, -21.7, -32.6, -43.5, -54.4, -65.3, -76.2, -87.1, -97.9, -108.6])
err_I_S = np.sqrt((0.5*1e-2*I_S)**2 + (np.full(len(I_S), 0.02))**2)
err_U_H = np.sqrt((0.5*1e-2*U_H_m)**2 + (np.full(len(U_H_m), 0.2))**2)
U_H = np.zeros(len(U_H_m))
for i, x in enumerate(U_H_m):
    u = ufloat(x, err_U_H[i])
    U_H_err = u - linear(I_S[i], *fit_pars_zero)
    U_H[i]= U_H_err.nominal_value
    err_U_H[i] = U_H_err.std_dev

err_eff = err_U_H
for i in range(3):
    params, covariance = curve_fit(linear, I_S, U_H, sigma=err_eff, absolute_sigma=False)
    errors = np.sqrt(np.diag(covariance))
    err_eff = np.sqrt((linear(I_S, *params) - U_H)**2 + err_U_H**2)

# calcola chi^2 ridotto
chi2 = np.sum(((U_H - linear(I_S, *params))**2)/err_eff**2)/(len(U_H)-2)
chi2_red = chi2/(len(U_H)-2)
print('chi2_red = ', chi2_red)

plt.errorbar(I_S, U_H, yerr=err_U_H, xerr=err_I_S, fmt='kx', label='Messwerte')
plt.plot(I_S, linear(I_S, *params), 'r-', label='Fit')
plt.xlabel('I_S [mA]')
plt.ylabel('U_H [mV]')
plt.title('B = ' + str(B))
plt.show()
print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])

#r_h calculation
d = ufloat(1.2, 0.1)*1e-3
pend = ufloat(params[0], errors[0])
r_h_500  = pend*d/(B*1e-4)
r_h[1]= r_h_500.nominal_value
err_r_h[1] = r_h_500.std_dev

#do the same for I_M = 700 mA
I_M = ufloat(700, np.sqrt((0.5*1e-2*700)**2 + (2)**2))
B = linear(I_M, *fit_pars_mag)
I_S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
U_H_m = np.array([0, -15.1, -30.3, -45.6, -60.8, -76.0, -91.1, -106.3, -121.3, -136.4, -151.5])
err_I_S = np.sqrt((0.5*1e-2*I_S)**2 + (np.full(len(I_S), 0.02))**2)
err_U_H = np.sqrt((0.5*1e-2*U_H_m)**2 + (np.full(len(U_H_m), 0.2))**2)
U_H = np.zeros(len(U_H_m))
for i, x in enumerate(U_H_m):
    u = ufloat(x, err_U_H[i])
    U_H_err = u - linear(I_S[i], *fit_pars_zero)
    U_H[i]= U_H_err.nominal_value
    err_U_H[i] = U_H_err.std_dev

err_eff = err_U_H
for i in range(3):
    params, covariance = curve_fit(linear, I_S, U_H, sigma=err_eff)
    errors = np.sqrt(np.diag(covariance))
    err_eff = np.sqrt((linear(I_S, *params) - U_H)**2 + err_U_H**2)

# calcola chi^2 ridotto
chi2 = np.sum(((U_H - linear(I_S, *params))**2)/err_eff**2)/(len(U_H)-2)
chi2_red = chi2/(len(U_H)-2)
print('chi2_red = ', chi2_red)

plt.errorbar(I_S, U_H, yerr=err_U_H, xerr=err_I_S, fmt='kx', label='Messwerte')
plt.plot(I_S, linear(I_S, *params), 'r-', label='Fit')
plt.xlabel('I_S [mA]')
plt.ylabel('U_H [mV]')
plt.title('B = ' + str(B)) 
plt.show()
print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])


#r_h calculation
d = ufloat(1.2, 0.1)*1e-3
pend = ufloat(params[0], errors[0])
r_h_700  = pend*d/(B*1e-4)
r_h[2]= r_h_700.nominal_value
err_r_h[2] = r_h_700.std_dev
print(r_h, err_r_h)
from scipy.constants import e
n = 1/(e*r_h)
err_n = err_r_h/(e*r_h**2)
print(n, err_n)

r_h_magcost = r_h
err_r_h_magcost = err_r_h
n_magcost = n
err_n_magcost = err_n

#save data with pickle
with open('rh0.pkl', 'wb') as f:
    pickle.dump((r_h, err_r_h, n, err_n), f)
