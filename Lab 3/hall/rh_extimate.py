import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import correlated_values
import pickle

#read data from rh0.pkl and rh1.pkl
with open('rh0.pkl', 'rb') as f:
    r_h0, err_r_h0, n0, err_n0 = pickle.load(f)

with open('rh1.pkl', 'rb') as f:
    r_h1, err_r_h1, n1, err_n1 = pickle.load(f)

#concatenate data
r_h = np.concatenate((r_h0, r_h1))
err_r_h = np.concatenate((err_r_h0, err_r_h1))
n = np.concatenate((n0, n1))
err_n = np.concatenate((err_n0, err_n1))

#make a ponderate mean of r_h and n
r_h_mean = np.average(r_h, weights=1/err_r_h**2)
n_mean = np.average(n, weights=1/err_n**2)

print('r_h_mean =', r_h_mean, '+-', np.sqrt(1/np.sum(1/err_r_h**2)))
print('n_mean =', n_mean, '+-', np.sqrt(1/np.sum(1/err_n**2)))

#knowing that w, b, d are the dimendions of the sample and Rw is the resistance of the sample, calculate the conductivity and the mobility
w = ufloat(3.9, 0.1)*1e-3
b = ufloat(2.3, 0.1)*1e-3
d = ufloat(1.2, 0.1)*1e-3
#read Rw from pickle file
with open('Rw.pkl', 'rb') as f:
    Rw = pickle.load(f)

from scipy.constants import e
sigma = w/(Rw*b*d)
mu = sigma/(n_mean*e)

print('sigma =', sigma)
print('mu =', mu)


