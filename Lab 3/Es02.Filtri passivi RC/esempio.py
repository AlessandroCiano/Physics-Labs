import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# function used for fit
# parameters: x, array of points for function evaluation
#             value, parameter to be used as value
# return val: NumPy array with same length of x
def my_constant_func(x, value):
    returnval = np.full_like(x, value)
    return returnval

# open file
with open('network_analyzer.csv', newline='') as csvfile:

    # convert csv removing comments and empty lines
#    reader = csv.DictReader(filter(lambda row: (row[0]!='#' and row[0]!='\n'), csvfile))
    reader = csv.DictReader(filter(lambda row: (row[0]!='#' and row[0]!='\n' and row[0]!='\r'), csvfile))

    # transform enumerator to list
    data = list(reader)

 #   print data in reader
   # for row in reader:
   #     print(row['Frequency (Hz)'] + " " + row['Channel 2 Magnitude (dB)'] + "\n");

    # convert list of dictionaries to NumPy arrays, only for fields of interest
    freq = np.array([float(row['Frequency (Hz)']) for row in data])
    ampl = np.array([float(row['Channel 2 Magnitude (dB)']) for row in data])
    phase = np.array([float(row['Channel 2 Phase (deg)']) for row in data])

    # print converted values
    #for f, a in zip(freq, ampl):
    #    print(str(f) + " "+ str(a) + "\n");

    # create array of Y-error
    # TODO: please change 0.01 with the appropriate value (may be a function of the amplitude value).
    ampl_error = np.array([0.01 for value in ampl]);
    phase_error = np.array([0.01 for value in phase]);
'''
    # select elements to be used in fit:
    # cut only items 800 Hz
    # TODO: tune this selection
    freq_fit = freq[freq < 800.];
    ampl_fit = ampl[freq < 800.];
    ampl_error_fit = ampl_error[freq < 800.]

    # fit ampl-freq with errors ampl_error using my_contant_func, note absolute_sigma
    # starting parameters value=0dB
    params, params_covariance = optimize.curve_fit(my_constant_func, freq_fit, ampl_fit, p0=[0], sigma=ampl_error_fit, absolute_sigma=True)

    # print fitted values with errors obtained from diagonal elements of covariance matrix
    print("fitted values:\n")
    print("value: ", params[0], " +/- ", np.sqrt(params_covariance[0][0]), " dB\n")

    #print actual covariance matrix
    print("covariance matrix:\n");
    print(params_covariance, "\n")

    # evaluate function for points in fit and compute chi2 and ndf
    best_fit = my_constant_func(freq_fit, params[0])
    chi_square = (((ampl_fit - best_fit)/ampl_error_fit)**2).sum()
    print("chi2/ndf = ", chi_square, "/", len(freq_fit)-len(params));
'''

    # plot data with error bar and fitted function
fig, axs = plt.subplots(2)
fig.suptitle('RC circuit fit with constant')
axs[0].set_xscale('log') #log scale
axs[0].set_ylabel('Gain (dB)')
axs[0].set_xlabel('Frequency (Hz)')
axs[0].scatter(freq, ampl, label="Data")
axs[0].errorbar(freq, ampl, yerr=ampl_error, fmt="o")
 #   axs[0].plot(freq, my_constant_func(freq, params[0]),label='Fitted function', color="red")

    # plot residuals on fit range
axs[1].set_xscale('log') #log scale
axs[1].set_ylabel('Phase (Deg)')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].scatter(freq, phase, label="Ciccio")
axs[1].errorbar(freq, phase, yerr=phase_error, fmt="o")

plt.tight_layout() # axis labels may be outside plot without this
plt.show()

