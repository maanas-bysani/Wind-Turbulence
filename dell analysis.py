# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:25:29 2024

@author: Maanas
"""


# imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as op
from pathlib import Path
import scipy as sp
from scipy.odr import *


def linear(p, x):
     m, c = p
     return m*x + c

# dataset 4

log_z = [4.143134726391533, 5.043425116919247, 5.262690188904886, 5.459585514144159, 5.6937321388027, 5.963579343618446]

mean_u = [1.8284090478307335, 1.9677130971633128, 2.384673304691925, 2.474661661885532, 2.3661642884683762, 2.8545403096463255]

mean_u_error = [0.1406093383634902, 0.12310657118068324, 0.07215002722183605, 0.11334256406865317, 0.17732129549146136, 0.1112217595671865]

log_z_plus_error = [0.01574835696813892, 0.0064308903302903175, 0.005167970158442614, 0.004246290881451564, 0.0033613477027047978, 0.002567395505246317]

log_z_minus_error = [0.016000341346440905, 0.006472514505617255, 0.0051948168771041026, 0.004264398786457235, 0.003372684478639698, 0.0025740039951722693]

average_z_error = (np.array(log_z_plus_error) + np.array(log_z_minus_error)) / 2


x_dummy = np.array(np.arange(min(log_z),max(log_z),0.01))

lin_model = Model(linear)

data = RealData(log_z, mean_u, sx = average_z_error, sy = mean_u_error)

odr = ODR(data, lin_model, beta0=[0., 1.])
out = odr.run()

x_fit = np.linspace(log_z[0], log_z[-1], 100)
y_fit = linear(out.beta, x_fit)

gradient = out.beta[0]
intercept = out.beta[1]

# gradient_error = out.sd_beta[0]
# intercept_error = out.sd_beta[1]

gradient_error = out.cov_beta[0,0]
intercept_error = out.cov_beta[1,1]


plt.errorbar(log_z, mean_u, xerr=(log_z_minus_error, log_z_plus_error), yerr=mean_u_error, \
            capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
                color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data')
plt.plot(x_fit, y_fit, label = 'ODR Fit', color = 'black')
y_min =  (gradient - gradient_error) * np.array(x_dummy) + (intercept - intercept_error)
y_max =  (gradient + gradient_error) * np.array(x_dummy) + (intercept + intercept_error)
# print(y_min)
# print(y_max)

# plt.fill_between(x_dummy, y_min, y_max, alpha = 0.5, label = 'Uncertainty', color='grey')

plt.grid()
plt.title('Wind Profile')
plt.xlabel('ln(height) (arb. units)')
plt.ylabel(r'ū ($\dfrac{m}{s}$)')

plt.legend()
print("gradient, intercept, gradient_error, intercept_error")
print(gradient, intercept, gradient_error, intercept_error)
plt.savefig('Wind Profile')
plt.show()


