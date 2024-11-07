# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:19:47 2024

@author: Maanas
"""

# imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as op
from pathlib import Path

plt.rcParams["figure.figsize"] = (10,8)

# Parameters
time = 5  # in minutes
nrows = (time * 60) + 1
device_number = [2, 3, 4, 5, 6, 7, 8, 9]


def gaussian(x, amp, mu, sigma):
    """Gaussian function for curve fitting."""
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def analysis(device_number=device_number, file_path=None, block=False, bins=50, show_fit = True):
    """
    Analyzes data for devices, plots speed vs. time and histogram with Gaussian fit.
    
    Parameters:
        device_number (list): List of device numbers to process.
        file_path (Path): Base directory for data files.
        block (bool): Whether to block plots on show.
        bins (int): Number of bins for histogram.
    """
    if file_path is None:
        file_path = Path('C:/Users/Maanas/OneDrive - Imperial College London/Blackboard/Lab/Cycle 2/Data/Session2 - 31st Oct/Tripod calibration/')

    df = pd.DataFrame()

    for device in device_number:
        file = file_path / f"{device} low.csv"
        data = pd.read_csv(file, nrows=nrows)
        column_names = data.columns

        df[f"Detector: {column_names[1]}"] = data.iloc[:, 1]  # temperature
        df[f"Detector: {column_names[2]}"] = data.iloc[:, 2]  # speed
    print(df)

    selected_columns = df.iloc[:, 1::2] # odd - speed
    # selected_columns = df.iloc[:, 0::2] # even - temp

    plt.figure()
    for i, column in enumerate(selected_columns.columns, start=1):
        plt.subplot(3, 3, i)
        plt.plot(df.index, df[column])
        plt.title(column.split('[')[0])
        plt.ylabel('Speed (m/s)')
        plt.xlabel('Time (sec)')
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(5)
    plt.close()


    amp_list, mu_list, sigma_list = [], [], []


    plt.figure()
    for i, column in enumerate(selected_columns.columns, start=1):
        counts, bins_location = np.histogram(df[column], bins=bins)
        bin_midpoints = 0.5 * (bins_location[1:] + bins_location[:-1])

        a_guess, m_guess, sig_guess = np.max(counts), np.median(bin_midpoints), 0.1
        p0 = [a_guess, m_guess, sig_guess]

        fit, cov = op.curve_fit(gaussian, bin_midpoints, counts, p0, maxfev=100000)

        # print("The parameters")
        # print(fit)
        # print('--'*45)
        # print('The covariance matrix')
        # print(cov)

        amp_list.append(fit[0])
        mu_list.append(fit[1])
        sigma_list.append(fit[2])

        plt.subplot(3, 3, i)
        plt.stairs(counts, bins_location, label='Data')
        if show_fit is True:
            plt.plot(bin_midpoints, gaussian(bin_midpoints, *fit), color='black', label='Fit')
            plt.plot(bin_midpoints, np.max(counts) / fit[0] * gaussian(bin_midpoints, *fit), color='purple', label='Scaled')
        plt.title(column.split('[')[0])
        plt.ylabel('Frequency')
        plt.xlabel('Speed (m/s)')
        plt.legend()
    
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(10)
    plt.close()

    if show_fit is True:
        print("Amplitudes:", amp_list)
        print("Means:", mu_list)
        print("Standard deviations:", sigma_list)

    if show_fit is True:
        plt.errorbar(device_number, mu_list, yerr=sigma_list, \
                     fmt = 'x', capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, \
                     color = 'black', alpha = 1, ecolor = 'black', label = 'Data - Random Error')

        plt.errorbar(device_number, mu_list, yerr=0.7, \
                     fmt = 'x', capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, \
                     color = 'black', alpha = .5, ecolor = 'tab:red', label = 'Data - Detector Error')

        plt.title('Detector Calibration')
        plt.xlabel('Device Number')
        plt.ylabel('Speed (m/s)')
        plt.xticks(device_number, [int(c.split(': ')[1].split()[0]) for c in selected_columns.columns])
        plt.legend()
        plt.tight_layout()
        plt.show(block=block)
        plt.pause(10)
        plt.close()

analysis(block = False, show_fit = True)

