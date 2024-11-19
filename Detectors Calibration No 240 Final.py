# -*- coding: utf-8 -*-
"""
@author: Maanas

no 240
"""

# imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as op
from pathlib import Path
from scipy.odr import *

plt.rcParams["figure.figsize"] = (10,6)

# Parameters
time = 3  # in minutes
nrows = (time * 60) + 1
device_number = [272, 284, 348, 518, 994]


def gaussian(x, amp, mu, sigma):
    """Gaussian function for curve fitting."""
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def linear2(x, m, c):
     return m*x + c

def linear(p, x):
     m, c = p
     return m*x + c


def analysis(device_number=device_number, file_path=None, block=False, bins=10, show_fit = True):
    """
    Analyzes data for devices, plots speed vs. time and histogram with Gaussian fit.
    
    Parameters:
        device_number (list): List of device numbers to process.
        file_path (Path): Base directory for data files.
        block (bool): Whether to block plots on show.
        bins (int): Number of bins for histogram.
    """
    # 351 calibration (stool)
    path = 'C:\\Users\Maanas\Documents\GitHub\Wind-Turbulence\\final\\'

    stool_data = np.loadtxt(path+'351 Calibration with errors.txt')
    # stool_data = np.loadtxt(path+'351 Calibration without errors.txt')
    stool_gradient = stool_data[0]
    stool_intercept = stool_data[1]
    stool_gradient_error = stool_data[2]
    stool_intercept_error = stool_data[3]
    print(stool_gradient, stool_intercept, stool_gradient_error, stool_intercept_error)

    # delta calibration - all detectors
    if file_path is None:
        file_path = Path('C:\\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 2\data p\Delta Calibration')

    df = pd.DataFrame()
    delta_df = pd.DataFrame()
    # df2 = pd.DataFrame() #high and low separate
    # df3 = pd.DataFrame() #high and low together
    calibrated_df = pd.DataFrame() #calibrated high and low together 
    error_df = pd.DataFrame() #ref error on calibrated high and low together 

    for device in device_number:
        file_low = file_path / f"{device} 1.csv"
        file_high = file_path / f"{device} 2.csv"
        data_low = pd.read_csv(file_low, nrows=nrows)
        data_high = pd.read_csv(file_high, nrows=nrows)
        
        column_names_low = data_low.columns
        column_names_high = data_high.columns

        df[f"Detector: {column_names_low[1]} low"] = data_low.iloc[:, 1]  # temperature
        df[f"Detector: {column_names_low[2]} low"] = data_low.iloc[:, 2]  # speed
        df[f"Detector: {column_names_low[3]} ref low"] = data_low.iloc[:, 3]  # temperature
        df[f"Detector: {column_names_low[4]} ref low"] = data_low.iloc[:, 4]  # speed
        
        df[f"Detector: {column_names_high[1]} high"] = data_high.iloc[:, 1]  # temperature
        df[f"Detector: {column_names_high[2]} high"] = data_high.iloc[:, 2]  # speed
        df[f"Detector: {column_names_high[3]} ref high"] = data_high.iloc[:, 3]  # temperature
        df[f"Detector: {column_names_high[4]} ref high"] = data_high.iloc[:, 4]  # speed

        difference_low = data_low.iloc[:, 4] - data_low.iloc[:, 2]
        difference_high = data_high.iloc[:, 4] - data_high.iloc[:, 2]

        delta_df[f"{device} low"] = difference_low
        delta_df[f"{device} high"] = difference_high

        # df2[f"{device} low"] = difference_low
        # df2[f"ref for {device} low"] = data_low.iloc[:, 4]
        # df2[f"{device} high"] = difference_high
        # df2[f"ref for {device} high"] = data_high.iloc[:, 4]
        # df3[f"{device}"] = pd.concat([difference_low, difference_high], ignore_index=True)
        # df3[f"ref for {device} high"] = pd.concat([data_low.iloc[:, 4], data_high.iloc[:, 4]], ignore_index=True)

        calibrated_low_ref = stool_gradient * data_low.iloc[:, 4] + stool_intercept
        calibrated_high_ref = stool_gradient * data_high.iloc[:, 4] + stool_intercept

        detector_speed_low = data_low.iloc[:, 2] #actual detector speed eg 272
        detector_speed_high = data_high.iloc[:, 2] #actual detector speed eg 272

        calibrated_df[f"ref for {device}"] = pd.concat([calibrated_low_ref, calibrated_high_ref], ignore_index=True)
        calibrated_df[f"{device}"] = pd.concat([detector_speed_low, detector_speed_high], ignore_index=True)

        # calibrated_low_ref_min = ((stool_gradient + stool_gradient_error)) * data_low.iloc[:, 4] + (stool_intercept + stool_intercept_error)
        calibrated_low_ref_max = ((stool_gradient - stool_gradient_error)) * data_low.iloc[:, 4] + (stool_intercept - stool_intercept_error)
        calibrated_low_plus_error = calibrated_low_ref_max - calibrated_low_ref
        # calibrated_low_minus_error = calibrated_low_ref -  calibrated_low_ref_min

        # calibrated_high_ref_min = ((stool_gradient + stool_gradient_error)) * data_high.iloc[:, 4] + (stool_intercept + stool_intercept_error)
        calibrated_high_ref_max = ((stool_gradient - stool_gradient_error)) * data_high.iloc[:, 4] + (stool_intercept - stool_intercept_error)
        calibrated_high_plus_error = calibrated_high_ref_max - data_high.iloc[:, 4]
        # calibrated_high_minus_error = data_high.iloc[:, 4] -  calibrated_high_ref_min

        error_df[f"error on ref for {device}"] = pd.concat([calibrated_low_plus_error, calibrated_high_plus_error], ignore_index=True)
        

    amp_list, mu_list, sigma_list = [], [], []

    plt.figure()
    for i, column in enumerate(delta_df.columns, start=1):
        counts, bins_location = np.histogram(delta_df[column], bins=bins)
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

        plt.subplot(3, 4, i)
        plt.stairs(counts, bins_location, label='Data')
        if show_fit is True:
            plt.plot(bin_midpoints, gaussian(bin_midpoints, *fit), color='black', label='Fit')
            # plt.plot(bin_midpoints, (np.max(counts) / fit[0]) * gaussian(bin_midpoints, *fit), color='purple', label='Scaled')
            text = f"Mean = {round(fit[1], 2)} $\pm$ {round(fit[2], 2)}"
            plt.text(min(bin_midpoints) + 0.2, 1.3, text, bbox = dict(facecolor = 'white'))

        plt.title(column + ' gaussian fits')
        plt.ylabel('Frequency')
        plt.xlabel('Speed (m/s)')
        plt.legend()
    
    plt.tight_layout()
    plt.show(block=block)
    # plt.pause(10)
    # plt.close()


    plt.figure()
    gradient_list = []
    intercept_list = []
    gradient_error_list = []
    intercept_error_list = []
    col_y_df = pd.DataFrame()
    col_x_df = pd.DataFrame()

    for i in range(1, 1 + len(calibrated_df.columns) // 2):
        j = 0
        col_x = 2 * (i-1) + 1
        col_y = 2 * (i-1)

    #     col_y_df[calibrated_df.columns[col_y]] = calibrated_df.iloc[:, col_y]
    #     col_x_df[calibrated_df.columns[col_x]] = calibrated_df.iloc[:, col_x]

    #     # print("col_y_df")
    #     # print(col_y_df)

    

    # for i in range(len(col_y_df.columns)):
    #     col_y_max = ((stool_gradient + stool_gradient_error) * col_y_df.iloc[:, i]) + (stool_intercept + stool_intercept_error)
    #     print("col_y_df.iloc[:, i], col_y_max")

    #     # print(col_y_df.iloc[:, i], col_y_min)

    #     col_y_min = ((stool_gradient - stool_gradient_error) * col_y_df.iloc[:, i]) + (stool_intercept - stool_intercept_error)
    #     print(col_y_df.iloc[:, i], col_y_min)
    #     col_y_error = col_y_max - col_y_df.iloc[:, i]
    #     col_y_error_2 = col_y_df.iloc[:, i] - col_y_min

    #     col_mean = (col_y_max + col_y_min) / 2
    #     print(col_y_df.iloc[:, i], col_mean)
        
        
    #     print("col_y_error", col_y_error)
    #     print("col_y_error_2",col_y_error_2)


    #     for i in range(len(calibrated_df.columns)):
    #         col_y_max = calibrated_df.iloc[:, col_y] * (stool_gradient + stool_gradient_error) + (stool_intercept + stool_intercept_error)
    #     print(col_y_max)
    #     for i in range(len(calibrated_df.columns)):
    #         col_y_max = calibrated_df.iloc[:, col_y] * (stool_gradient + stool_gradient_error) + (stool_intercept + stool_intercept_error)
    #     print(col_y_max)


        plt.figure(figsize = (10,6))
        # plt.subplot(3, 3, i)
        lin_model_1 = Model(linear)
        data_1 = RealData(calibrated_df.iloc[:, col_x], calibrated_df.iloc[:, col_y], sy = error_df.iloc[:, j])

        odr_1 = ODR(data_1, lin_model_1, beta0=[0., 0.])
        out_1 = odr_1.run()

        x_fit_1 = np.array(np.arange(min(calibrated_df.iloc[:, col_x]),max(calibrated_df.iloc[:, col_x]),0.001))
        y_fit_1 = linear(out_1.beta, x_fit_1)

        gradient_1 = out_1.beta[0]
        intercept_1 = out_1.beta[1]
        gradient_list.append(gradient_1)
        intercept_list.append(intercept_1)

        gradient_error_1 = out_1.sd_beta[0]
        intercept_error_1 = out_1.sd_beta[1]

        # gradient_error_1 = out_1.cov_beta[0,0]
        # intercept_error_1 = out_1.cov_beta[1,1]

        gradient_error_list.append(gradient_error_1)
        intercept_error_list.append(intercept_error_1)

        plt.errorbar(calibrated_df.iloc[:, col_x], calibrated_df.iloc[:, col_y], \
                    capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
                        color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data', ms = 1)
        plt.plot( x_fit_1, y_fit_1, label = 'ODR Fit', color = 'black')
        plt.ylabel('351 Speed (m/s)')
        plt.xlabel(f'{calibrated_df.columns[col_x]} Speed (m/s)')
        # plt.title(f"{calibrated_df.columns[col_x]}")
        plt.text(((min(x_fit_1) + max(x_fit_1)) / 2), 0.4, "m = {0:.1e} \u00b1 {1:.1e} \n c = {2:.1e} \u00b1 {3:.1e}" \
                .format(gradient_1, gradient_error_1, intercept_1, intercept_error_1), bbox = dict(facecolor = 'white'), size = 'medium')

        y_min_1 =  (gradient_1 - gradient_error_1) * np.array(x_fit_1) + (intercept_1 - intercept_error_1)
        y_max_1 =  (gradient_1 + gradient_error_1) * np.array(x_fit_1) + (intercept_1 + intercept_error_1)
        plt.fill_between(x_fit_1, y_min_1, y_max_1, alpha = 0.5, label = 'Uncertainty', color='grey')

        plt.grid()
        plt.legend(fontsize = 'medium')

        plt.title("Detector Calibration w.r.t. Reference Detector #351")
        plt.tight_layout()

        # plt.savefig(f"final\\{calibrated_df.columns[col_x]} Detector Calibration.png", dpi=1200)

        print('Params')
        print("---" * 30)
        out_1.pprint()
        j = j + 1

        plt.show(block = block)
    # plt.pause(10)
    # plt.close()

    print("gradient_list", gradient_list)
    print("intercept_list", intercept_list)

    # np.savetxt('final\\no delta\gradient_list.txt', gradient_list)
    # np.savetxt('final\\no delta\intercept_list.txt', intercept_list)
    # np.savetxt('final\\no delta\gradient_error_list.txt', gradient_error_list)
    # np.savetxt('final\\no delta\intercept_error_list.txt', intercept_error_list)


    # curve fit algorithm - no longer needed
    
    # plt.figure()
    # gradient_list = []
    # intercept_list = []
    # gradient_error_list = []
    # intercept_error_list = []

    # for i in range(1, 1 + len(calibrated_df.columns) // 2):
    #     col_x = 2 * (i-1) + 1
    #     col_y = 2 * (i-1)
    
    #     plt.subplot(3, 3, i)
    
    #     p0 = [0., 0.]
    #     fit, cov = op.curve_fit(linear2, calibrated_df.iloc[:, col_x], calibrated_df.iloc[:, col_y], p0, maxfev=100000)

    #     print("Gradient", fit[0], '+-', cov[0,0])
    #     print("Intercept", fit[1], '+-', cov[1,1])

    #     x_fit_1 = np.array(np.arange(min(calibrated_df.iloc[:, col_x]),max(calibrated_df.iloc[:, col_x]),0.001))
    #     y_fit_1 = linear2(x_fit_1, fit[0], fit[1])

    #     gradient_1 = fit[0]
    #     intercept_1 = fit[1]
    #     gradient_list.append(gradient_1)
    #     intercept_list.append(intercept_1)


    #     gradient_error_1 = cov[0,0]
    #     intercept_error_1 = cov[1,1]
    #     gradient_error_list.append(gradient_error_1)
    #     intercept_error_list.append(intercept_error_1)


    #     plt.errorbar(calibrated_df.iloc[:, col_x], calibrated_df.iloc[:, col_y], \
    #                 capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
    #                     color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data', ms = 1)
    #     plt.plot( x_fit_1, y_fit_1, label = 'Linear Fit', color = 'black')
    #     plt.ylabel('351 Speed (m/s)')
    #     plt.xlabel(f'{calibrated_df.columns[col_x]} Speed (m/s)')
    #     plt.title(f"{calibrated_df.columns[col_x]} Curve Fit")
    #     plt.text(((min(x_fit_1) + max(x_fit_1)) / 2), 0.4, "m = {0:.1e} \u00b1 {1:.1e} \nc = {2:.1e} \u00b1 {3:.1e}" \
    #             .format(gradient_1, gradient_error_1, intercept_1, intercept_error_1), bbox = dict(facecolor = 'white'), size = 'xx-small')

    #     y_min_1 =  (gradient_1 - gradient_error_1) * np.array(x_fit_1) + (intercept_1 - intercept_error_1)
    #     y_max_1 =  (gradient_1 + gradient_error_1) * np.array(x_fit_1) + (intercept_1 + intercept_error_1)
    #     plt.fill_between(x_fit_1, y_min_1, y_max_1, alpha = 0.5, label = 'Uncertainty', color='grey')

    #     plt.grid()
    #     plt.legend(fontsize = 'xx-small')

    #     plt.tight_layout()
    #     plt.suptitle("Detector Calibration w.r.t. Reference Detector #351")

    #     # plt.savefig('Delta Calibration.png', dpi=1200)

    # plt.show(block = block)
    # # plt.pause(10)
    # # plt.close()

    # print("gradient_list", gradient_list)
    # print("intercept_list", intercept_list)

    # # np.savetxt('no 240\\no delta\gradient_list.txt', gradient_list)
    # # np.savetxt('no 240\\no delta\intercept_list.txt', intercept_list)
    # # np.savetxt('no 240\\no delta\gradient_error_list.txt', gradient_error_list)
    # # np.savetxt('no 240\\no delta\intercept_error_list.txt', intercept_error_list)


analysis(block = True, show_fit = True)