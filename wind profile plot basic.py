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


plt.rcParams["figure.figsize"] = (14,8)


# Parameters
time = 15  # in minutes
nrows = (time * 60) + 1
selected_csv = '3_2'


def linear(p, x):
     m, c = p
     return m*x + c

def gaussian(x, amp, mu, sigma):
    """Gaussian function for curve fitting."""
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def analysis(block = False, filename = selected_csv):

    path = 'C:\\Users\Maanas\Documents\GitHub\Wind-Turbulence\\final\\'

    # detectors calibration data
    delta_gradient_list = np.loadtxt(path+'no delta\\gradient_list.txt')
    delta_intercept_list = np.loadtxt(path+'no delta\\intercept_list.txt')

    delta_gradient_error_list = np.loadtxt(path+'no delta\\gradient_error_list.txt')
    delta_intercept_error_list = np.loadtxt(path+'no delta\\intercept_error_list.txt')

    # stool data
    stool_data = np.loadtxt(path+'351 Calibration with errors.txt')
    stool_gradient = stool_data[0]
    stool_intercept = stool_data[1]
    stool_gradient_error = stool_data[2]
    stool_intercept_error = stool_data[3]

    # beach data

    file = 'C:\\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 2\Data\Session6 - 7th Nov\\' + str(filename) +'.csv'
    
    data = pd.read_csv(file, nrows=nrows+1)
    data = data.iloc[:, 1:]

    auto_averages = data.tail(1).iloc[:, 1::2] #speed
    # auto_averages = data.tail(1).iloc[:, 0::2] #temp
    auto_averages = auto_averages.apply(pd.to_numeric)
    auto_averages = auto_averages.values.flatten().tolist()

    # print("auto_averages")
    # print(auto_averages)
    # print("---" * 30)
 
    df = data.head(nrows)
    df.replace(to_replace='-', value = 0.0, inplace = True)
    df.fillna(0, inplace = True)
    df = df.apply(pd.to_numeric)

    # print(df)

    selected_columns = df.iloc[:, 1::2] # odd - speed
    # selected_columns = df.iloc[:, 0::2] # even - temp

    # plt.figure()
    # for i, column in enumerate(selected_columns.columns, start=1):
    #     plt.subplot(3, 3, i)
    #     plt.plot(df.index, df[column])
    #     plt.title(f"Detector: {column.split('[')[0]}")
    #     plt.ylabel('Speed (m/s)')
    #     plt.xlabel('Time (sec)')
    # plt.tight_layout()
    # # plt.show(block=block)
    # plt.pause(5)
    # plt.close()

    average_list = []

    for i in range(len(selected_columns.columns)):
        column_data = selected_columns.iloc[:, i]
        non_zero_data = column_data[column_data != 0]
        
        if len(non_zero_data) > 0:
            average = np.average(non_zero_data)
        else:
            average = np.nan
            print("Error Computing Average")

        average_list.append(average)

    # print(average_list)

# change this to reflect order of sensors:
    # x_dummy = [2, 1, 3, 4, 5, 6, 7]
    x_dummy = [63, 155, 193, 235, 297, 389]
    # x_dummy = [63, 155, 193, 235, 297, 389]

    # average_list = auto_averages

    # all_but_351_average = average_list[:3] + average_list[3+1:]
    # # print(all_but_351_average)

    # calibrated_average_speed_list = []
    # for i in range(len(delta_gradient_list)):
    #     calibrated_average_speed = ((delta_gradient_list[i]) * np.array(all_but_351_average[i])) + (delta_intercept_list[i])
    #     calibrated_average_speed_list.append(calibrated_average_speed)
    # print("mean list", calibrated_average_speed_list)

    # calibrated_average_speed_list_min = []
    # for i in range(len(delta_gradient_list)):
    #     calibrated_average_speed = ((delta_gradient_list[i] - delta_gradient_error_list[i]) * np.array(all_but_351_average[i])) + (delta_intercept_list[i] - delta_intercept_error_list[i])
    #     calibrated_average_speed_list_min.append(calibrated_average_speed)
    # print("min list", calibrated_average_speed_list_min)

    # calibrated_average_speed_list_max = []
    # for i in range(len(delta_gradient_list)):
    #     calibrated_average_speed = ((delta_gradient_list[i] + delta_gradient_error_list[i]) * np.array(all_but_351_average[i])) + (delta_intercept_list[i] + delta_intercept_error_list[i])
    #     calibrated_average_speed_list_max.append(calibrated_average_speed)
    # print("max list", calibrated_average_speed_list_max)


    # calibrated_average_351 = [(average_list.pop(3) * stool_gradient) + stool_intercept]
    # calibrated_average_351_min = [(average_list.pop(3) * (stool_gradient - stool_gradient_error)) + (stool_intercept - stool_intercept_error)]
    # calibrated_average_351_max = [(average_list.pop(3) * (stool_gradient + stool_gradient_error)) + (stool_intercept + stool_intercept_error)]
    # # print(calibrated_average_351_min)
    # # print(calibrated_average_351)
    # # print(calibrated_average_351_max)

    # print("---")
    # calibrated_average_speed_list_all = calibrated_average_speed_list[:3] + calibrated_average_351 + calibrated_average_speed_list[3:]
    # calibrated_average_speed_list_all_min = calibrated_average_speed_list_min[:3] + calibrated_average_351_min + calibrated_average_speed_list_min[3:]
    # calibrated_average_speed_list_all_max = calibrated_average_speed_list_max[:3] + calibrated_average_351_max + calibrated_average_speed_list_max[3:]

    # calibrated_average_speed_list_all_plus_error = np.array(calibrated_average_speed_list_all_max) - np.array(calibrated_average_speed_list_all)
    # print("plus error", calibrated_average_speed_list_all_plus_error)

    # calibrated_average_speed_list_all_minus_error = np.array(calibrated_average_speed_list_all) - np.array(calibrated_average_speed_list_all_min)
    # print("minus error", calibrated_average_speed_list_all_minus_error)


    # -x-


    # plt.figure()
    # # plt.scatter(np.log(x_dummy), calibrated_average_speed_list_all, color = 'red', label = 'Calculated Averages')
    # print("x_dummy", x_dummy)
    # print("calibrated_average_speed_list_all", calibrated_average_speed_list_all)
    # plt.errorbar(x_dummy, calibrated_average_speed_list_all, xerr = 5, \
    #              yerr = calibrated_average_speed_list_all_plus_error, \
    #                 fmt = 'x', color = 'red', label = 'Calculated Averages')
    # # plt.errorbar(x_dummy, calibrated_average_speed_list_all, xerr = 5, \
    # #              yerr = (calibrated_average_speed_list_all_minus_error, calibrated_average_speed_list_all_plus_error), \
    # #                 fmt = 'x', color = 'red', label = 'Calculated Averages')
    # plt.xscale('log', base=np.e)
    # # plt.scatter(0,0)
    # x_ticks = plt.gca().get_xticks()  # get the positions of the ticks
    # x_tick_labels = [f"{int(np.log(x_tick))}" if x_tick > 0 else '' for x_tick in x_ticks]
    # plt.gca().set_xticks(x_ticks)  # set the positions of the ticks
    # plt.gca().set_xticklabels(x_tick_labels)  # set the labels as the exponents

    # # plt.xticks(np.log(x_dummy), [int(c.split('[')[0]) for c in selected_columns.columns])
    # plt.title('Beach Measurements - All datapoints!!!')
    # plt.xlabel('log Height (cm)')
    # plt.ylabel('Speed (m/s)')
    # plt.xlim(np.exp(4), np.exp(6))
    # plt.legend()
    # plt.tight_layout()
    # plt.show(block=block)
    # plt.pause(5)
    # plt.close()

    # gaussian method
    # calibrating 351 using stool
    only_351_df = selected_columns.iloc[:, 3]
    # print("only_351_df")
    # print(only_351_df)

    calibrated_351_df = (stool_gradient * only_351_df) + stool_intercept
    calibrated_351_df_min = ((stool_gradient - stool_gradient_error) * only_351_df) + (stool_intercept - stool_intercept_error)
    calibrated_351_df_max = ((stool_gradient + stool_gradient_error) * only_351_df) + (stool_intercept + stool_intercept_error)
    # print("calibrated_351_df")
    # print(calibrated_351_df)

    # calibrating all but 351 using delta calibration values

    # print("selected_columns")
    # print(selected_columns)
    all_but_351_df = selected_columns.drop(selected_columns.columns[[3]], axis = 1)
    # print("all_but_351_df")
    # print(all_but_351_df)

    print("---" *30)
    calibrated_df_all_but_351 = pd.DataFrame()
    calibrated_df_all_but_351_min = pd.DataFrame()
    calibrated_df_all_but_351_max = pd.DataFrame()

    for i in range(len(all_but_351_df.columns)):
        calibrated_df_all_but_351[all_but_351_df.columns[i]] = (all_but_351_df.iloc[:, i] * delta_gradient_list[i]) + delta_intercept_list[i]
    print(calibrated_df_all_but_351)

    for i in range(len(all_but_351_df.columns)):
        calibrated_df_all_but_351_min[all_but_351_df.columns[i]] = (all_but_351_df.iloc[:, i] * (delta_gradient_list[i] - delta_gradient_error_list[i])) + (delta_intercept_list[i] - delta_intercept_error_list[i])
    print(calibrated_df_all_but_351_min)

    for i in range(len(all_but_351_df.columns)):
        calibrated_df_all_but_351_max[all_but_351_df.columns[i]] = (all_but_351_df.iloc[:, i] * (delta_gradient_list[i] + delta_gradient_error_list[i])) + (delta_intercept_list[i] + delta_intercept_error_list[i])
    print(calibrated_df_all_but_351_max)


    calibrated_df_all = pd.concat([calibrated_df_all_but_351.iloc[:, :3], calibrated_351_df, calibrated_df_all_but_351.iloc[:, 3:]], axis = 1)
    calibrated_df_all_min = pd.concat([calibrated_df_all_but_351_min.iloc[:, :3], calibrated_351_df_min, calibrated_df_all_but_351_min.iloc[:, 3:]], axis = 1)
    calibrated_df_all_max = pd.concat([calibrated_df_all_but_351_max.iloc[:, :3], calibrated_351_df_max, calibrated_df_all_but_351_max.iloc[:, 3:]], axis = 1)

    # print("calibrated_df_all")    
    # print(calibrated_df_all)

    # print("calibrated_df_all_min")    
    # print(calibrated_df_all_min)

    # print("calibrated_df_all_max")    
    # print(calibrated_df_all_max)

    # doing histogram + gaussian fits on mean, then min, then max

    amp_list_mean, mu_list_mean, sigma_list_mean, mu_error_list_mean = [], [], [], []

    bins = 100
    plt.figure()
    for i, column in enumerate(calibrated_df_all.columns, start=1):
        counts, bins_location = np.histogram(calibrated_df_all[column], bins=bins)
        counts[0] = 0
        bin_midpoints = 0.5 * (bins_location[1:] + bins_location[:-1])

        a_guess, m_guess, sig_guess = np.max(counts), 2.5, 0.1
        p0 = [a_guess, m_guess, sig_guess]

        fit, cov = op.curve_fit(gaussian, bin_midpoints, counts, p0, maxfev=100000)

        # print("The parameters")
        # print(fit)
        # print('--'*45)
        # print('The covariance matrix')
        # print(cov)

        amp_list_mean.append(fit[0])
        mu_list_mean.append(fit[1])
        sigma_list_mean.append(fit[2])
        mu_error_list_mean.append(np.sqrt(cov[1,1]))

        plt.subplot(3, 3, i)
        plt.stairs(counts, bins_location, label='Data')
        plt.plot(bin_midpoints, gaussian(bin_midpoints, *fit), color='black', label='Fit')
        plt.plot(bin_midpoints, (np.max(counts) / fit[0]) * gaussian(bin_midpoints, *fit), color='purple', label='Scaled')
        text = f"Mean = {round(fit[1], 2)} $\pm$ {round(np.sqrt(cov[1,1]), 2)}"
        plt.text(min(bin_midpoints) + 0.2, 1.3, text, bbox = dict(facecolor = 'white'))
        plt.title(column)
        plt.ylabel('Frequency')
        plt.xlabel('Speed (m/s)')
        plt.legend()
    
    plt.suptitle('Calibrated Histograms Mean')
    plt.tight_layout()
    plt.show(block=block)
    # plt.pause(10)
    # plt.close()
    print("Means")
    print("Amplitudes:", amp_list_mean)
    print("Means:", mu_list_mean)
    print("Means Errors:", mu_error_list_mean)
    print("Standard deviations:", sigma_list_mean)

    # # calibrated_average_speed_list_all = mu_list[:3] + calibrated_average_351 + mu_list[3:]
    # # calibrated_sigma_speed_list_all = sigma_list[:3] + [0.11334256406865317] + sigma_list[3:]
    # # print("calibrated_average_speed_list_all")
    # # print(calibrated_average_speed_list_all)
    # # print("calibrated_sigma_speed_list_all")
    # # print(calibrated_sigma_speed_list_all)


    # for minimum

    amp_list_min, mu_list_min, sigma_list_min, mu_error_list_min = [], [], [], []

    bins = 100
    plt.figure()
    for i, column in enumerate(calibrated_df_all_min.columns, start=1):
        counts, bins_location = np.histogram(calibrated_df_all_min[column], bins=bins)
        counts[0] = 0
        bin_midpoints = 0.5 * (bins_location[1:] + bins_location[:-1])

        a_guess, m_guess, sig_guess = np.max(counts), 2.5, 0.1
        p0 = [a_guess, m_guess, sig_guess]

        fit, cov = op.curve_fit(gaussian, bin_midpoints, counts, p0, maxfev=100000)

        # print("The parameters")
        # print(fit)
        # print('--'*45)
        # print('The covariance matrix')
        # print(cov)

        amp_list_min.append(fit[0])
        mu_list_min.append(fit[1])
        sigma_list_min.append(fit[2])
        mu_error_list_min.append(np.sqrt(cov[1,1]))

        plt.subplot(3, 3, i)
        plt.stairs(counts, bins_location, label='Data')
        plt.plot(bin_midpoints, gaussian(bin_midpoints, *fit), color='black', label='Fit')
        plt.plot(bin_midpoints, (np.max(counts) / fit[0]) * gaussian(bin_midpoints, *fit), color='purple', label='Scaled')
        text = f"Mean = {round(fit[1], 2)} $\pm$ {round(np.sqrt(cov[1,1]), 2)}"
        plt.text(min(bin_midpoints) + 0.2, 1.3, text, bbox = dict(facecolor = 'white'))
        plt.title(column)
        plt.ylabel('Frequency')
        plt.xlabel('Speed (m/s)')
        plt.legend()
    
    plt.suptitle('Calibrated Histograms Min')
    plt.tight_layout()
    plt.show(block=block)
    # plt.pause(10)
    # plt.close()

    print("Minimums")
    print("Amplitudes:", amp_list_min)
    print("Means:", mu_list_min)
    print("Means Errors:", mu_error_list_min)
    print("Standard deviations:", sigma_list_min)

    # for maximums


    amp_list_max, mu_list_max, sigma_list_max, mu_error_list_max = [], [], [], []

    bins = 100
    plt.figure()
    for i, column in enumerate(calibrated_df_all_max.columns, start=1):
        counts, bins_location = np.histogram(calibrated_df_all_max[column], bins=bins)
        counts[0] = 0
        bin_midpoints = 0.5 * (bins_location[1:] + bins_location[:-1])

        a_guess, m_guess, sig_guess = np.max(counts), 2.5, 0.1
        p0 = [a_guess, m_guess, sig_guess]

        fit, cov = op.curve_fit(gaussian, bin_midpoints, counts, p0, maxfev=100000)

        # print("The parameters")
        # print(fit)
        # print('--'*45)
        # print('The covariance matrix')
        # print(cov)

        amp_list_max.append(fit[0])
        mu_list_max.append(fit[1])
        sigma_list_max.append(fit[2])
        mu_error_list_max.append(np.sqrt(cov[1,1]))

        plt.subplot(3, 3, i)
        plt.stairs(counts, bins_location, label='Data')
        plt.plot(bin_midpoints, gaussian(bin_midpoints, *fit), color='black', label='Fit')
        plt.plot(bin_midpoints, (np.max(counts) / fit[0]) * gaussian(bin_midpoints, *fit), color='purple', label='Scaled')
        text = f"Mean = {round(fit[1], 2)} $\pm$ {round(np.sqrt(cov[1,1]), 2)}"
        plt.text(min(bin_midpoints) + 0.2, 1.3, text, bbox = dict(facecolor = 'white'))
        plt.title(column)
        plt.ylabel('Frequency')
        plt.xlabel('Speed (m/s)')
        plt.legend()
    
    plt.suptitle('Calibrated Histograms Max')
    plt.tight_layout()
    plt.show(block=block)
    # plt.pause(10)
    # plt.close()
    print("Maxiumums")
    print("Amplitudes:", amp_list_max)
    print("Means:", mu_list_max)
    print("Means Errors:", mu_error_list_max)
    print("Standard deviations:", sigma_list_max)


    # test for #518

    plt.figure(figsize = (10,5))
    x_dummy = np.arange(0, 6, 0.001)
    chosen_number = 2
    plt.plot(x_dummy, gaussian(x_dummy, amp_list_min[chosen_number] * 1.5, mu_list_min[chosen_number], sigma_list_min[chosen_number]), label = 'Min', color = 'red', ls = 'dashed')
    plt.plot(x_dummy, gaussian(x_dummy, amp_list_mean[chosen_number] * 1.5, mu_list_mean[chosen_number], sigma_list_mean[chosen_number]), label = 'Mean', color = 'black')
    plt.plot(x_dummy, gaussian(x_dummy, amp_list_max[chosen_number] * 1.5, mu_list_max[chosen_number], sigma_list_max[chosen_number]), label = 'Max', color = 'green', ls = 'dashed')
    counts, bins_location = np.histogram(calibrated_df_all.iloc[:, chosen_number], bins=bins)
    counts[0] = 0
    bin_midpoints = 0.5 * (bins_location[1:] + bins_location[:-1])

    a_guess, m_guess, sig_guess = np.max(counts), 2.5, 0.1
    p0 = [a_guess, m_guess, sig_guess]

    fit, cov = op.curve_fit(gaussian, bin_midpoints, counts, p0, maxfev=100000)
    plt.stairs(counts, bins_location, label='Data (Mean)', alpha = 0.5, color = 'black')
    plt.title("Gaussian Fit Example for Detector #351 (100 bins)")
    plt.ylabel('Frequency', fontsize = 12)
    plt.xlabel('Speed (m/s)', fontsize = 12)
    plt.legend()
    # plt.savefig(f"final\\Detector Calibration and Gaussian Fits.png", dpi=1200)
    plt.show()



    # so our mean is in mu_list_mean, max mean is in mu_list_max, min mean is in mu_list_min
    mu_plus_error = np.array(mu_list_max) - np.array(mu_list_mean)
    mu_minus_error = np.array(mu_list_mean) - np.array(mu_list_min)
    mu_avg_error = (mu_plus_error + mu_minus_error) / 2
    print(mu_avg_error)

    # expect std dev to be same for all gaussians so just use avg
    std_dev_avg = (np.array(sigma_list_mean) + np.array(sigma_list_max) + np.array(sigma_list_min)) / 3

    
    
    
    
    # final plots

    # x_dummy = [2, 1, 3, 4, 5, 6, 7]
    # distances = [113, 63, 155, 193, 235, 297, 389]
    distances = [63, 155, 193, 235, 297, 389]
    distances = np.array(distances)/100
    dist_errors = 3/100
    x_max = np.array(distances) + dist_errors
    x_min = np.array(distances) - dist_errors
    print(x_max - distances)
    
    plt.figure()

    plt.errorbar(x = distances, y = mu_list_mean, xerr = dist_errors, \
                yerr = mu_avg_error, \
                capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
                color = 'black', alpha = 1, ecolor = 'tab:blue', label='Calibrated Gaussian Averages')

    plt.xscale('log', base=np.e)
    # x_ticks = plt.gca().get_xticks()  # get the positions of the ticks
    # x_tick_labels = [f"{int(np.log(x_tick))}" if x_tick > 0 else '' for x_tick in x_ticks]
    # plt.gca().set_xticks(x_ticks)  # set the positions of the ticks
    # plt.gca().set_xticklabels(x_tick_labels)  # set the labels as the exponents
    # plt.xlim(np.exp(4), np.exp(6))
    plt.title('Beach Measurements')
    plt.xlabel('log Height in m (arb. units)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(5)
    plt.close()


    # odr fit

    def ln(p, x):
        a, b = p
        return a * np.log(x) + b

    model_ln = Model(ln)

    data_ln = RealData(distances, mu_list_mean, sx=dist_errors, sy=mu_avg_error)

    initial_params = [1., 1.]

    odr_ln = ODR(data_ln, model_ln, beta0=initial_params)

    output_ln = odr_ln.run()
    a, b = output_ln.beta
    a_err_sd, b_err_sd = output_ln.sd_beta
    a_err = output_ln.cov_beta[0,0]
    b_err = output_ln.cov_beta[1,1]
    print("Fitted parameters:")
    print(f"a = {a:.4f} ± {a_err:.4f}")
    print(f"b = {b:.4f} ± {b_err:.4f}")

    x_fit_ln = np.linspace(min(distances), max(distances), 100)
    y_fit_ln = ln([a, b], x_fit_ln)

    gradient = a
    gradient_error = a_err
    gradient_error_sd = a_err_sd
    intercept = b
    intercept_error = b_err
    intercept_error_sd = b_err_sd

    z_0 = np.exp(-intercept/gradient)*1000
    z_0_min = np.exp(-(intercept + intercept_error_sd)/(gradient - gradient_error_sd))*1000
    z_0_max = np.exp(-(intercept - intercept_error_sd)/(gradient + gradient_error_sd))*1000

    z_0_rel_err = np.sqrt(((gradient_error_sd/gradient)**2) + ((intercept_error_sd/intercept)**2))
    z_0_err = z_0_rel_err * z_0
    z_0_err_plus = np.abs(z_0_max - z_0)
    z_0_err_plus = np.abs(z_0 - z_0_min)
    z_0_avg_error = (z_0_err_plus + z_0_err_plus)/2
    z_0_avg = (z_0_max + z_0_min) / 2

    z1 = np.exp(-(intercept + intercept_error_sd)/(gradient - gradient_error_sd))*1000
    z2 = np.exp(-(intercept + intercept_error_sd)/(gradient + gradient_error_sd))*1000
    z3 = np.exp(-(intercept - intercept_error_sd)/(gradient - gradient_error_sd))*1000
    z4 = np.exp(-(intercept - intercept_error_sd)/(gradient + gradient_error_sd))*1000

    print(z1, z2, z3, z4)
    print("z_0_min, z_0, z_0_max")
    print(z_0_min, z_0, z_0_max)
    
    y_min =  (gradient - gradient_error) * np.log(np.array(x_fit_ln)) + (intercept - intercept_error)
    y_max =  (gradient + gradient_error) * np.log(np.array(x_fit_ln)) + (intercept + intercept_error)

    y_min_sd =  (gradient - a_err_sd) * np.log(np.array(x_fit_ln)) + (intercept - b_err_sd)
    y_max_sd =  (gradient + a_err_sd) * np.log(np.array(x_fit_ln)) + (intercept + b_err_sd)


    plt.figure(figsize = (10,5))
    plt.errorbar(distances, mu_list_mean, \
                xerr = dist_errors,\
                yerr = mu_avg_error, \
                capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
                color = 'black', alpha = 1, ecolor = 'tab:blue', label='Calibrated Gaussian Averages')
    plt.plot(x_fit_ln, y_fit_ln, label = 'ODR Fit', color = 'black')
    plt.xscale('log', base=np.exp(1))
    xtick_values = []
    xtick_values_ln = []
    for i in np.arange(-0.5, 1.5, 0.25):
        xtick_values_ln.append(np.exp(i))
        xtick_values.append(f"{i:.2f}")
    plt.xticks(xtick_values_ln, xtick_values)
    plt.text(np.exp(1), 2.57, "Gradient = {0:.2f} \u00b1 {1:.2f} \nIntercept = {2:.2f} \u00b1 {3:.2f}" \
                    .format(gradient, gradient_error, intercept, intercept_error), bbox = dict(facecolor = 'white'), \
                        size = 'medium', ha = 'center', va = 'center')
    plt.text(np.exp(1), 2.46, "$\Rightarrow$ u* = {0:.2f} \u00b1 {1:.2f} m/s \n & z\u0305\u2080 = {2:.2f} mm; range: {3:.2f} \u2194 {4:.2f} mm" \
                    .format(gradient*0.4, gradient_error*0.4, z_0, z_0_min, z_0_max), bbox = dict(facecolor = 'white'), \
                        size = 'medium', ha = 'center', va = 'top')






    # plt.text(np.exp(1), 2.5, "$\Rightarrow$ u* = {0:.2f} \u00b1 {1:.2f} m/s \n & $z_0$ = {2:.2f} +{3:.2f} -{4:.2f} mm" \
    #                 .format(gradient*0.4, gradient_error*0.4, z_0, (z_0_max - z_0), (z_0 - z_0_min)), bbox = dict(facecolor = 'white'), \
    #                     size = 'medium', ha = 'center', va = 'top')    

    # plt.text(np.exp(1), 2.5, r'$\substack{a \\ b}$', 
    #      bbox=dict(facecolor='white'), size='medium', ha='center', va='top')

    # plt.text(np.exp(1), 2.5, r'$\Rightarrow$ u* = {0:.2f} \u00b1 {1:.2f} m/s \n & $z_0$ = {2:.2f} \u00b1 $\\substack{1 // 2}$ mm' \
    #             .format(gradient*0.4, gradient_error*0.4, z_0), 
    #      bbox=dict(facecolor='white'), size='medium', ha='center', va='top')

    plt.fill_between(x_fit_ln, y_min, y_max, alpha = 0.7, label = 'Uncertainty (Covariance)', color='grey')
    plt.fill_between(x_fit_ln, y_min_sd, y_min, alpha = 0.3, color='grey')
    plt.fill_between(x_fit_ln, y_max, y_max_sd, alpha = 0.3, label = 'Uncertainty (Std Dev)', color='grey')

    plt.title('Wind Profile & ODR Fit', fontsize = 18)
    plt.xlabel('ln(height) (arb. units)', fontsize = 12, labelpad=1.2)
    plt.ylabel(r'ū ($\dfrac{m}{s}$)', fontsize = 12, labelpad=-1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("final\\Wind Profile Final", dpi=1200)
    plt.show(block=block)





    # # Define the ln model function
    # def ln(p, x):
    #     a, b = p
    #     return a * np.log(x) + b

    # # Model setup
    # model_ln = Model(ln)
    # data_ln = RealData(distances, mu_list_mean, sx=dist_errors, sy=mu_avg_error)

    # # Initial parameters for optimization
    # initial_params = [1.0, 1.0]
    # odr_ln = ODR(data_ln, model_ln, beta0=initial_params)
    # output_ln = odr_ln.run()

    # # Extract fitted parameters and their errors
    # a, b = output_ln.beta
    # a_err, b_err = output_ln.sd_beta

    # # Display fitted parameters
    # print("Fitted parameters:")
    # print(f"a = {a:.4f} ± {a_err:.4f}")
    # print(f"b = {b:.4f} ± {b_err:.4f}")

    # # Fit line for plotting
    # x_fit_ln = np.linspace(min(distances), max(distances), 100)
    # y_fit_ln = ln([a, b], x_fit_ln)

    # # Gradient and intercept calculations
    # gradient = a
    # gradient_error = a_err
    # intercept = b
    # intercept_error = b_err

    # # Calculation of z_0 and its error
    # z_0 = np.exp(-intercept / gradient) * 1000
    # z_0_max = np.exp(-(intercept + intercept_error) / (gradient - gradient_error)) * 1000
    # z_0_min = np.exp(-(intercept - intercept_error) / (gradient + gradient_error)) * 1000
    # z_0_err_plus = np.abs(z_0_max - z_0)
    # z_0_err_minus = np.abs(z_0 - z_0_min)
    # z_0_avg_error = (z_0_err_plus + z_0_err_minus) / 2

    # # Fit uncertainty band
    # y_min = (gradient - gradient_error) * np.log(x_fit_ln) + (intercept - intercept_error)
    # y_max = (gradient + gradient_error) * np.log(x_fit_ln) + (intercept + intercept_error)

    # # Plotting
    # plt.figure()
    # plt.errorbar(
    #     distances, mu_list_mean, xerr=dist_errors, yerr=mu_avg_error,
    #     capsize=2, elinewidth=1, capthick=1, fmt='x', color='black', 
    #     alpha=1, ecolor='tab:blue', label='Calibrated Gaussian Averages'
    # )
    # plt.plot(x_fit_ln, y_fit_ln, label='ODR Fit')
    # plt.xscale('log')  # log scale on the x-axis
    # plt.fill_between(x_fit_ln, y_min, y_max, alpha=0.5, color='grey', label='Uncertainty')

    # # Display fit parameters on plot
    # plt.text(
    #     0.8, 2.6, f"Gradient = {gradient:.2f} ± {gradient_error:.2f}\nIntercept = {intercept:.2f} ± {intercept_error:.2f}",
    #     bbox=dict(facecolor='white'), size='medium', ha='center'
    # )
    # plt.text(
    #     0.8, 2.4, r"$\Rightarrow$ u* = {0:.2f} ± {1:.2f} m/s \n & $z_0$ = {2:.2f} ± {3:.2f} mm"
    #             .format(gradient * 0.4, gradient_error * 0.4, z_0, z_0_avg_error),
    #     bbox=dict(facecolor='white'), size='medium', ha='center'
    # )

    # # Labels, title, and grid
    # plt.title('Wind Profile & ODR Fit', fontsize=18)
    # plt.xlabel('ln(height) (arb. units)', fontsize=12)
    # plt.ylabel(r'ū ($\dfrac{m}{s}$)', fontsize=12)
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # # plt.savefig("Wind Profile Final", dpi=1200)
    # plt.show()


















    # log_z = np.log(np.array(distances))
    # log_z_max = np.log(np.array(distances) + dist_errors)
    # log_z_min = np.log(np.array(distances) - dist_errors)
    # plus_error_log_z = log_z_max - log_z
    # minus_error_log_z = log_z - log_z_min
    # average_log_z_error = (plus_error_log_z + minus_error_log_z) / 2 

    # x_dummy = np.array(np.arange(min(log_z),max(log_z),0.01))

    # lin_model = Model(linear)

    # data = RealData(log_z, mu_list_mean, sx=average_log_z_error, sy=mu_avg_error)

    # odr = ODR(data, lin_model, beta0=[0., 1.])
    # out = odr.run()

    # x_fit = np.linspace(min(log_z), max(log_z), 100)
    # y_fit = linear(out.beta, x_fit)

    # gradient = out.beta[0]
    # intercept = out.beta[1]

    # gradient_error = out.sd_beta[0]
    # intercept_error = out.sd_beta[1]

    # # gradient_error_g = out_g.cov_beta[0,0]
    # # intercept_error_g = out_g.cov_beta[1,1]

    # plt.errorbar(log_z, mu_list_mean, xerr = average_log_z_error, \
    #             yerr = mu_avg_error, \
    #             capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
    #             color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data')

    # plt.plot(x_fit, y_fit, label = 'ODR Fit', color = 'black')
    # # y_min =  (gradient - gradient_error) * np.array(x_dummy) + (intercept - intercept_error)
    # # y_max =  (gradient + gradient_error) * np.array(x_dummy) + (intercept + intercept_error)
    # # plt.fill_between(x_dummy, y_min, y_max, alpha = 0.5, label = 'Uncertainty', color='grey')

    # plt.grid()
    # plt.title('Wind Profile')
    # plt.xlabel('ln(height) (arb. units)')
    # plt.ylabel(r'ū ($\dfrac{m}{s}$)')
    # plt.legend()
    # plt.show()

    # print("gradient, intercept, gradient_error, intercept_error")
    # print(gradient, intercept, gradient_error, intercept_error)



analysis(block = True)