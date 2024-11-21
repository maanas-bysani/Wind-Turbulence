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
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import chisquare



plt.rcParams["figure.figsize"] = (14,8)


# Parameters
time = 15  # in minutes
nrows = (time * 60) + 1
selected_csv = '3'


def linear(p, x):
     m, c = p
     return m*x + c

def gaussian(x, amp, mu, sigma):
    """Gaussian function for curve fitting."""
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def ln(p, x):
    a, b = p
    return a * np.log(x) + b

def horizontal_line(params, x):
    c = params
    return c * np.ones_like(x)


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


    # gaussian method

    column_of_351 = 4

    # calibrating 351 using stool
    only_351_df = selected_columns.iloc[:, column_of_351]
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
    all_but_351_df = selected_columns.drop(selected_columns.columns[[column_of_351]], axis = 1)
    # print("all_but_351_df")
    # print(all_but_351_df)
    
    # print("---" *30)
    calibrated_df_all_but_351 = pd.DataFrame()
    calibrated_df_all_but_351_min = pd.DataFrame()
    calibrated_df_all_but_351_max = pd.DataFrame()

    for i in range(len(all_but_351_df.columns)):
        calibrated_df_all_but_351[all_but_351_df.columns[i]] = (all_but_351_df.iloc[:, i] * delta_gradient_list[i]) + delta_intercept_list[i]
    # print(calibrated_df_all_but_351)

    for i in range(len(all_but_351_df.columns)):
        calibrated_df_all_but_351_min[all_but_351_df.columns[i]] = (all_but_351_df.iloc[:, i] * (delta_gradient_list[i] - delta_gradient_error_list[i])) + (delta_intercept_list[i] - delta_intercept_error_list[i])
    # print(calibrated_df_all_but_351_min)

    for i in range(len(all_but_351_df.columns)):
        calibrated_df_all_but_351_max[all_but_351_df.columns[i]] = (all_but_351_df.iloc[:, i] * (delta_gradient_list[i] + delta_gradient_error_list[i])) + (delta_intercept_list[i] + delta_intercept_error_list[i])
    # print(calibrated_df_all_but_351_max)


    calibrated_df_all = pd.concat([calibrated_df_all_but_351.iloc[:, :column_of_351], calibrated_351_df, calibrated_df_all_but_351.iloc[:, column_of_351:]], axis = 1)
    calibrated_df_all_min = pd.concat([calibrated_df_all_but_351_min.iloc[:, :column_of_351], calibrated_351_df_min, calibrated_df_all_but_351_min.iloc[:, column_of_351:]], axis = 1)
    calibrated_df_all_max = pd.concat([calibrated_df_all_but_351_max.iloc[:, :column_of_351], calibrated_351_df_max, calibrated_df_all_but_351_max.iloc[:, column_of_351:]], axis = 1)


    # doing histogram + gaussian fits on mean, then min, then max

    amp_list_mean, mu_list_mean, sigma_list_mean, mu_error_list_mean, sigma_list_error_mean = [], [], [], [], []

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
        sigma_list_error_mean.append(np.sqrt(cov[2,2]))

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

    amp_list_min, mu_list_min, sigma_list_min, mu_error_list_min, sigma_list_error_min = [], [], [], [], []

    bins = 100
    plt.figure()
    for i, column in enumerate(calibrated_df_all_min.columns, start=1):
        counts, bins_location = np.histogram(calibrated_df_all_min[column], bins=bins)
        counts[0] = 0
        bin_midpoints = 0.5 * (bins_location[1:] + bins_location[:-1])

        a_guess, m_guess, sig_guess = np.max(counts), 2.5, 0.1
        p0 = [a_guess, m_guess, sig_guess]

        fit, cov = op.curve_fit(gaussian, bin_midpoints, counts, p0, maxfev=100000)

        amp_list_min.append(fit[0])
        mu_list_min.append(fit[1])
        sigma_list_min.append(fit[2])
        mu_error_list_min.append(np.sqrt(cov[1,1]))
        sigma_list_error_min.append(np.sqrt(cov[2,2]))

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

    amp_list_max, mu_list_max, sigma_list_max, mu_error_list_max, sigma_list_error_max = [], [], [], [], []

    bins = 100
    plt.figure()
    for i, column in enumerate(calibrated_df_all_max.columns, start=1):
        counts, bins_location = np.histogram(calibrated_df_all_max[column], bins=bins)
        counts[0] = 0
        bin_midpoints = 0.5 * (bins_location[1:] + bins_location[:-1])

        a_guess, m_guess, sig_guess = np.max(counts), 2.5, 0.1
        p0 = [a_guess, m_guess, sig_guess]

        fit, cov = op.curve_fit(gaussian, bin_midpoints, counts, p0, maxfev=100000)

        amp_list_max.append(fit[0])
        mu_list_max.append(fit[1])
        sigma_list_max.append(fit[2])
        mu_error_list_max.append(np.sqrt(cov[1,1]))
        sigma_list_error_max.append(np.sqrt(cov[2,2]))

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


    # test plot for #348

    plt.figure(figsize = (10,5))
    x_dummy = np.arange(0, 6, 0.001)
    chosen_number = 3 #2 without 240 in dataset
    plt.plot(x_dummy, gaussian(x_dummy, amp_list_min[chosen_number] * 1.5, mu_list_min[chosen_number], sigma_list_min[chosen_number]), label = 'Min = {0:.2f}' .format(mu_list_min[chosen_number]), color = 'red', ls = 'dashed')
    plt.plot(x_dummy, gaussian(x_dummy, amp_list_mean[chosen_number] * 1.5, mu_list_mean[chosen_number], sigma_list_mean[chosen_number]), label = 'Mean = {0:.2f}' .format(mu_list_mean[chosen_number]), color = 'black')
    plt.plot(x_dummy, gaussian(x_dummy, amp_list_max[chosen_number] * 1.5, mu_list_max[chosen_number], sigma_list_max[chosen_number]), label = 'Max = {0:.2f}' .format(mu_list_max[chosen_number]), color = 'green', ls = 'dashed')

    counts, bins_location = np.histogram(calibrated_df_all.iloc[:, chosen_number], bins=bins)
    counts[0] = 0
    bin_midpoints = 0.5 * (bins_location[1:] + bins_location[:-1])

    a_guess, m_guess, sig_guess = np.max(counts), 2.5, 0.1
    p0 = [a_guess, m_guess, sig_guess]

    fit, cov = op.curve_fit(gaussian, bin_midpoints, counts, p0, maxfev=100000)
    plt.stairs(counts, bins_location, label='Data (Mean)', alpha = 0.5, color = 'black')
    plt.title("Gaussian Fit Example for Detector #" + str(calibrated_df_all_max.columns[chosen_number]).split('[')[0] + " (100 bins)")
    # plt.title(calibrated_df_all_max.columns[chosen_number])
    plt.ylabel('Frequency', fontsize = 12)
    plt.xlabel('Calibrated Speed (m/s)', fontsize = 12)
    plt.legend()
    # plt.savefig(f"final\\Detector Calibration and Gaussian Fits.png", dpi=1200)
    plt.show()



    # so our mean is in mu_list_mean, max mean is in mu_list_max, min mean is in mu_list_min
    mu_plus_error = np.array(mu_list_max) - np.array(mu_list_mean)
    mu_minus_error = np.array(mu_list_mean) - np.array(mu_list_min)
    mu_avg_error = (mu_plus_error + mu_minus_error) / 2
    # print(mu_avg_error)

    # expect std dev to be same for all gaussians so just use avg
    sigma_avg = (np.array(sigma_list_mean) + np.array(sigma_list_max) + np.array(sigma_list_min)) / 3

    # print("sigma_list_mean, sigma_list_max, sigma_list_min")
    # print(sigma_list_mean, sigma_list_max, sigma_list_min)
    sigma_err_avg = (np.array(sigma_list_error_mean) + np.array(sigma_list_error_max) + np.array(sigma_list_error_min)) / 3
    
    

    # final plots with odr fit


    # x_dummy = [2, 1, 3, 4, 5, 6, 7]
    distances = [113, 63, 155, 193, 235, 297, 389]
    # distances = [63, 155, 193, 235, 297, 389]
    distances = np.array(distances)/100
    dist_errors = 3/100
    x_max = np.array(distances) + dist_errors
    x_min = np.array(distances) - dist_errors



    model_ln = Model(ln)
    data_ln = RealData(distances[1:], mu_list_mean[1:], sx=dist_errors, sy=mu_avg_error[1:])

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

    x_fit_ln = np.linspace(min(distances[1:]), max(distances[1:]), 100)
    y_fit_ln = ln([a, b], x_fit_ln)

    gradient = a
    gradient_error = a_err
    gradient_error_sd = a_err_sd
    intercept = b
    intercept_error = b_err
    intercept_error_sd = b_err_sd
    # print(gradient_error, gradient_error_sd)
    # print(intercept_error, intercept_error_sd)

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

    # print(z1, z2, z3, z4)
    print("z_0_min, z_0, z_0_max")
    print(z_0_min, z_0, z_0_max)
    print(z_0_max - z_0)
    print(z_0 - z_0_min)

    y_min =  (gradient - gradient_error) * np.log(np.array(x_fit_ln)) + (intercept - intercept_error)
    y_max =  (gradient + gradient_error) * np.log(np.array(x_fit_ln)) + (intercept + intercept_error)

    y_min_sd =  (gradient - a_err_sd) * np.log(np.array(x_fit_ln)) + (intercept - b_err_sd)
    y_max_sd =  (gradient + a_err_sd) * np.log(np.array(x_fit_ln)) + (intercept + b_err_sd)


    plt.figure(figsize = (10,5))
    plt.errorbar(distances[1:], mu_list_mean[1:], \
                xerr = dist_errors,\
                yerr = mu_avg_error[1:], \
                capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
                color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data (Mean of Calibrated Gaussian)')
    plt.plot(x_fit_ln, y_fit_ln, label = 'ODR Fit', color = 'black')
    plt.errorbar(distances[0], mu_list_mean[0], xerr = dist_errors, yerr = mu_avg_error[0], fmt = '.',  color = 'red', label = 'Detector #240 - Outlier')
    plt.xscale('log', base=np.exp(1))
    xtick_values = []
    xtick_values_ln = []
    for i in np.arange(-0.5, 1.5, 0.25):
        xtick_values_ln.append(np.exp(i))
        xtick_values.append(f"{i:.2f}")
    plt.xticks(xtick_values_ln, xtick_values)
    plt.text(np.exp(1), 2.57, "Gradient = {0:.2f} \u00b1 {1:.2f} (Std. Dev.) \u00b1 {2:.2f} (Cov.) \nIntercept = {3:.2f} \u00b1 {4:.2f} (Std. Dev.) \u00b1 {5:.2f} (Cov.)" \
                    .format(gradient, gradient_error_sd, gradient_error, intercept, intercept_error_sd, intercept_error), bbox = dict(facecolor = 'white'), \
                        size = 'medium', ha = 'center', va = 'center')
    plt.text(np.exp(1), 2.44, "$\Rightarrow$ u* = {0:.2f} \u00b1 {1:.2f} m/s \n & z\u0305\u2080 = {2:.2f} mm; range: {3:.2f} \u2194 {4:.2f} mm" \
                    .format(gradient*0.4, gradient_error_sd*0.4, z_0, z_0_min, z_0_max), bbox = dict(facecolor = 'white'), \
                        size = 'medium', ha = 'center', va = 'top')

    plt.fill_between(x_fit_ln, y_min, y_max, alpha = 0.7, label = 'Uncertainty (Cov.)', color='grey')
    plt.fill_between(x_fit_ln, y_min_sd, y_min, alpha = 0.3, color='grey')
    plt.fill_between(x_fit_ln, y_max, y_max_sd, alpha = 0.3, label = 'Uncertainty (Std. Dev.)', color='grey')

    plt.title('Wind Profile & ODR Fit', fontsize = 18)
    plt.xlabel('ln(height) (arb. units)', fontsize = 12, labelpad=1.2)
    # plt.ylabel(r'ū ($\dfrac{m}{s}$)', fontsize = 12, labelpad=-1)
    plt.ylabel('ū (m/s)', fontsize = 12, labelpad=-1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plt.savefig("final\\Wind Profile Final v3", dpi=1200)
    plt.show(block=block)



    # u star plot

    model_h = Model(horizontal_line)
    data_h = RealData(distances[1:], sigma_avg[1:])

    initial_guess_h = [np.mean(sigma_avg)]

    odr_h = ODR(data_h, model_h, beta0=initial_guess_h)
    output_h = odr_h.run()

    y_value = output_h.beta[0]
    y_value_sd = output_h.sd_beta[0]
    y_value_cov = output_h.cov_beta[0,0]

    print("Fitted horizontal line y =", y_value, "+-", y_value_sd, 'or +-', y_value_cov)


    x_fit_h = np.linspace(min(distances[1:]), max(distances[1:]), 100)
    y_fit_h = horizontal_line([y_value], x_fit_h)
    

    r2 = r2_score(sigma_avg[1:], horizontal_line([y_value], distances[1:]))
    print("R^2 score:", r2)

    chisquared = chisquare(sigma_avg[1:], horizontal_line([y_value], distances[1:]))
    print("c2", chisquared)

    plt.figure(figsize = (10,5))
    plt.errorbar(distances[1:], sigma_avg[1:], \
                xerr = dist_errors,\
                yerr = sigma_err_avg[1:], \
                capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
                color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data (Standard Deviation of Calibrated Gaussian)')
    plt.plot(x_fit_h, y_fit_h, label = 'ODR Fit', color = 'black')


    y_min =  (y_value - y_value_cov) 
    y_max =  (y_value + y_value_cov) 

    y_min_sd =  (y_value - y_value_sd)  
    y_max_sd =  (y_value + y_value_sd) 


    plt.text(3.2, .70, "u* = {0:.2f} \u00b1 {1:.2f} (Std. Dev.) \u00b1 {2:.2f} (Cov.)" \
                    .format(y_value, y_value_sd, y_value_cov), bbox = dict(facecolor = 'white'), \
                        size = 'medium', ha = 'center', va = 'center')

    plt.text(3.2, .67, "R-squared = {0:.2f} & p-value from Chi-squared = {1:.2f}" \
                    .format(r2, chisquared[1]), bbox = dict(facecolor = 'white'), \
                        size = 'medium', ha = 'center', va = 'center')

    plt.fill_between(x_fit_h, y_min_sd, y_max_sd, alpha = 0.7, label = 'Uncertainty (Std. Dev.)', color='grey')
    # plt.fill_between(x_fit_h, y_min, y_min_sd, alpha = 0.3, color='grey')
    # plt.fill_between(x_fit_h, y_max_sd, y_max, alpha = 0.3, label = 'Uncertainty (Cov.)', color='grey')

    plt.title('u* vs height & Linear Fit', fontsize = 18)
    plt.xlabel('Height (m)', fontsize = 12, labelpad=1.2)
    plt.ylabel('u* (m/s)', fontsize = 12, labelpad=-1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plt.savefig("final\\u-star plot", dpi=1200)
    plt.show(block=block)


analysis(block = True)