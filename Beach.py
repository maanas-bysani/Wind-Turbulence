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
# device_number = [2, 3, 4, 5, 6, 7, 8, 9]

# filenames = ['Large Spacing Grass', 'Large Spacing Water', 'Low Spacing Grass', 'Low Spacing Water']
selected_csv = 4


def moving_average(df, window_size):
    """
    Calculate the simple moving average of a given dataset.
    """
    weights = np.ones(window_size) / window_size

    moving_avg_df = pd.DataFrame({col: np.convolve(df[col], weights, mode='valid') for col in df.columns})
    
    return moving_avg_df

def linear(p, x):
     m, c = p
     return m*x + c

def gaussian(x, amp, mu, sigma):
    """Gaussian function for curve fitting."""
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def analysis(block = False, filename = selected_csv):

# change this to reflect file name

    path = 'C:\\Users\Maanas\Documents\GitHub\Wind-Turbulence\data\\'

    # delta data
    delta_gradient_list = np.loadtxt(path+'gradient_list delta.txt')
    delta_intercept_list = np.loadtxt(path+'intercept_list delta.txt')

    delta_gradient_error_list = np.loadtxt(path+'gradient_error_list delta.txt')
    delta_intercept_error_list = np.loadtxt(path+'intercept_error_list delta.txt')

    # stool data
    stool_data = np.loadtxt(path+'351 calibration statistics.txt')
    stool_gradient = stool_data[0]
    stool_intercept = stool_data[1]
    stool_gradient_error = stool_data[2]
    stool_intercept_error = stool_data[3]

    # beach data

    file = 'C:\\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 2\Data\Session6 - 7th Nov\\' + str(filename) +'.csv'
    title_name = file.split('Nov\\')[1].split('.csv')[0]
    
    data = pd.read_csv(file, nrows=nrows+1)
    data = data.iloc[:, 1:-1]

    auto_averages = data.tail(1).iloc[:, 1::2]
    # auto_averages = data.tail(1).iloc[:, 0::2]
    auto_averages = auto_averages.apply(pd.to_numeric)

    print("auto_averages")
    print(auto_averages)
    print("---" * 30)
 
    df = data.head(nrows)
    df.replace(to_replace='-', value = 0.0, inplace = True)
    df.fillna(0, inplace = True)
    df = df.apply(pd.to_numeric)

    print(df)

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


    # moving_average_df = moving_average(df, 50)

    # plt.figure()
    # for i, column in enumerate(selected_columns.columns, start=1):
    #     plt.subplot(2, 3, i)
    #     plt.plot(moving_average_df.index, moving_average_df[column])
    #     plt.title(f"Detector: {column.split('[')[0]}")
    #     plt.ylabel('Speed (m/s)')
    #     plt.xlabel('Time (sec)')
    # plt.suptitle("Moving Average")
    # plt.tight_layout()
    # plt.show(block=block)
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

    print(average_list)

    # [300, 284, 348, 351, 179, 53, None, None, None, None]
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # data set:
    # [240, 284, 348, 351, 402, 994]

# change this to reflect order of sensors:
    x_dummy = [2, 1, 3, 4, 5, 6, 7]
    x_dummy = [113, 63, 155, 193, 235, 297, 389]

    # plt.figure()
    # plt.scatter(np.log(x_dummy), average_list, color = 'red', label = 'Calculated Averages')
    # plt.scatter(np.log(x_dummy), auto_averages, color = 'blue', label = 'Auto Averages')
    # # plt.scatter(0,0)
    # # plt.xticks(np.log(x_dummy), [int(c.split('[')[0]) for c in selected_columns.columns])
    # plt.title('Beach Measurements')
    # plt.xlabel('log(Height)')
    # plt.ylabel('Speed (m/s)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show(block=block)
    # plt.pause(5)
    # plt.close()

    # plt.figure()
    # plt.scatter(average_list, x_dummy, color = 'red', label = 'Calculated Averages')
    # plt.scatter(auto_averages, x_dummy, color = 'blue', label = 'Auto Averages')
    # # plt.scatter(0,0)
    # # plt.yticks(x_dummy, [int(c.split('[')[0]) for c in selected_columns.columns])
    # plt.title('Beach Measurements')
    # plt.ylabel('Height (cm)')
    # plt.xlabel('Speed (m/s)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show(block=block)
    # # plt.pause(5)
    # plt.close()

    # calibration stuff

    x_dummy = np.array(np.arange(0, 6, 0.001))
    y_plot = (stool_gradient) * np.array(x_dummy) + (stool_intercept)
    y_min_1 =  (stool_gradient - stool_gradient_error) * np.array(x_dummy) + (stool_intercept - stool_intercept_error)
    y_max_1 =  (stool_gradient + stool_gradient_error) * np.array(x_dummy) + (stool_intercept + stool_intercept_error)

    # plt.figure()
    # plt.plot( x_dummy, y_plot, label = 'ODR Fit', color = 'black')
    # plt.ylabel('Delta (m/s)')
    # plt.xlabel('Ref Speed (m/s)')
    # plt.title('ODR Fit')
    # plt.text(4, 2, "Gradient = {0:.2e} \u00b1 {1:.2e} \nIntercept = {2:.2e} \u00b1 {3:.2e}" \
    #         .format(stool_gradient, stool_gradient_error, stool_intercept, stool_intercept_error), bbox = dict(facecolor = 'white'))

    # plt.fill_between(x_dummy, y_min_1, y_max_1, alpha = 0.5, label = 'Uncertainty', color='grey')

    # plt.grid()
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


    all_but_351_average = average_list[:4] + average_list[4+1:]
    print(all_but_351_average)


    calibrated_average_speed_list = []
    for i in range(len(delta_gradient_list)):
        calibrated_average_speed = (delta_gradient_list[i]) * np.array(all_but_351_average[i]) + (delta_intercept_list[i]) + np.array(all_but_351_average[i])
        calibrated_average_speed_list.append(calibrated_average_speed)
    print(calibrated_average_speed_list)

    calibrated_average_speed_list_min = []
    for i in range(len(delta_gradient_list)):
        calibrated_average_speed = (delta_gradient_list[i] - delta_gradient_error_list[i]) * np.array(all_but_351_average[i]) + (delta_intercept_list[i] - delta_intercept_error_list[i]) + np.array(all_but_351_average[i])
        calibrated_average_speed_list_min.append(calibrated_average_speed)
    print(calibrated_average_speed_list_min)

    calibrated_average_speed_list_max = []
    for i in range(len(delta_gradient_list)):
        calibrated_average_speed = (delta_gradient_list[i] + delta_gradient_error_list[i]) * np.array(all_but_351_average[i]) + (delta_intercept_list[i] + delta_intercept_error_list[i]) + np.array(all_but_351_average[i])
        calibrated_average_speed_list_max.append(calibrated_average_speed)
    print(calibrated_average_speed_list_max)


    x_dummy = [2, 1, 3, 4, 5, 6, 7]
    x_dummy = [113, 63, 155, 193, 235, 297, 389]
    
    calibrated_average_351 = [average_list.pop(4) * stool_gradient + stool_intercept]
    print(calibrated_average_351)

    calibrated_average_351_min = [average_list.pop(4) * (stool_gradient - stool_gradient_error) + (stool_intercept - stool_intercept_error)]
    calibrated_average_351_max = [average_list.pop(4) * (stool_gradient + stool_gradient_error) + (stool_intercept + stool_intercept_error)]


    calibrated_average_speed_list_all = calibrated_average_speed_list[:4] + calibrated_average_351 + calibrated_average_speed_list[4:]

    calibrated_average_speed_list_all_min = calibrated_average_speed_list_min[:4] + calibrated_average_351_min + calibrated_average_speed_list_min[4:]

    calibrated_average_speed_list_all_max = calibrated_average_speed_list_max[:4] + calibrated_average_351_max + calibrated_average_speed_list_max[4:]


    print(calibrated_average_speed_list)
    print(calibrated_average_speed_list_all)

    plt.figure()
    plt.scatter(np.log(x_dummy), calibrated_average_speed_list_all, color = 'red', label = 'Calculated Averages')
    plt.scatter(0,0)
    # plt.xticks(np.log(x_dummy), [int(c.split('[')[0]) for c in selected_columns.columns])
    plt.title('Beach Measurements - All datapoints')
    plt.xlabel('Height (cm)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(5)
    plt.close()


    # gaussian method - from dell

    all_but_351_df = selected_columns.drop(selected_columns.columns[[4]], axis = 1)
    print(all_but_351_df)

    print("---" *30)
    calibrated_df = pd.DataFrame()

    for i in range(len(all_but_351_df.columns)):
        calibrated_df[all_but_351_df.columns[i]] = all_but_351_df.iloc[:, i] * delta_gradient_list[i] + delta_intercept_list[i] + all_but_351_df.iloc[:,i]
    print(calibrated_df)


    # dummy_df = all_but_351_df[all_but_351_df > 0]
    # print(dummy_df)
    # print("dummy_df")
    # print("---"*100)

    # calibrated_df_no_zero = pd.DataFrame()

    # for i in range(len(dummy_df.columns)):
    #     calibrated_df_no_zero[dummy_df.columns[i]] = dummy_df.iloc[:, i] * delta_gradient_list[i] + delta_intercept_list[i] + dummy_df.iloc[:,i]
    # print(calibrated_df_no_zero)
    # print("---"*100)


    amp_list, mu_list, sigma_list, mu_error_list = [], [], [], []

    bins = 100

    plt.figure()
    for i, column in enumerate(calibrated_df.columns, start=1):
        counts, bins_location = np.histogram(calibrated_df[column], bins=bins)
        counts[0] = 0
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
        mu_error_list.append(np.sqrt(cov[1,1]))

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
    
    plt.suptitle('Calibrated Histograms')
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(10)
    plt.close()

    print("Amplitudes:", amp_list)
    print("Means:", mu_list)
    print("Means Errors:", mu_error_list)
    print("Standard deviations:", sigma_list)

    calibrated_average_speed_list_all = mu_list[:4] + calibrated_average_351 + mu_list[4:]

    u_star_list = sigma_list


    x_dummy = [2, 1, 3, 4, 5, 6, 7]
    x_dummy = [113, 63, 155, 193, 235, 297, 389]
    dist_errors = 5 #cm
    x_max = np.array(x_dummy) + dist_errors
    x_min = np.array(x_dummy) - dist_errors
    
    plt.figure()
    plt.scatter(np.log(x_dummy), calibrated_average_speed_list_all, color = 'red', label = 'Gaussian Averages')
    # plt.errorbar(np.log(x_dummy), calibrated_average_speed_list_all, xerr = np.array(x_min, x_max), 
                #  color = 'red', label = 'Gaussian Averages')
    # plt.scatter(0,0)
    # plt.xticks(np.log(x_dummy), [int(c.split('[')[0]) for c in selected_columns.columns])
    plt.title('Beach Measurements')
    plt.xlabel('log Height (cm)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(5)
    plt.close()

    # 4

    log_z = [4.143134726391533, 5.043425116919247, 5.262690188904886, 5.459585514144159, 5.6937321388027, 5.963579343618446]
    
    mean_u = [1.8284090478307335, 1.9677130971633128, 2.384673304691925, 2.474661661885532, 2.3661642884683762, 2.8545403096463255]

    mean_u_error = [0.1406093383634902, 0.12310657118068324, 0.07215002722183605, 0.11334256406865317, 0.17732129549146136, 0.1112217595671865]

    log_z_plus_error = [0.01574835696813892, 0.0064308903302903175, 0.005167970158442614, 0.004246290881451564, 0.0033613477027047978, 0.002567395505246317]

    log_z_minus_error = [0.016000341346440905, 0.006472514505617255, 0.0051948168771041026, 0.004264398786457235, 0.003372684478639698, 0.0025740039951722693]



    x_dummy = np.array(np.arange(min(log_z),max(log_z),0.01))

    lin_model = Model(linear)

    data = RealData(log_z, mean_u, sx=(np.array(log_z_minus_error) + np.array(log_z_plus_error)) / 2, sy=mean_u_error)

    odr = ODR(data, lin_model, beta0=[0., 1.])
    out = odr.run()

    x_fit = np.linspace(log_z[0], log_z[-1], 100)
    y_fit = linear(out.beta, x_fit)

    gradient = out.beta[0]
    intercept = out.beta[1]

    gradient_error = out.sd_beta[0]
    intercept_error = out.sd_beta[1]

    # gradient_error_g = out_g.cov_beta[0,0]
    # intercept_error_g = out_g.cov_beta[1,1]


    plt.errorbar(log_z, mean_u, xerr=(log_z_minus_error, log_z_plus_error), yerr=mean_u_error, \
                capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
                    color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data')
    plt.plot(x_fit, y_fit, label = 'ODR Fit', color = 'black')
    y_min =  (gradient - gradient_error) * np.array(x_dummy) + (intercept - intercept_error)
    y_max =  (gradient + gradient_error) * np.array(x_dummy) + (intercept + intercept_error)
    plt.fill_between(x_dummy, y_min, y_max, alpha = 0.5, label = 'Uncertainty', color='grey')

    plt.grid()
    plt.title('main graph')
    plt.xlabel('log z')
    plt.ylabel('mean u')


    plt.show()

analysis(block = True)
