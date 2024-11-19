import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.odr import *

plt.rcParams["figure.figsize"] = (10,5)

# Parameters
time = 3  # in minutes
nrows = (time * 60) + 1

radius = 40
radius_error = 1

def angular_vel_converstion(RPM, radius = radius):
    return (np.pi / 30) * RPM * (radius / 100)

def linear(p, x):
     m, c = p
     return m*x + c

def analysis(file_path=None, block=False, bins=10, show_fit = True):
    """
    Analyzes data for devices, plots speed vs. time and histogram with Gaussian fit.
    
    Parameters:
        device_number (list): List of device numbers to process.
        file_path (Path): Base directory for data files.
        block (bool): Whether to block plots on show.
        bins (int): Number of bins for histogram.
    """
    file = 'C:\\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 2\Data\\351 Calibration\RPMs.csv'
    rpm_df = pd.read_csv(file)
    print(rpm_df.shape)
    print(rpm_df)
    rpms_list = rpm_df.columns
    print(rpms_list)

    path = 'C:\\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 2\Data\\351 Calibration\Stool Data\\'
    df = pd.DataFrame()

    for rpm in rpms_list:
        print(path + f'{str(rpm)}.csv')
        data = pd.read_csv(path + f'{str(rpm)}.csv', nrows = nrows)
        df[rpm] = data.iloc[:,2]
    print(df)

    rpms_average_list = []
    rpms_std_dev_list = []

    for i in range(len(rpm_df.columns)):
        average = rpm_df.iloc[:, i].mean()
        std_dev = rpm_df.iloc[:, i].std()

        rpms_average_list.append(average)
        rpms_std_dev_list.append(std_dev)

    print(rpms_average_list)
    print(rpms_std_dev_list)


    speed_list = []

    for rpm in rpms_average_list:
        speed = angular_vel_converstion(rpm)
        speed_list.append(speed)

    print("speed_list")    
    print(speed_list)

    speed_rel_error = np.sqrt(((radius_error/radius) ** 2) + ((rpms_std_dev_list/rpm) ** 2))
    speed_error_list = speed_rel_error * np.array(speed_list)

    average_wind_speed_list = []
    std_dev_wind_speed_list = []

    for i in range(len(df.columns)):
        column_data = df.iloc[:, i]
        non_zero_data = column_data[column_data != 0]
        
        if len(non_zero_data) > 0:
            average = np.average(non_zero_data)
            std_dev = np.std(non_zero_data)
        else:
            average = np.nan
            print("Error Computing Average")

        average_wind_speed_list.append(average)
        std_dev_wind_speed_list.append(std_dev)

    print("average_wind_speed_list")
    print(average_wind_speed_list)
    print("std_dev_wind_speed_list")
    print(std_dev_wind_speed_list)

    # plt.scatter(speed_list, average_wind_speed_list, marker='x')
    plt.errorbar(speed_list, average_wind_speed_list, xerr = speed_error_list, yerr = std_dev_wind_speed_list, fmt = 'x', label = 'data')
    plt.xlabel('Theoretical Speed (m/s)')
    plt.ylabel('Average Measured Wind Speed (m/s)')
    plt.legend()
    plt.show()


    gradient_list = []
    intercept_list = []

    gradient_error_list = []
    intercept_error_list = []

    x_dummy_1 = np.array(np.arange(min(speed_list),max(speed_list),0.001))
    lin_model_1 = Model(linear)
    # data_1 = RealData(speed_list, average_wind_speed_list)
    # filename = '351 Calibration without errors'
    data_1 = RealData(speed_list, average_wind_speed_list, sx = speed_error_list, sy = std_dev_wind_speed_list)
    filename = '351 Calibration with errors'


    odr_1 = ODR(data_1, lin_model_1, beta0=[0., 0.])
    out_1 = odr_1.run()


    x_fit_1 = np.array(np.arange(min(speed_list),max(speed_list),0.001))
    y_fit_1 = linear(out_1.beta, x_fit_1)

    x_fit_2 = np.array(np.arange(0, min(speed_list),0.001))
    y_fit_2 = linear(out_1.beta, x_fit_2)

    x_fit_3 = np.array(np.arange(max(speed_list), 6, 0.001))
    y_fit_3 = linear(out_1.beta, x_fit_3)

    x_fit_4 = np.array(np.arange(0, 6, 0.001))
    y_fit_4 = linear(out_1.beta, x_fit_4)


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

    plt.errorbar(speed_list, average_wind_speed_list, xerr = speed_error_list, yerr = std_dev_wind_speed_list,\
                capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
                    color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data')
    plt.plot( x_fit_1, y_fit_1, label = 'ODR Fit', color = 'black')
    # plt.plot( x_fit_2, y_fit_2, label = 'ODR Fit - Extrapolate', color = 'blue', linestyle = 'dashed')
    # plt.plot( x_fit_3, y_fit_3, label = 'ODR Fit - Extrapolate', color = 'blue', linestyle = 'dashed')
    
    plt.xlabel(r'Theoretical Speed ($\dfrac{m}{s}$)')
    plt.ylabel(r'Average Measured Wind Speed ($\dfrac{m}{s}$)')
    plt.title('Turntable Stool Calibration for Reference Detector (#351)')
    plt.text(4, 2, "Gradient = {0:.2e} \u00b1 {1:.2e} \nIntercept = {2:.2e} \u00b1 {3:.2e}" \
            .format(gradient_1, gradient_error_1, intercept_1, intercept_error_1), bbox = dict(facecolor = 'white'))

    y_min_1 =  (gradient_1 - gradient_error_1) * np.array(x_dummy_1) + (intercept_1 - intercept_error_1)
    y_max_1 =  (gradient_1 + gradient_error_1) * np.array(x_dummy_1) + (intercept_1 + intercept_error_1)
    plt.fill_between(x_dummy_1, y_min_1, y_max_1, alpha = 0.5, label = 'Uncertainty', color='grey')
    # plt.fill_between(x_fit_4, y_min_1, y_max_1, alpha = 0.5, label = 'Uncertainty', color='grey')

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig('final\\'+filename+'.png', dpi=1200)
    plt.show()


    print('Params')
    print("---" * 30)
    out_1.pprint()
    print("Gradient", gradient_1)
    print("Intercept", intercept_1)
    print("Gradient Error", gradient_error_1)
    print("Intercept Error", intercept_error_1)

    save_list = [gradient_1, intercept_1, gradient_error_1, intercept_error_1]
    print(save_list)
    np.savetxt('final\\'+filename+'.txt', save_list)

analysis()
