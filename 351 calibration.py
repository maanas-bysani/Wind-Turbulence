import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.odr import *

plt.rcParams["figure.figsize"] = (10,8)

# Parameters
time = 3  # in minutes
nrows = (time * 60) + 1
# device_number = [2, 3, 4, 5, 6, 7, 8, 9]
# rpms = [240, 272, 284, 348, 518, 994]
# device_number = [240, 272, 284, 348]

def angular_vel_converstion(RPM, radius = 40):
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

    average_wind_speed_list = []

    for i in range(len(df.columns)):
        column_data = df.iloc[:, i]
        non_zero_data = column_data[column_data != 0]
        
        if len(non_zero_data) > 0:
            average = np.average(non_zero_data)
        else:
            average = np.nan
            print("Error Computing Average")

        average_wind_speed_list.append(average)

    print("average_wind_speed_list")
    print(average_wind_speed_list)

    plt.scatter(speed_list, average_wind_speed_list, marker='x')
    plt.xlabel('Theoretical Speed (m/s)')
    plt.ylabel('Wind Speed (m/s)')
    plt.show()


    gradient_list = []
    intercept_list = []

    x_dummy_1 = np.array(np.arange(min(speed_list),max(speed_list),0.001))
    lin_model_1 = Model(linear)
    data_1 = RealData(speed_list, average_wind_speed_list)

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
    # gradient_list.append(gradient_1)
    # intercept_list.append(intercept_1)

    gradient_error_1 = out_1.sd_beta[0]
    intercept_error_1 = out_1.sd_beta[1]

    # gradient_error_g = out_g.cov_beta[0,0]
    # intercept_error_g = out_g.cov_beta[1,1]

    plt.errorbar(speed_list, average_wind_speed_list, \
                capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
                    color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data')
    plt.plot( x_fit_1, y_fit_1, label = 'ODR Fit', color = 'black')
    # plt.plot( x_fit_2, y_fit_2, label = 'ODR Fit - Extrapolate', color = 'blue', linestyle = 'dashed')
    # plt.plot( x_fit_3, y_fit_3, label = 'ODR Fit - Extrapolate', color = 'blue', linestyle = 'dashed')
    
    plt.ylabel('Delta (m/s)')
    plt.xlabel('Ref Speed (m/s)')
    plt.title('ODR Fit')
    # plt.text(((min(x_fit_1) + max(x_fit_1)) / 2), 1, "Gradient = {0:.2e} \u00b1 {1:.2e} \nIntercept = {2:.2e} \u00b1 {3:.2e}" \
    #         .format(gradient_1, gradient_error_1, intercept_1, intercept_error_1), bbox = dict(facecolor = 'white'))
    plt.text(4, 2, "Gradient = {0:.2e} \u00b1 {1:.2e} \nIntercept = {2:.2e} \u00b1 {3:.2e}" \
            .format(gradient_1, gradient_error_1, intercept_1, intercept_error_1), bbox = dict(facecolor = 'white'))

    y_min_1 =  (gradient_1 - gradient_error_1) * np.array(x_dummy_1) + (intercept_1 - intercept_error_1)
    y_max_1 =  (gradient_1 + gradient_error_1) * np.array(x_dummy_1) + (intercept_1 + intercept_error_1)
    plt.fill_between(x_dummy_1, y_min_1, y_max_1, alpha = 0.5, label = 'Uncertainty', color='grey')
    # plt.fill_between(x_fit_4, y_min_1, y_max_1, alpha = 0.5, label = 'Uncertainty', color='grey')

    plt.grid()
    plt.legend()

    # plt.yticks(rotation=45)
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))

    plt.tight_layout()
    # plt.savefig('Compton - Gaussian.png', dpi=1200)
    plt.show()


    print('Params')
    print("---" * 30)
    out_1.pprint()
    print("Gradient", gradient_1)
    print("Intercept", intercept_1)
    print("Gradient Error", gradient_error_1)
    print("Intercept Error", intercept_error_1)

    save_list = [gradient_1, intercept_1, gradient_error_1, intercept_error_1]

    np.savetxt('data\\351 calibration statistics.txt', save_list)

analysis()



"""
    li = 

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)


    if file_path is None:
        file_path = Path('C:\\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 2\data p\Delta Calibration')

    df = pd.DataFrame()
    delta_df = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()

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

        difference_low = data_low.iloc[:, 4] - data_low.iloc[:, 2]
        
        df[f"Detector: {column_names_high[1]} high"] = data_high.iloc[:, 1]  # temperature
        df[f"Detector: {column_names_high[2]} high"] = data_high.iloc[:, 2]  # speed
        df[f"Detector: {column_names_high[3]} ref high"] = data_high.iloc[:, 3]  # temperature
        df[f"Detector: {column_names_high[4]} ref high"] = data_high.iloc[:, 4]  # speed

        difference_high = data_high.iloc[:, 4] - data_high.iloc[:, 2]

        delta_df[f"{device} low"] = difference_low

        df2[f"{device} low"] = difference_low
        df2[f"ref for {device} low"] = data_low.iloc[:, 4]

        delta_df[f"{device} high"] = difference_high

        df2[f"{device} high"] = difference_high
        df2[f"ref for {device} high"] = data_high.iloc[:, 4]

        df3[f"{device}"] = pd.concat([difference_low, difference_high], ignore_index=True)
        df3[f"ref for {device} high"] = pd.concat([data_low.iloc[:, 4], data_high.iloc[:, 4]], ignore_index=True)


    print(df)
    print(delta_df)
    print(df2)
    print(df3)

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

        plt.title(column)

        # plt.title(column.split('[')[0])
        plt.ylabel('Frequency')
        plt.xlabel('Speed (m/s)')
        plt.legend()
    
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(10)
    plt.close()

    plt.figure()

    for i in range(1, 1 + len(df2.columns) // 2):
        col_x = 2 * (i-1) + 1
        col_y = 2 * (i-1)
        plt.subplot(3, 4, i)
        plt.scatter(df2.iloc[:, col_x], df2.iloc[:, col_y], marker = 'x')
        plt.title(df2.columns[col_y])
        plt.ylabel('Delta (m/s)')
        plt.xlabel('Ref Speed (m/s)')
        plt.tight_layout()

        # plt.subplot(4, 4, i+2)
        # plt.scatter(df2.iloc[:, i+3], df2.iloc[:, i+2], marker = 'x')
        # plt.title(df2.columns[i+2])
        # plt.ylabel('Delta (m/s)')
        # plt.xlabel('Ref Speed (m/s)')
        # plt.tight_layout()

    plt.show(block=block)
    plt.pause(5)
    plt.close()


    plt.figure()
    gradient_list = []
    intercept_list = []

    for i in range(1, 1 + len(df2.columns) // 2):
        col_x = 2 * (i-1) + 1
        col_y = 2 * (i-1)

        plt.subplot(3, 4, i)
        x_dummy_1 = np.array(np.arange(min(df2.iloc[:, col_x]),max(df2.iloc[:, col_x]),0.001))
        lin_model_1 = Model(linear)
        data_1 = RealData(df2.iloc[:, col_x], df2.iloc[:, col_y])

        odr_1 = ODR(data_1, lin_model_1, beta0=[0., 0.])
        out_1 = odr_1.run()

        # x_fit_1 = np.linspace(df2.iloc[:, 1][0], df2.iloc[:, 1][-1], 100)
        x_fit_1 = np.array(np.arange(min(df2.iloc[:, col_x]),max(df2.iloc[:, col_x]),0.001))
        y_fit_1 = linear(out_1.beta, x_fit_1)

        gradient_1 = out_1.beta[0]
        intercept_1 = out_1.beta[1]
        gradient_list.append(gradient_1)
        intercept_list.append(intercept_1)

        gradient_error_1 = out_1.sd_beta[0]
        intercept_error_1 = out_1.sd_beta[1]

        # gradient_error_g = out_g.cov_beta[0,0]
        # intercept_error_g = out_g.cov_beta[1,1]

        plt.errorbar(df2.iloc[:, col_x], df2.iloc[:, col_y], \
                    capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
                        color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data', ms = 1)
        plt.plot( x_fit_1, y_fit_1, label = 'ODR Fit', color = 'black')
        plt.ylabel('Delta (m/s)')
        plt.xlabel('Ref Speed (m/s)')
        plt.title(df2.columns[col_y])
        # plt.text(((min(x_fit_1) + max(x_fit_1)) / 2), 1, "Gradient = {0:.2e} \u00b1 {1:.2e} \nIntercept = {2:.2e} \u00b1 {3:.2e}" \
        #         .format(gradient_1, gradient_error_1, intercept_1, intercept_error_1), bbox = dict(facecolor = 'white'))
        plt.text(((min(x_fit_1) + max(x_fit_1)) / 2), 0.4, "m = {0:.1e} \u00b1 {1:.1e} \nc = {2:.1e} \u00b1 {3:.1e}" \
                .format(gradient_1, gradient_error_1, intercept_1, intercept_error_1), bbox = dict(facecolor = 'white'), size = 'xx-small')

        y_min_1 =  (gradient_1 - gradient_error_1) * np.array(x_dummy_1) + (intercept_1 - intercept_error_1)
        y_max_1 =  (gradient_1 + gradient_error_1) * np.array(x_dummy_1) + (intercept_1 + intercept_error_1)
        plt.fill_between(x_dummy_1, y_min_1, y_max_1, alpha = 0.5, label = 'Uncertainty', color='grey')

        plt.grid()
        plt.legend(fontsize = 'xx-small')

        # plt.yticks(rotation=45)
        # plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))

        plt.tight_layout()

        # plt.savefig('Compton - Gaussian.png', dpi=1200)
        # plt.show()

        print('Params')
        print("---" * 30)
        out_1.pprint()

    plt.show()

    print("gradient_list", gradient_list)
    print("intercept_list", intercept_list)



    plt.figure()
    gradient_list = []
    intercept_list = []
    for i in range(1, 1 + len(df3.columns) // 2):
        col_x = 2 * (i-1) + 1
        col_y = 2 * (i-1)

        plt.subplot(3, 3, i)
        x_dummy_1 = np.array(np.arange(min(df3.iloc[:, col_x]),max(df3.iloc[:, col_x]),0.001))
        lin_model_1 = Model(linear)
        data_1 = RealData(df3.iloc[:, col_x], df3.iloc[:, col_y])

        odr_1 = ODR(data_1, lin_model_1, beta0=[0., 0.])
        out_1 = odr_1.run()

        # x_fit_1 = np.linspace(df2.iloc[:, 1][0], df2.iloc[:, 1][-1], 100)
        x_fit_1 = np.array(np.arange(min(df3.iloc[:, col_x]),max(df3.iloc[:, col_x]),0.001))
        y_fit_1 = linear(out_1.beta, x_fit_1)

        gradient_1 = out_1.beta[0]
        intercept_1 = out_1.beta[1]
        gradient_list.append(gradient_1)
        intercept_list.append(intercept_1)

        gradient_error_1 = out_1.sd_beta[0]
        intercept_error_1 = out_1.sd_beta[1]

        # gradient_error_g = out_g.cov_beta[0,0]
        # intercept_error_g = out_g.cov_beta[1,1]

        plt.errorbar(df3.iloc[:, col_x], df3.iloc[:, col_y], \
                    capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
                        color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data', ms = 1)
        plt.plot( x_fit_1, y_fit_1, label = 'ODR Fit', color = 'black')
        plt.ylabel('Delta (m/s)')
        plt.xlabel('Ref Speed (m/s)')
        plt.title(f"{df3.columns[col_y]}")
        # plt.text(((min(x_fit_1) + max(x_fit_1)) / 2), 1, "Gradient = {0:.2e} \u00b1 {1:.2e} \nIntercept = {2:.2e} \u00b1 {3:.2e}" \
        #         .format(gradient_1, gradient_error_1, intercept_1, intercept_error_1), bbox = dict(facecolor = 'white'))
        plt.text(((min(x_fit_1) + max(x_fit_1)) / 2), 0.4, "m = {0:.1e} \u00b1 {1:.1e} \nc = {2:.1e} \u00b1 {3:.1e}" \
                .format(gradient_1, gradient_error_1, intercept_1, intercept_error_1), bbox = dict(facecolor = 'white'), size = 'xx-small')

        y_min_1 =  (gradient_1 - gradient_error_1) * np.array(x_dummy_1) + (intercept_1 - intercept_error_1)
        y_max_1 =  (gradient_1 + gradient_error_1) * np.array(x_dummy_1) + (intercept_1 + intercept_error_1)
        plt.fill_between(x_dummy_1, y_min_1, y_max_1, alpha = 0.5, label = 'Uncertainty', color='grey')

        plt.grid()
        plt.legend(fontsize = 'xx-small')

        # plt.yticks(rotation=45)
        # plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))

        plt.tight_layout()
        plt.suptitle("high and low combined")

        # plt.savefig('Compton - Gaussian.png', dpi=1200)
        # plt.show()

        print('Params')
        print("---" * 30)
        out_1.pprint()

    plt.show()

    print("gradient_list", gradient_list)
    print("intercept_list", intercept_list)


"""