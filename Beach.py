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

def analysis(block = False, filename = selected_csv):

# change this to reflect file name
    file = 'C:\\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 2\Data\Session6 - 7th Nov\\' + str(filename) +'.csv'
    title_name = file.split('Nov\\')[1].split('.csv')[0]
    
    data = pd.read_csv(file, nrows=nrows+1)
    data = data.iloc[:, 1:-1]

    auto_averages = data.tail(1).iloc[:, 1::2]
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

    plt.figure()
    for i, column in enumerate(selected_columns.columns, start=1):
        plt.subplot(3, 3, i)
        plt.plot(df.index, df[column])
        plt.title(f"Detector: {column.split('[')[0]}")
        plt.ylabel('Speed (m/s)')
        plt.xlabel('Time (sec)')
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(5)
    plt.close()

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

    plt.scatter(x_dummy, average_list, color = 'red', label = 'Calculated Averages')
    plt.scatter(x_dummy, auto_averages, color = 'blue', label = 'Auto Averages')
    plt.scatter(0,0)
    plt.xticks(x_dummy, [int(c.split('[')[0]) for c in selected_columns.columns])
    plt.title('Detector Calibration')
    plt.xlabel('Device Number')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(5)
    plt.close()

analysis(block = False)
