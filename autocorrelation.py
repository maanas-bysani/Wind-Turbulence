# -*- coding: utf-8 -*-
"""
Created on Sun Nov  10

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


def autocorrelation(series, time_delay):
    """
    Calculate the autocorrelation of a given series and a time delay.
    """
    array = np.array(series)
    N = len(array)

    num = np.mean(array[:N - time_delay] * array[time_delay:])

    denom = np.var(array)

    return num/denom if denom != 0 else 0

def analysis(block = False, filename = selected_csv):

# change this to reflect file name
    file = 'C:\\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 2\Data\Session3 - 1st Nov\\'  + 'Level 8.csv'
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

    time_delay_list = np.linspace(0, 500, 10)
    print(time_delay_list)
    autocorrelation_df = pd.DataFrame(index=time_delay_list)

    for column in selected_columns.columns:
        autocorrelation_list = []
        for tau in time_delay_list:
            r = autocorrelation(selected_columns[column], int(tau)) 
            autocorrelation_list.append(r)
        autocorrelation_df[column] = autocorrelation_list

    for i, column in enumerate(autocorrelation_df.columns, start=1):
        plt.subplot(3, 3, i)
        plt.plot(autocorrelation_df.index, autocorrelation_df[column])
        plt.title(f"Detector: {column.split('[')[0]}")
        plt.ylabel("R(\u03C4)")  # This will display 'R uτ'
        plt.xlabel('Lag Time, \u03C4 (sec)')
    plt.suptitle('AutoCorrelation')
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(5)
    plt.close()


analysis(block = True)
