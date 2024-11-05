# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:25:29 2024

@author: Maanas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["figure.figsize"] = (10,8)

#%%
time = 15
nrows = (time * 60) + 1

device_number = 9


# file = 'C:\\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 2\Data\Session1 - 29th Oct\\' + str(device_number) + ' high.csv'
file = 'C:\\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 2\Data\Session3 - 1st Nov\\'  + 'Level 8.csv'
title_name = file.split('Nov\\')[1].split('.csv')[0]

data = pd.read_csv(file, nrows=nrows)

# data.drop(data.columns[13], axis=1, inplace = True)

data['seconds'] = data.index.values

data.replace(to_replace='-', value = 0.0, inplace = True)

plt.plot(data['seconds'] , data[data.columns[2]], label = '240 = pos 5')
plt.plot(data['seconds'] , data[data.columns[4]], label = '284 = pos 2')
plt.plot(data['seconds'] , data[data.columns[6]], label = '348 = pos 3')
plt.plot(data['seconds'] , data[data.columns[8]], label = '351 = pos 4')
plt.plot(data['seconds'] , data[data.columns[10]], label = '402 = pos 6')
plt.plot(data['seconds'] , data[data.columns[12]], label = '994 = pos 1' )
plt.legend()
plt.show()

#%%
# 240 284, 348, 351, 402, 994


dist = [0.15, 0.45, 0.75, 1.05, 1.35, 1.65]
# avg = [1, 0.75, 0.8, 0.94, 0.98, 0.65]
avg = [0.65, 0.75, 0.8, 0.94, 1, 0.98]
plt.scatter(dist, avg)
plt.show()

#%%
data['seconds'] = data.index.values
# data['temp'] = data[data.columns[1]]
# data['speed'] = data[data.columns[2]]

# data.drop(data.columns[3], axis=1, inplace = True)

plt.plot(data.seconds, data.speed)
plt.title(title_name + ' speed')
plt.ylabel('speed (m/s)')
plt.xlabel('time (sec)')
plt.tight_layout()
plt.show()


hist, bin_edges = np.histogram(data['speed'])
print(hist)

plt.hist(data['speed'], bins=50)
plt.title(title_name + ' speed')
plt.show()

