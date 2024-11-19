#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:26:27 2024

@author: dellkang
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit



#%%
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def line_unc(x,m,c,munc,cunc):
    maxy=(m+munc)*x+(c+cunc)
    miny=(m-munc)*x+(c-cunc)
    return 0.5*(maxy-miny) #Uncertainty of detector against 351
def chi2(E,o):
    s=0
    for i in range(len(E)):
        s+=((E[i]-o[i])**2)/E[i]
    return s

    #%%

data=np.loadtxt('beach/3.csv',skiprows=1,unpack='true',delimiter=','
                ,usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14)
                , max_rows=901, dtype = str)
data = [[0.0 if x == '-' or x is None else float(x) for x in row] for row in data]

z=[113/100,63/100,155/100,193/100,235/100,297/100,389/100]
new_z=[63/100,155/100,193/100,235/100,297/100,389/100]

# new_z=[63,155,193,235,297,389] #cm
# new_mean=[mean272,mean284,mean348,mean351,mean518,mean994]
new_z_err=[1/100,1/100,1/100,1/100,1/100,1/100]
# new_z_err=[1,1,1,1,1,1] #cm

# %%
T_240=data[0]
v_240=data[1]
T_272=data[2]
v_272=data[3]
T_284=data[4]
v_284=data[5]
T_348=data[6]
v_348=data[7]
T_351=data[8]
v_351=data[9]
T_518=data[10]
v_518=data[11]
T_994=data[12]
v_994=data[13]



#%%
#Calibration
m240=1.4879
m272=9.0321e-1
m284=1.0825
m348=1.1434
m518=1.5063
m994=1.1801

c240=-1.8035e-1
c272=-1.9527e-1
c284=-5.8069e-1
c348=-2.1143e-1
c518=-4.6976e-1
c994=-7.6689e-2

munc240=3.241320319734977695e-02
munc272=3.324891361030293813e-02
munc284=2.517884353813937617e-02
munc348=1.432271694145211341e-02
munc518=4.249759182352629006e-02
munc994=1.883829136208326621e-02

cunc240=6.841217878119516815e-02
cunc272=7.981672388787001970e-02
cunc284=7.356183097926022318e-02
cunc348=3.799502648089626844e-02
cunc518=7.676501137272796760e-02
cunc994=5.744709750925769409e-02

m351=1.0235
c351=-1.573e-1
munc351=2.05e-2
cunc351=6.2612e-2

def calibrate(data,m,c):
    data351=[]
    for i in range(len(data)):
        data351.append(m*data[i]+c)
        
    ac_data=[]
    for i in range(len(data351)):
        ac_data.append(m351*data351[i]+c351)
    return ac_data

v_240=calibrate(v_240,m240,c240)
v_272=calibrate(v_272,m272,c272)
v_284=calibrate(v_284,m284,c284)
v_348=calibrate(v_348,m348,c348)
v_518=calibrate(v_518,m518,c518)
v_994=calibrate(v_994,m994,c994)
v_351=calibrate(v_351,1,0)

#%%

def gaussian_fit(data):
    counts, bin_edges,_ = plt.hist(data, bins=100)

    counts[0]=0


    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    guesses=[max(counts),(max(bin_centers)/2),(max(bin_centers)/5)]
    fit_gaussian=curve_fit(gaussian,bin_centers,counts,p0=guesses)
    gaussian_plot=gaussian(bin_centers,*fit_gaussian[0])
    plt.plot(bin_centers,gaussian_plot,color='red')
    plt.grid()
    plt.xlabel('Wind speed (m/s)')
    plt.ylabel('Frequency bins=100')
    plt.legend(['Gaussian Fit','Measured Data'])

    std=fit_gaussian[0][2]
    mean=fit_gaussian[0][1]
    #print(bin_edges[2]-bin_edges[1])
    return [mean,std]

#%%

mean240,std240=gaussian_fit(v_240)
mean272,std272=gaussian_fit(v_272)
mean284,std284=gaussian_fit(v_284)
mean348,std348=gaussian_fit(v_348)
mean351,std351=gaussian_fit(v_351)
mean518,std518=gaussian_fit(v_518)
mean994,std994=gaussian_fit(v_994)
unc240=line_unc(mean240,m240,c240,munc240,cunc240)
unc272=line_unc(mean272,m272,c272,munc272,cunc272)
unc284=line_unc(mean284,m284,c284,munc284,cunc284)
unc351=line_unc(mean351,m351,c351,munc351,cunc351)
unc348=line_unc(mean348,m348,c348,munc348,cunc348)
unc518=line_unc(mean518,m518,c518,munc518,cunc518)
unc994=line_unc(mean994,m994,c994,munc994,cunc994)
unc_u=[unc240,unc272,unc284,unc348,unc351,unc518,unc994]

new_unc=[]
for i in range(1,len(unc_u)):
    new_unc.append(unc_u[i])
    #%%
gaussian_fit(v_351)

#%%
params={
        'font.size':12,
        'axes.labelsize':14,
        'figure.figsize':[10,5]}
plt.rcParams.update(params)

std=[std240,std272,std284,std348,std351,std518,std994]
mean=[mean240,mean272,mean284,mean348,mean351,mean518,mean994]
new_std=[std272,std284,std348,std351,std518,std994]
# Not 2xstd since u'=u-mean u
std_err=[]
for i in range(len(new_std)):
    std_err.append(0.0452/2)
fit_a,cov_a=np.polyfit(new_z,new_std,0,cov='unscaled')
p_a=np.poly1d(fit_a)
plt.plot(new_z,p_a(new_z))

plt.subplot(1,2,1)
plt.errorbar(new_z,new_std,yerr=std_err,xerr=new_z_err,fmt='x',capsize=4)
plt.plot(new_z,p_a(new_z))
plt.legend(['Fitted Line','Measured Data'])
plt.xlabel('Height (m)')
plt.ylabel('Standard Deviation of u (m/s)')
plt.grid()

plt.title('u* Value')

plt.ylim([0.5,1])
plt.subplot(1,2,2)
gaussian_fit(v_351)

plt.xlabel('Wind speed detected (m/s)')
plt.ylabel("Frequency bins=100")
plt.title('Detector Histogram Example')

# plt.savefig('u star and histogram.jpg')
print(p_a/0.4)

print('chi2 =',chi2(p_a(new_z),new_std),'p=1')
#%%
z_err=[]
new_z_err=[1/100,1/100,1/100,1/100,1/100,1/100]
for i in range(len(mean)):
    z_err.append(1)

new_z=[63/100,155/100,193/100,235/100,297/100,389/100]
new_mean=[mean272,mean284,mean348,mean351,mean518,mean994]

#plt.errorbar(mean,z,xerr=unc_u,yerr=z_err,fmt='x',capsize=4)
plt.errorbar(new_mean,new_z,xerr=new_unc,yerr=new_z_err,fmt='x',capsize=4)
plt.xlim([0,3.7])
plt.ylim([0,4])
#plt.plot(z,mean_T,'x')

plt.grid()
plt.ylabel('z (cm)')
plt.xlabel('mean u (m/s)')

#%%
z2=[]
for i in range(1,400):
    z2.append(i/100)
def log_fit(z,star,z0):
    mean_u=[]
    for i in range(len(z)):
        mean_u.append((star/0.4)*np.log(z[i]/z0))
    return mean_u
guesses=[0.02,0.1]
fit_log=curve_fit(log_fit,z,mean,p0=guesses)

log_plot=log_fit(z2,*fit_log[0])
plt.plot(log_plot,z2)
plt.plot(mean,z,'x')
plt.grid()
plt.ylabel('z (cm)')
plt.xlabel('mean u (m/s)')

z0_val=fit_log[0][1]
eddy_vel=fit_log[0][0]

print(z0_val,eddy_vel)
#%% Now with no 240

def log_fit(z,amp,c_int):
    mean_u=[]
    for i in range(len(z)):
        mean_u.append(amp*np.log(z[i])+c_int)
    return mean_u


#%%

params={
        'font.size':15,
        'axes.labelsize':15,
        'figure.figsize':[10,5]}
plt.rcParams.update(params)
upper_z=[]
lower_z=[]
for i in range(0,len(new_z)):
    log_unc_up=np.log(new_z[i]+new_z_err[i])-np.log(new_z[i])
    log_unc_low=np.log(new_z[i])-np.log(new_z[i]-new_z_err[i])
    upper_z.append(log_unc_up)
    lower_z.append(log_unc_low)


log_z=[]
for i in range(len(new_z)):
    log_z.append(np.log(new_z[i]))



def linear(x,m,c):
    y_list=[]
    for i in range(len(x)):
        y=m*x[i]+c
        y_list.append(y)
    return y_list

guess_fit=[0.58,2.5]
fit_linear=curve_fit(linear,log_z,new_mean,sigma=new_unc,p0=guess_fit)
linear_plot=linear(log_z,*fit_linear[0])
#plt.plot(log_z,linear_plot)

max_mean=[new_mean[0]-new_unc[0],new_mean[1]-new_unc[1]
         ,new_mean[2]+new_unc[2],new_mean[3]+new_unc[3]
         ,new_mean[4]+new_unc[4],new_mean[5]+new_unc[5]]

min_mean=[new_mean[0]+new_unc[0],new_mean[1]+new_unc[1]
         ,new_mean[2]-new_unc[2],new_mean[3]-new_unc[3]
         ,new_mean[4]-new_unc[4],new_mean[5]-new_unc[5]]

plt.errorbar(log_z,new_mean,xerr=[upper_z,lower_z],yerr=new_unc,fmt='x',capsize=4)

fit_a,cov_a=np.polyfit(log_z,new_mean,1,cov='unscaled')
p_a=np.poly1d(fit_a)
fit_a1,cov_a1=np.polyfit(log_z,max_mean,1,cov='unscaled')
p_a1=np.poly1d(fit_a1)
fit_a2,cov_a2=np.polyfit(log_z,min_mean,1,cov='unscaled')
p_a2=np.poly1d(fit_a2)
plt.plot(log_z,p_a(log_z))
plt.title('Wind Profile Results')
plt.legend(['Fitted Line','Measured Data'])
#plt.plot(log_z,p_a1(log_z))
#plt.plot(log_z,p_a2(log_z))
#plt.plot(log_z,new_mean,'x',color='black')

#plt.errorbar(log_z,new_mean,fmt='x',capsize=4)

plt.grid()
print(p_a)
plt.xlabel('log(z)')
plt.ylabel('mean u (m/s)')

amp=p_a[1]
c_int=p_a[0]
# print(c_int)
print('z0 =',np.exp(-c_int/amp)*1000,'mm')
print('Eddy vel =', amp*0.4,'m/s')
# plt.savefig('Final results graph')

#%%
print(new_unc)

#%%

log_plot=log_fit(z2,amp,c_int)
plt.plot(z2,log_plot)
plt.errorbar(new_z,new_mean,yerr=new_unc,fmt='x',capsize=4)
plt.grid()
plt.xlabel('z Value (m)')
plt.ylabel('mean u m/s')

#%%
print('log_z values (cm)',log_z)
print('mean u values m/s', new_mean)
print('u uncertainty', new_unc)
print('log z unc (Upper bound)',upper_z)
print('log z uncertainty lower bound',lower_z)
#%%
def mean(T):
    return sum(T)/len(T)

Tval=[mean(T_272)+273.15,mean(T_284)+273.15
      ,mean(T_348)+273.15,mean(T_351)+273.15,
      mean(T_518)+273.15,mean(T_994)+273.15]

av_T=mean(Tval) #+273.15
plt.plot(new_z,Tval,'x')

fit_T,cov_T=np.polyfit(new_z,Tval,1,cov='unscaled')
p_T=np.poly1d(fit_T)
plt.plot(z2,p_T(z2))

def differentiate (x,y):
    dif_list=[]
    for i in range(len(y)-1):
        dy=y[i+1]-y[i]
        dx=x[i+1]-x[i]
        dif_list.append(dy/dx)
    return dif_list

dif_T=differentiate(z2,p_T(z2))
dif_u=differentiate(z2,log_plot)
#plt.plot(z2,log_plot)
def Ri(dif_T,dif_u):
    Ri_list=[]
    for i in range(len(dif_T)):
        val=(9.81/av_T)*((dif_T[i])/(dif_u[i])**2)
        #print((dif_T[i])/(dif_u[i]**2))
        Ri_list.append(val)
    return Ri_list
z3=[]
for i in range(len(z2)-1):
    z3.append(z2[i])

Ri_val=Ri(dif_T,dif_u)
print(Ri_val)
#plt.plot(z3,Ri_val)
#plt.plot(z3,dif_T)
plt.grid()

plt.xlabel('Height (m)')
plt.ylabel('Ri Value')

#%%



print('Chi squared for linear fit is:',chi2(p_T(new_z),Tval),'p=1, thus good fit')



