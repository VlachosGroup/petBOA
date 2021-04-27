# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:42:15 2020

@author: djrob
"""
import os
from pathlib import Path
import pandas as pd
from IPython.display import display
import os
from pprint import pprint
from pathlib import Path
from pmutt.io.excel import read_excel
import numpy as np
from scipy.interpolate import Rbf, interp1d, BSpline, make_interp_spline
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate

excel_path = r'C:\Users\djrob\Desktop\Work\Fall_2020\openMKM v Chemkin\Methane\methane_transient_data.xlsx'
chemkin_cov_data = read_excel(io=excel_path, sheet_name='chemkin_cov')
omkm_cov_data = read_excel(io=excel_path, sheet_name='omkm_cov')
chemkin_mass_data = read_excel(io=excel_path, sheet_name='chemkin_mass')
omkm_mass_data = read_excel(io=excel_path, sheet_name='omkm_mass')

'''Declaring Variables'''
chemkin_time = []
omkm_time = [0.001]
chemkin_conv = []
omkm_conv = [0.000]
chemkin_o2 = []
omkm_o2 = [8.92E-04]
chemkin_co2 = []
omkm_co2 = [3.53E-06]
chemkin_pt_s = []
omkm_pt_s = [9.04E-01]
chemkin_o_s = []
omkm_o_s = [9.00E-02]
chemkin_ch4_s = []
omkm_ch4_s = [2.20E-04]
chemkin_o2_s = []
omkm_o2_s = [2.17E-08]


'''Adding Values from Sheet'''
for i in range(0, 500):
    chemkin_time.append(chemkin_cov_data[i]['Time [s]'])
    chemkin_conv.append(chemkin_mass_data[i]['Conv %'])
    chemkin_o2.append(chemkin_mass_data[i]['O2'])
    chemkin_co2.append(chemkin_mass_data[i]['CO2'])
    chemkin_pt_s.append(chemkin_cov_data[i]['PT(S)'])
    chemkin_o_s.append(chemkin_cov_data[i]['O(S)'])
    chemkin_ch4_s.append(chemkin_cov_data[i]['CH4(S)'])
    chemkin_o2_s.append(chemkin_cov_data[i]['O2(S)'])

    
for i in range(0, 999):
    omkm_time.append(omkm_cov_data[i]['t(s)'])
    omkm_conv.append(omkm_mass_data[i]['Conv %'])
    omkm_o2.append(omkm_mass_data[i]['O2'])
    omkm_co2.append(omkm_mass_data[i]['CO2'])
    omkm_pt_s.append(omkm_cov_data[i]['PT(S)'])
    omkm_o_s.append(omkm_cov_data[i]['O(S)'])
    omkm_ch4_s.append(omkm_cov_data[i]['CH4(S)'])
    omkm_o2_s.append(omkm_cov_data[i]['O2(S)'])
    
'''Interpolations'''
conv_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_conv), 2)
o2_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_o2),2)
co2_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_co2),2)
pt_s_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_pt_s),2)
o_s_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_o_s),2)
ch4_s_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_ch4_s),2)
o2_s_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_o2_s),2)

'''ABSOLUTE MEAN AND MAX DIFF MASS'''
conv_diff = np.array(chemkin_conv) - np.array(conv_inter(chemkin_time))
conv_avg_diff = np.average(conv_diff)
o2_diff = np.array(chemkin_o2) - np.array(o2_inter(chemkin_time))
o2_avg_diff = np.average(o2_diff)
co2_diff = np.array(chemkin_co2) - np.array(co2_inter(chemkin_time))
co2_avg_diff = np.average(co2_diff)
overall_mass_diff = np.concatenate((conv_diff, o2_diff, co2_diff), axis = 0)
max_mass_mean_diff = max(abs(np.array(overall_mass_diff)))
print("Maximum Difference for Mass: " + str(max_mass_mean_diff))
print('Absolute Mean for Mass: ' + str(np.average(abs(overall_mass_diff))))


'''ABSOLUTE MEAN AND MAX DIFF COV'''
pt_s_diff = np.array(chemkin_pt_s) - np.array(pt_s_inter(chemkin_time))
pt_s_avg_diff = np.average(pt_s_diff)
o_s_diff = np.array(chemkin_o_s) - np.array(o_s_inter(chemkin_time))
o_s_avg_diff = np.average(o_s_diff)
ch4_s_diff = np.array(chemkin_ch4_s) - np.array(ch4_s_inter(chemkin_time))
ch4_s_avg_diff = np.average(ch4_s_diff)
o2_s_diff = np.array(chemkin_o2_s) - np.array(o2_s_inter(chemkin_time))
o2_s_avg_diff = np.average(o2_s_diff)
overall_cov_diff = np.concatenate((pt_s_diff, o_s_diff, ch4_s_diff, o2_s_diff), axis = 0)
max_cov_mean_diff = max(abs(np.array(overall_cov_diff)))
print("Maximum Difference for Cov: " + str(max_cov_mean_diff))
print('Absolute Mean for Cov: ' + str(np.average(abs(overall_cov_diff))))
plt.rc('font', size=28)

plt.figure('Conversion')
chemkin_time_adjusted = [chemkin_time[i]*100 for i in range(0, 500)]
plt.plot(chemkin_time_adjusted, conv_inter(chemkin_time), 'k', linewidth=3)
plt.plot(chemkin_time_adjusted, chemkin_conv, '--r', linewidth=3)
#plt.plot(chemkin_time, conv_inter(chemkin_time), '--b', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0, 0.025*100])
plt.ylim([0, .15])
plt.xlabel('Time[s*$10^{-2}$]', fontsize=35)
plt.ylabel('Conversion %', fontsize=40)


plt.figure('O2 Mass')
omkm_o2_mass = o2_inter(chemkin_time)
omkm_o2_adjusted = [omkm_o2_mass[i]*100 for i in range(0, 500)]
chemkin_o2_adjusted = [chemkin_o2[i]*100 for i in range(0, 500)]
plt.plot(chemkin_time_adjusted, omkm_o2_adjusted, 'k', linewidth=3)
plt.plot(chemkin_time_adjusted, chemkin_o2_adjusted, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0.005*100, 0.02*100])
plt.ylim([0, 0.07*100])
plt.xlabel('Time[s*$10^{-2}$]', fontsize=35)
plt.ylabel('O2 Mass fraction*$10^{-2}$', fontsize=40)

plt.figure('CO2 Mass')
plt.plot(chemkin_time, co2_inter(chemkin_time), 'k', linewidth=3)
plt.plot(chemkin_time, chemkin_co2, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0, 0.02])
plt.ylim([0, 0.04])
plt.xlabel('Time[s]', fontsize=35)
plt.ylabel('CO2 Mass fraction', fontsize=40)

plt.figure('Step Vacancies')
plt.plot(chemkin_time_adjusted, pt_s_inter(chemkin_time), 'k', linewidth=3)
plt.plot(chemkin_time_adjusted, chemkin_pt_s, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0.0075*100, 0.02*100])
plt.ylim([0.5, 0.65])
plt.xlabel('Time[s*$10^{-2}$]', fontsize=35)
plt.ylabel('Step Vacancies', fontsize=40)

plt.figure('O(S) Coverage')
plt.plot(chemkin_time_adjusted, o_s_inter(chemkin_time), 'k', linewidth=3)
plt.plot(chemkin_time_adjusted, chemkin_o_s, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0.01*100, 0.015*100])
plt.ylim([0.4, 0.55])
plt.xlabel('Time[s*$10^{-2}$]', fontsize=35)
plt.ylabel('O(S) Coverage', fontsize=40)

plt.figure('CH4(S) Coverage')
plt.plot(chemkin_time, ch4_s_inter(chemkin_time), 'k', linewidth=3)
plt.plot(chemkin_time, chemkin_ch4_s, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0, 0.02])
plt.ylim([0, 0.0003])
plt.xlabel('Time[s]', fontsize=35)
plt.ylabel('CH4(S) Coverage', fontsize=40)

plt.figure('O2(S) Coverage')
omkm_o2_cov = o2_s_inter(chemkin_time)
omkm_o2_s_adjusted = [omkm_o2_cov[i]*1000 for i in range(0, 500)]
chemkin_o2_s_adjusted = [chemkin_o2_s[i]*1000 for i in range(0, 500)]
plt.plot(chemkin_time_adjusted, omkm_o2_s_adjusted, 'k', linewidth=3)
plt.plot(chemkin_time_adjusted, chemkin_o2_s_adjusted, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0.005*100, 0.02*100])
plt.ylim([0, 0.0006*1000])
plt.xlabel('Time[s*$10^{-2}$]', fontsize=35)
plt.ylabel('O2(S) Coverage*$10^{-3}$', fontsize=40)
