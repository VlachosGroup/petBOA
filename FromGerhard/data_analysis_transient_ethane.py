# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:15:19 2020

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

excel_path = r'C:\Users\djrob\Desktop\Work\Fall_2020\openMKM v Chemkin\Ethane\ethane_transient_data.xlsx'
chemkin_cov_data = read_excel(io=excel_path, sheet_name='chemkin_cov')
omkm_cov_data = read_excel(io=excel_path, sheet_name='omkm_cov')
chemkin_mass_data = read_excel(io=excel_path, sheet_name='chemkin_mass')
omkm_mass_data = read_excel(io=excel_path, sheet_name='omkm_mass')

'''Declaring Variables'''
chemkin_time = []
omkm_time = []
chemkin_conv = []
omkm_conv = []
chemkin_o2 = []
omkm_o2 = []
chemkin_ch2ch2 = []
omkm_ch2ch2 = []
chemkin_pt_s = []
omkm_pt_s = []
chemkin_c_s = []
omkm_c_s = []
chemkin_ch_s = []
omkm_ch_s = []
chemkin_cch3_s = []
omkm_cch3_s = []
chemkin_cch2_s =[]
omkm_cch2_s = []

'''Adding Values from Sheet'''
for i in range(0, 254):
    chemkin_time.append(chemkin_cov_data[i]['Time [s]'])
    chemkin_conv.append(chemkin_mass_data[i]['Conv %'])
    chemkin_o2.append(chemkin_mass_data[i]['O2'])
    chemkin_ch2ch2.append(chemkin_mass_data[i]['CH2CH2'])
    chemkin_pt_s.append(chemkin_cov_data[i]['PT(S)'])
    chemkin_c_s.append(chemkin_cov_data[i]['C(S)'])
    chemkin_ch_s.append(chemkin_cov_data[i]['CH(S)'])
    chemkin_cch3_s.append(chemkin_cov_data[i]['CCH3(S)'])
    chemkin_cch2_s.append(chemkin_cov_data[i]['CCH2(S)'])

    
for i in range(0, 209):
    omkm_time.append(omkm_cov_data[i]['t(s)'])
    omkm_conv.append(omkm_mass_data[i]['Conv %'])
    omkm_o2.append(omkm_mass_data[i]['O2'])
    omkm_ch2ch2.append(omkm_mass_data[i]['CH2CH2'])
    omkm_pt_s.append(omkm_cov_data[i]['PT(S)'])
    omkm_c_s.append(omkm_cov_data[i]['C(S)'])
    omkm_ch_s.append(omkm_cov_data[i]['CH(S)'])
    omkm_cch3_s.append(omkm_cov_data[i]['CCH3(S)'])
    omkm_cch2_s.append(omkm_cov_data[i]['CCH2(S)'])
    
'''Interpolations'''
conv_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_conv),2)
o2_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_o2),2)
ch2ch2_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_ch2ch2),2)
pt_s_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_pt_s),2)
c_s_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_c_s),2)
ch_s_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_ch_s),2)
cch3_s_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_cch3_s),2)
cch2_s_inter = make_interp_spline(np.array(omkm_time), np.array(omkm_cch2_s),2)

'''ABSOLUTE MEAN AND MAX DIFF MASS'''
conv_diff = np.array(chemkin_conv) - np.array(conv_inter(chemkin_time))
conv_avg_diff = np.average(conv_diff)
o2_diff = np.array(chemkin_o2) - np.array(o2_inter(chemkin_time))
o2_avg_diff = np.average(o2_diff)
ch2ch2_diff = np.array(chemkin_ch2ch2) - np.array(ch2ch2_inter(chemkin_time))
ch2ch2_avg_diff = np.average(ch2ch2_diff)
overall_mass_diff = np.concatenate((conv_diff, o2_diff, ch2ch2_diff), axis = 0)
max_mass_mean_diff = max(abs(np.array(overall_mass_diff)))
print("Maximum Difference for Mass: " + str(max_mass_mean_diff))
print('Absolute Mean for Mass: ' + str(np.average(abs(overall_mass_diff))))


'''ABSOLUTE MEAN AND MAX DIFF COV'''
pt_s_diff = np.array(chemkin_pt_s) - np.array(pt_s_inter(chemkin_time))
pt_s_avg_diff = np.average(pt_s_diff)
c_s_diff = np.array(chemkin_c_s) - np.array(c_s_inter(chemkin_time))
c_s_avg_diff = np.average(c_s_diff)
ch_s_diff = np.array(chemkin_ch_s) - np.array(ch_s_inter(chemkin_time))
ch_s_avg_diff = np.average(ch_s_diff)
cch3_s_diff = np.array(chemkin_cch3_s) - np.array(cch3_s_inter(chemkin_time))
cch3_s_avg_diff = np.average(cch3_s_diff)
cch2_s_diff = np.array(chemkin_cch2_s) - np.array(cch2_s_inter(chemkin_time))
cch2_s_avg_diff = np.average(cch2_s_diff)
overall_cov_diff = np.concatenate((pt_s_diff, c_s_diff, ch_s_diff, cch3_s_diff,cch2_s_diff), axis = 0)
max_cov_mean_diff = max(abs(np.array(overall_cov_diff)))
print("Maximum Difference for Cov: " + str(max_cov_mean_diff))
print('Absolute Mean for Cov: ' + str(np.average(abs(overall_cov_diff))))
plt.rc('font', size=28)

plt.figure('Conversion')
chemkin_time_adjusted = [chemkin_time[i]*1000 for i in range(0, 254)]
plt.plot(chemkin_time_adjusted, conv_inter(chemkin_time), 'k', linewidth=3)
plt.plot(chemkin_time_adjusted, chemkin_conv, '--r', linewidth=3)
#plt.plot(chemkin_time, conv_inter(chemkin_time), '--b', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0, 0.006*1000])
plt.ylim([0, 1])
plt.xlabel('Time[s*$10^{-3}$]', fontsize=36)
plt.ylabel('Conversion %', fontsize=40)

plt.figure('O2 Mass')
plt.plot(omkm_time, omkm_o2, 'k', linewidth=3)
plt.plot(chemkin_time, chemkin_o2, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0, 0.0006])
plt.ylim([0, 0.35])
plt.xlabel('Time[s]', fontsize=36)
plt.ylabel('O2 Mass fraction', fontsize=40)

plt.figure('CH2CH2 Mass')
chemkin_time_adjusted = [chemkin_time[i]*100 for i in range(0, 254)]
plt.plot(chemkin_time_adjusted, ch2ch2_inter(chemkin_time), 'k', linewidth=3)
plt.plot(chemkin_time_adjusted, chemkin_ch2ch2, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0, 0.02*100])
plt.ylim([0, 0.25])
plt.xlabel('Time[s*$10^{-2}$]', fontsize=36)
plt.ylabel('CH2CH2 Mass fraction', fontsize=40)

plt.figure('Step Vacancies')
plt.plot(chemkin_time_adjusted, pt_s_inter(chemkin_time), 'k', linewidth=3)
plt.plot(chemkin_time_adjusted, chemkin_pt_s, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0, 0.025*100])
plt.ylim([0.4, 0.8])
plt.xlabel('Time[s*$10^{-2}$]', fontsize=36)
plt.ylabel('Step Vacancies', fontsize=40)

plt.figure('C(S) Coverage')
chemkin_time_adjusted = [chemkin_time[i]*1000 for i in range(0, 254)]
plt.plot(chemkin_time_adjusted, c_s_inter(chemkin_time), 'k', linewidth=3)
plt.plot(chemkin_time_adjusted, chemkin_c_s, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0, 0.001*1000])
plt.ylim([0, 0.35])
plt.xlabel('Time[s*$10^{-3}$]', fontsize=36)
plt.ylabel('C(S) Coverage', fontsize=40)

plt.figure('CH(S) Coverage')
plt.plot(chemkin_time_adjusted, ch_s_inter(chemkin_time), 'k', linewidth=3)
plt.plot(chemkin_time_adjusted, chemkin_ch_s, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0, 0.001*1000])
plt.ylim([0, 0.30])
plt.xlabel('Time[s*$10^{-3}$]', fontsize=36)
plt.ylabel('CH(S) Coverage', fontsize=40)

plt.figure('CCH3(S) Coverage')
plt.plot(omkm_time, omkm_cch3_s, 'k', linewidth=3)
plt.plot(chemkin_time, chemkin_cch3_s, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0, 0.002])
plt.ylim([0, 0.05])
plt.xlabel('Time[s]', fontsize=36)
plt.ylabel('CCH3(S) Coverage', fontsize=40)

plt.figure('CCH2(S) Coverage')
chemkin_time_adjusted = [chemkin_time[i]*100 for i in range(0, 254)]
plt.plot(chemkin_time_adjusted, cch2_s_inter(chemkin_time), 'k', linewidth=3)
plt.plot(chemkin_time_adjusted, chemkin_cch2_s, '--r', linewidth=3)
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlim([0, 0.015*100])
plt.ylim([0, 0.02])
plt.xlabel('Time[s*$10^{-2}$]', fontsize=36)
plt.ylabel('CCH2(S) Coverage', fontsize=40)