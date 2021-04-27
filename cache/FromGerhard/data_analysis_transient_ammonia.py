# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:54:18 2020

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

excel_path = r'C:\Users\djrob\Desktop\Work\Fall_2020\openMKM v Chemkin\Ammonia\ammonia_transient_data.xlsx'
chemkin_cov_data = read_excel(io=excel_path, sheet_name='chemkin_cov')
openmkm_cov_data = read_excel(io=excel_path, sheet_name='omkm_cov')
chemkin_mass_data = read_excel(io=excel_path, sheet_name='chemkin_mass')
openmkm_mass_data = read_excel(io=excel_path, sheet_name='omkm_mass')
chemkin_time = []
chemkin_conv = []
openmkm_time = []
openmkm_conv = []
chemkin_ru_s = []
chemkin_ru_t = []
openmkm_ru_s = []
openmkm_ru_t = []
omkm_H_s = []
omkm_H_t = []
omkm_NH_s = []
omkm_NH_t = []
chemkin_H_s = []
chemkin_H_t = []
chemkin_NH_s = []
chemkin_NH_t = []
omkm_H2 = []
chemkin_H2 = []
omkm_N2 = []
chemkin_N2 = []

for i in range(0, 1002):
    chemkin_time.append(chemkin_cov_data[i]['Time [s]'])
    chemkin_conv.append(chemkin_mass_data[i]['Conv %'])
    chemkin_ru_s.append(chemkin_cov_data[i]['RU(S)'])
    chemkin_ru_t.append(chemkin_cov_data[i]['RU(T)'])
    chemkin_H_s.append(chemkin_cov_data[i]['H(S)'])
    chemkin_H_t.append(chemkin_cov_data[i]['H(T)'])
    chemkin_NH_s.append(chemkin_cov_data[i]['NH(S)'])
    chemkin_NH_t.append(chemkin_cov_data[i]['NH(T)'])
    chemkin_H2.append(chemkin_mass_data[i]['H2'])
    chemkin_N2.append(chemkin_mass_data[i]['N2'])
    
for i in range(0, 329):
    openmkm_time.append(openmkm_mass_data[i]['t(s)'])
    openmkm_conv.append(openmkm_mass_data[i]['Conv %'])
    openmkm_ru_s.append(openmkm_cov_data[i]['RU(S)'])
    openmkm_ru_t.append(openmkm_cov_data[i]['RU(T)'])
    omkm_H_s.append(openmkm_cov_data[i]['H(S)'])
    omkm_H_t.append(openmkm_cov_data[i]['H(T)'])
    omkm_NH_s.append(openmkm_cov_data[i]['NH(S)'])
    omkm_NH_t.append(openmkm_cov_data[i]['NH(T)'])
    omkm_H2.append(openmkm_mass_data[i]['H2'])
    omkm_N2.append(openmkm_mass_data[i]['N2'])
    
interpol_openmkm = make_interp_spline(np.array(openmkm_time), np.array(openmkm_conv),2)
interpol_step = make_interp_spline(np.array(openmkm_time), np.array(openmkm_ru_s),2)
interpol_terr = make_interp_spline(np.array(openmkm_time), np.array(openmkm_ru_t),2)
h2_interpol = make_interp_spline(openmkm_time, omkm_H2, 2)
n2_interpol = make_interp_spline(openmkm_time, omkm_N2, 2)
step_interpol = make_interp_spline(openmkm_time, openmkm_ru_s,2)
terr_interpol = make_interp_spline(openmkm_time, openmkm_ru_t, 2)
h_s_interpol = make_interp_spline(openmkm_time, omkm_H_s, 2)
h_t_interpol = make_interp_spline(openmkm_time, omkm_H_t, 2)
nh_s_interpol = make_interp_spline(openmkm_time, omkm_NH_s, 2)
nh_t_interpol = make_interp_spline(openmkm_time, omkm_NH_t, 2)


'''ABSOLUTE MEAN AND MAX DIFF MASS'''
conv_diff = np.array(chemkin_conv) - np.array(interpol_openmkm(chemkin_time))
conv_avg_diff = np.average(conv_diff)
H2_diff = np.array(chemkin_H2) - np.array(h2_interpol(chemkin_time))
H2_avg_diff = np.average(H2_diff)
N2_diff = np.array(chemkin_N2) - np.array(n2_interpol(chemkin_time))
N2_avg_diff = np.average(N2_diff)
overall_mass_diff = np.concatenate((conv_diff, H2_diff, N2_diff), axis = 0)
max_mass_mean_diff = max(abs(np.array(overall_mass_diff)))
print("Maximum Difference for Mass: " + str(max_mass_mean_diff))
print('Absolute Mean for Mass: ' + str(np.average(abs(overall_mass_diff))))

'''ABSOLUTE MEAN AND MAX DIFF COV'''
ru_s_diff = np.array(chemkin_ru_s) - np.array(interpol_step(chemkin_time))
ru_s_avg_diff = np.average(ru_s_diff)
ru_t_diff = np.array(chemkin_ru_t) - np.array(interpol_terr(chemkin_time))
ru_t_avg_diff = np.average(ru_t_diff)
h_s_diff = np.array(chemkin_H_s) - np.array(h_s_interpol(chemkin_time))
h_s_avg_diff = np.average(h_s_diff)
h_t_diff = np.array(chemkin_H_t) - np.array(h_t_interpol(chemkin_time))
h_t_avg_diff = np.average(h_t_diff)
nh_s_diff = np.array(chemkin_NH_s) - np.array(nh_s_interpol(chemkin_time))
nh_s_avg_diff = np.average(nh_s_diff)
nh_t_diff = np.array(chemkin_NH_t) - np.array(nh_t_interpol(chemkin_time))
nh_t_avg_diff = np.average(nh_t_diff)
overall_cov_diff = np.concatenate((ru_s_diff, ru_t_diff, h_s_diff, h_t_diff, nh_s_diff, nh_t_diff), axis = 0)
max_cov_mean_diff = max(abs(np.array(overall_cov_diff)))
print("Maximum Difference for Cov: " + str(max_cov_mean_diff))
print('Absolute Mean for Cov: ' + str(np.average(abs(overall_cov_diff))))
plt.rc('font', size=28)

plt.figure('Conversion')
omkm = plt.plot(openmkm_time, openmkm_conv, 'k', linewidth=3, label = 'oMKM')
chemkin = plt.plot(chemkin_time, chemkin_conv, '--r', linewidth=3, label = 'Chemkin')
#plt.plot(chemkin_time, interpol_openmkm(chemkin_time), '--b', linewidth=3)
#plt.xticks(fontsize=32)
plt.xlim([0, 25])
plt.ylim([0, 0.65])
plt.xlabel('Time[s]', fontsize=40)
plt.ylabel('Conversion %', fontsize=40)
plt.legend(loc ='lower right')


plt.figure('Step Vacancies')
plt.plot(chemkin_time, chemkin_ru_s, '--r', linewidth=3, label = 'Chemkin')
plt.plot(openmkm_time, openmkm_ru_s, 'k', linewidth=3, label = 'oMKM')
plt.xlim([0, 5])
plt.ylim([0.3, 0.5])
#plt.xticks(fontsize=32)
plt.yticks(np.arange(.30, .51, step = 0.05))
plt.xlabel('Time[s]', fontsize=40)
plt.ylabel('RU(S) Vacancies', fontsize=40)
plt.legend(loc ='lower right')

plt.figure('Terrace Vacancies')
plt.plot(chemkin_time, chemkin_ru_t, '--r', linewidth=3)
plt.plot(openmkm_time, openmkm_ru_t, 'k', linewidth=3)
plt.xlim([0, 8])
plt.ylim([0, 0.5])
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlabel('Time[s]', fontsize=40)
plt.ylabel('RU(T) Vacancies', fontsize=40)

plt.figure('H(S) Coverage')
plt.plot(chemkin_time, chemkin_H_s, '--r', linewidth=3)
plt.plot(openmkm_time, omkm_H_s, 'k', linewidth=3)
plt.xlim([0, 10])
plt.ylim([0.00, 0.35])
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlabel('Time[s]', fontsize=40)
plt.ylabel('H(S) Coverage', fontsize=40)

plt.figure('H(T) Coverage')
plt.plot(chemkin_time, chemkin_H_t, '--r', linewidth=3)
plt.plot(openmkm_time, omkm_H_t, 'k', linewidth=3)
plt.xlim([0, 15])
plt.ylim([0, 0.3])
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlabel('Time[s]', fontsize=40)
plt.ylabel('H(T) Coverage', fontsize=40)

plt.figure('NH(S) Coverage')
plt.plot(chemkin_time, chemkin_NH_s, '--r', linewidth=3)
plt.plot(openmkm_time, omkm_NH_s, 'k', linewidth=3)
plt.xlim([0, 8])
plt.ylim([0, 0.50])
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlabel('Time[s]', fontsize=40)
plt.ylabel('NH(S) Coverage', fontsize=40)

plt.figure('NH(T) Coverage')
plt.plot(chemkin_time, chemkin_NH_t, '--r', linewidth=3, label = 'Chemkin')
plt.plot(openmkm_time, omkm_NH_t, 'k', linewidth=3, label = 'oMKM')
plt.xlim([0, 10])
plt.ylim([0.2, 0.7])
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlabel('Time[s]', fontsize=40)
plt.ylabel('NH(T) Coverage', fontsize=40)
plt.legend(loc ='upper right')

plt.figure('H2 Mass')
plt.plot(chemkin_time, chemkin_H2, '--r', linewidth=3)
plt.plot(openmkm_time, omkm_H2, 'k', linewidth=3)
plt.xlim([0, 50])
plt.ylim([0, .12])
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlabel('Time[s]', fontsize=40)
plt.ylabel('H2 Mass Fraction', fontsize=40)

plt.figure('N2 Mass')
plt.plot(chemkin_time, chemkin_N2, '--r', linewidth=3)
plt.plot(openmkm_time, omkm_N2, 'k', linewidth=3)
plt.xlim([0, 25])
plt.ylim([0, 0.55])
#plt.xticks(fontsize=32)
#plt.yticks(fontsize=32)
plt.xlabel('Time[s]', fontsize=40)
plt.ylabel('N2 Mass Fraction', fontsize=40)