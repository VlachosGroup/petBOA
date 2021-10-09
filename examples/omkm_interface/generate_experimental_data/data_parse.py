import os
import time

import numpy as np
import pandas as pd
import yaml


# Define all functions
# define some utility functions
def get_value(string_data):
    if isinstance(string_data, str):
        if len(string_data.split(' ')) > 1:
            return float(string_data.split(' ')[0])
        else:
            return float(string_data)
    else:
        return string_data


def get_unit(string_data):
    if isinstance(string_data, str):
        if len(string_data.split(' ')) > 1:
            return string_data.split(' ')[-1]
        else:
            return ''
    else:
        return 'NA'


def read_all_files(dirPath):
    # Molar Composition
    _df1 = pd.read_csv(dirPath+'/gas_mole_ss.csv')
    _df1 = _df1.rename(columns={'N2': 'N2_molefrac',
                                'NH3': 'NH3_molefrac',
                                'H2': 'H2_molefrac'
                                })

    # Optional Mass Fractions
    _df2 = pd.read_csv(dirPath+'/gas_mass_ss.csv')
    _df2 = _df2.rename(columns={'N2': 'N2_massfrac',
                                'NH3': 'NH3_massfrac',
                                'H2': 'H2_massfrac'
                                })

    # Molar Rate or production for gas species in (kmol/sec)
    _df3 = pd.read_csv(dirPath+'/gas_sdot_ss.csv')
    _df3 = _df3.rename(columns={'N2': 'dN2_rate(kmol/sec)',  # in kmol/sec
                                'NH3': 'dNH3_rate(kmol/sec)',  # in kmol/sec
                                'H2': 'dH2_rate(kmol/sec)'  # in kmol/sec
                                })

    _yaml_dict = yaml.safe_load(open(dirPath+'/reactor.yaml', 'r'))
    return _df1, _df2, _df3, _yaml_dict


def modify_data_frames(_fulldf, _yaml_dict):
    # Calculate mass conversion
    if not (_fulldf['NH3_molefrac'][0] == 0.0):
        _fulldf['NH3_Conversion_mass'] = (_fulldf['NH3_massfrac'].iloc[0] - (
            _fulldf['NH3_massfrac']) / (_fulldf['NH3_massfrac'].iloc[0]))
        print(_fulldf['NH3_Conversion_mass'])
    # Operation condition and reactor information
    vol_flowrate = None
    pressure, pressure_unit = None, None
    if get_unit(_yaml_dict['inlet_gas']['flow_rate']) == 'cm3/s':
        vol_flowrate = get_value(_yaml_dict['inlet_gas']['flow_rate'])
    if _yaml_dict['reactor']['pressure_mode'] == 'isobaric':
        pressure = get_value(_yaml_dict['reactor']['pressure'])
        pressure_unit = get_unit(_yaml_dict['reactor']['pressure'])
    # else:
    # parse from reactor state file to get changes across the length of PFR
    temperature = None
    if _yaml_dict['reactor']['temperature_mode'] == 'isothermal':
        temperature = get_value(_yaml_dict['reactor']['temperature'])
        temperature_unit = get_unit(_yaml_dict['reactor']['temperature'])
    # else:
        # parse from reactor state file to get changes across the length of PFR
    volume = get_value(_yaml_dict['reactor']['volume'])
    residence_time = volume/vol_flowrate

    _fulldf['pressure'+'('+pressure_unit+')'] = pressure
    _fulldf['temperature'+'(K)'] = temperature
    _fulldf['residence_time'+'(sec)'] = residence_time
    _fulldf['vol_flow_rate'+'(cm3/sec)'] = vol_flowrate
    _fulldf['P_N2'+'('+pressure_unit+')'] = _fulldf['N2_molefrac']*pressure
    _fulldf['P_NH3'+'('+pressure_unit+')'] = _fulldf['NH3_molefrac']*pressure
    _fulldf['P_H2'+'('+pressure_unit+')'] = _fulldf['H2_molefrac']*pressure
    _fulldf['CSTR_slice_no'] = np.arange(1,len(_fulldf)+1)

    #edit dataframe such that each row represents a CSTR slice
    _fulldf = _fulldf.drop(_fulldf.index[0],axis=0)
    return _fulldf


# Get the list of all files and directories in a directory tree
# at given path
listOfDirs = list()
listOfFiles = list()
listOfDataFrames = list()
baseDirName = os.getcwd()
for (dirpath, dirnames, filenames) in os.walk(baseDirName):
    listOfDirs += [dirpath]
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]
for dirName in listOfDirs:
    yamlFile = dirName+'/reactor.yaml'
    print (dirName)
    if os.path.isfile(yamlFile) and os.path.getsize(yamlFile) > 0:
         # now start data parsing and adding to the csv file
         df1, df2, df3, yaml_dict = read_all_files(dirName)
         df1 = pd.merge(df1, df2, on='t(s)')
         df1 = pd.merge(df1, df3, on='t(s)')
         fulldf = pd.merge(df1, df2, on='t(s)')
         fulldf = pd.merge(fulldf, df3, on='t(s)')
         listOfDataFrames.append(modify_data_frames(df1, yaml_dict))
finalDF = pd.concat(listOfDataFrames, ignore_index=True)
start = time.time()
finalDF.to_csv('all_data.csv')
print("For csv time in sec {}".format(time.time()-start))
