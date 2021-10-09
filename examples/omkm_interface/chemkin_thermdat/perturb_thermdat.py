import pmutt.constants
from pmutt.io import thermdat

thermo_data = thermdat.read_thermdat(filename='thermdat')
R = pmutt.constants.R(units='kJ/mol/K')

for species in thermo_data:
    print("Species {} Gibbs at 800K is {} kJ/mol".format(species.name, species.get_G(800, 'kJ/mol')))

param_groundtruth = {'N2(T)': 6.959765516670089,
                     'N(T)': -4.629843081311661,
                     'TS4_N2(T)': -16.88979206205836,
                     'N2(S)': 5.97427204174473,
                     'N(S)': 17.524267700455674,
                     'TS4_N2(S)': -10.841388069480615}

for species in thermo_data:
    if species.name in param_groundtruth.keys():
        print(species.name)
        delta_H = (param_groundtruth[species.name] / R)
        species.a_high[-2] += delta_H
        species.a_low[-2] += delta_H

for species in thermo_data:
    print("Species {} Gibbs at 800K is {} kJ/mol".format(species.name, species.get_G(800, 'kJ/mol')))

thermdat.write_thermdat(thermo_data, filename='thermdat_new')
