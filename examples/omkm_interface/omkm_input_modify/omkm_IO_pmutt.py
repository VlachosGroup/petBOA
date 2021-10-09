# coding: utf-8
import os
import random
from pathlib import Path

import numpy as np
import pmutt.constants
from pmutt import pmutt_list_to_dict
from pmutt.empirical.nasa import Nasa
from pmutt.empirical.references import Reference, References
from pmutt.io.excel import read_excel
from pmutt.io.omkm import organize_phases, write_cti, write_yaml
from pmutt.mixture.cov import PiecewiseCovEffect
from pmutt.omkm.reaction import BEP, SurfaceReaction
from pmutt.omkm.units import Units
from pmutt.io import thermdat

R = pmutt.constants.R(units='kJ/mol/K')
try:
    file_path = os.path.dirname(__file__)
except NameError:
    file_path = Path().resolve()

os.chdir(file_path)
input_path = Path('inputs/NH3_Input_Data.xlsx').resolve()

# Section Reading data

# First, we will designate the units to write the CTI and YAML file.
units_data = read_excel(io=input_path, sheet_name='units')[0]
units = Units(**units_data)

# Second, we will open the input spreadsheet and read the `refs` sheet.
try:
    refs_data = read_excel(io=input_path, sheet_name='refs')
except:
    # If references are not used, skip this section
    print(('The "refs" sheet could not be found in {}.'
           'Skipping references'.format(input_path)))
    refs = None
else:
    refs = [Reference(**ref_data) for ref_data in refs_data]
    refs = References(references=refs)

# Third, we will use the ``refs`` defined before and the ``species`` sheet to convert statistical mechanical data to
# (https://vlachosgroup.github.io/pMuTT/api/empirical/nasa/pmutt.empirical.nasa.Nasa.html#pmutt.empirical.nasa.Nasa).
# Read the species' data
species_data = read_excel(io=input_path, sheet_name='species')

# Create NASA polynomials from the species
species = [Nasa.from_model(references=refs, **ind_species_data) \
           for ind_species_data in species_data]

random.seed(np.pi)

species_to_perturb = ["N(T)", "N2(T)", "N(S)", "N2(S)", "TS4_N2(T)", "TS4_N2(S)"]
param_ground_truth = {}
for spec in species:
    if spec.name in species_to_perturb:
        delta_H = (20.0 / R) * random.uniform(-1, 1)
        spec.a_high[-2] += delta_H
        spec.a_low[-2] += delta_H
        param_ground_truth[spec.name] = delta_H * R
print("These are the perturbation values {} for species {}".format(param_ground_truth,
                                                                   species_to_perturb))
# Optionally you could also randomly perturb all species as
# shown below.

# for spec in species:
#     if "TS" not in spec.name and "RU" not in spec.name and "(T)" not in spec.name and "(S)" not in spec.name:
#         delta_H = (20.0/R) * random.uniform(-1,1)
#         spec.a_high[-2] += delta_H
#         spec.a_low[-2] += delta_H

# Reading BEP (optional)
# Next, we read the BEP relationships to include.
try:
    beps_data = read_excel(io=input_path, sheet_name='beps')
except:
    print(('The "beps" sheet could not be found in {}. '
           'Skipping BEPs'.format(input_path)))
    beps = None
    species_with_beps = species.copy()
else:
    beps = [BEP(**bep_data) for bep_data in beps_data]
    species_with_beps = species + beps

# Read reactions
# Convert species to dictionary for easier reaction assignment
species_with_beps_dict = pmutt_list_to_dict(species_with_beps)

reactions_data = read_excel(io=input_path, sheet_name='reactions')
reactions = [SurfaceReaction.from_string(species=species_with_beps_dict, **reaction_data) \
             for reaction_data in reactions_data]

# Read lateral interactions (optional)
# After, we read lateral interactions to include.
try:
    interactions_data = read_excel(io=input_path,
                                   sheet_name='lateral_interactions')
except:
    # If no lateral interactions exist, skip this section
    print(('The "lateral_interactions" sheet could not be found in {}.'
           'Skipping lateral interactions'.format(input_path)))
    interactions = None
else:
    interactions = [PiecewiseCovEffect(**interaction_data) \
                    for interaction_data in interactions_data]

# Reading Phases
# Finally, we read the phases data from Excel and organize it for use in OpenMKM.
# Read data from Excel sheet about phases
phases_data = read_excel(io=input_path, sheet_name='phases')
phases = organize_phases(phases_data, species=species, reactions=reactions,
                         interactions=interactions)

# ## Write YAML File
# The YAML file specifying the reactor configuration can be written using the
# [``write_yaml``](https://vlachosgroup.github.io/pMuTT/api/kinetic_models/omkm/pmutt.io.omkm.write_yaml.html) function.
# Note that if:
# - ``units`` is not specified, float values are assumed to be in SI units
# - ``units`` is specified, float values are consistent with ``unit``'s attributes
# - you would like a quantity to have particular units, pass the value as a string with the units  (e.g. "10 cm3/s").
Path('outputs').mkdir(exist_ok=True)
yaml_path = 'outputs/reactor.yaml'
reactor_data = read_excel(io=input_path, sheet_name='reactor')[0]
write_yaml(filename=yaml_path, phases=phases, units=units, **reactor_data)

# ## Write CTI File
# The CTI file species the thermodynamics and kinetics of the system.
# It can be written using
# (https://vlachosgroup.github.io/pMuTT/api/kinetic_models/omkm/pmutt.io.omkm.write_cti.html#pmutt.io.omkm.write_cti).
# Note: We take the reactor operating conditions from YAML file to calculate thermodynamic and kinetic parameters.
cti_path = 'outputs/thermo.cti'
use_motz_wise = True
T = reactor_data['T']

write_cti(reactions=reactions, species=species, phases=phases, units=units,
          lateral_interactions=interactions, filename=cti_path,
          use_motz_wise=use_motz_wise, T=T, P=1.)
thermdat_data = species.copy()
for spec in thermdat_data:
    try:
        if spec.phase.name == "gas":
            spec.phase = "G"
        elif spec.phase.name in ["terrace", "step", "bulk"]:
            spec.phase = "S"
    except:
        if spec.phase is None:
            spec.phase = "S"
thermdat.write_thermdat(thermdat_data, filename='outputs/thermdat')

# Finally save the perturbation ground truth values in a text file.
file = open('outputs/param_groundtruth.txt', 'w')
file.write(str(param_ground_truth))
file.close()
