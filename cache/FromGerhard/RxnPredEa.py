# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:44:53 2019

@author: GR Wittreich, P.E.

Reaction: A <=> R
 - Adsorption equilibrated
 - Surface reaction rate controlling

"""
from pmutt import constants as c
import numpy as np
import matplotlib.pyplot as plt
"""
TOF Method
"""


def get_TOF(T, Ea, A_sr, cat_moles,
            K_A_ref, H_A, T_A_ref, p_A,
            K_R_ref, H_R, T_R_ref, p_R,
            K_rxn_ref, H_rxn, T_rxn_ref):
    K_A = K_A_ref*np.exp(-H_A/c.R('kcal/mol/K')*(1/T - 1/T_A_ref))
    K_R = K_R_ref*np.exp(-H_R/c.R('kcal/mol/K')*(1/T - 1/T_R_ref))
    K_rxn = K_rxn_ref*np.exp(-H_rxn/c.R('kcal/mol/K')*(1/T - 1/T_rxn_ref))
    A_conc = (p_A/T/c.R('cm3 atm/mol/K'))*0+1
    k_rxn = A_conc*A_sr*T*np.exp(-Ea/c.R('kcal/mol/K')/T)
    rate = k_rxn*K_A*(p_A - p_R/K_rxn)/(1 + K_A*p_A + K_R*p_R)
    return rate


"""
Reactant
"""
p_A = 0.95               # Reactant partial pressure [atm]
H_A = -15.5              # Reactant enthalpy of adsorbtion [kcal/mol]
S_A = -30.5               # Reaction entropy of adsorbtion [cal/mol K]
T_A_ref = 298.15        # Reactant reference temperature [K]
G_A = H_A - T_A_ref*S_A*c.convert_unit(initial='cal', final='kcal')
K_A_ref = np.exp(-G_A/c.R('kcal/mol/K')/T_A_ref)
"""
Product
"""
p_R = 0.05               # Product partial pressure [atm]
H_R = -28.6              # Product enthalpy of adsorbtion [kcal/mol]
S_R = -39.0               # Product entropy of adsorbtion [cal/mol K]
T_R_ref = 298.15        # Product reference temperature [K]
G_R = H_R - T_R_ref*S_R*c.convert_unit(initial='cal', final='kcal')
K_R_ref = np.exp(-G_R/c.R('kcal/mol/K')/T_R_ref)
"""
Reaction
"""
H_rxn = -250.            # Reaction enthalpy [kcal/mol]
S_rxn = -10.0            # Reaction entropy [cal/mol K]
T_rxn_ref = 298.15        # Reaction reference temperature [K]
G_rxn = H_rxn - T_rxn_ref*S_rxn*c.convert_unit(initial='cal', final='kcal')
K_rxn_ref = np.exp(-G_rxn/c.R('kcal/mol/K')/T_rxn_ref)
"""
Misc
"""
A_sr = c.kb('J/K')/c.h('J s')  # Pre-exponential [1/s K]
cat_density = 2.5e-9  # Catalyst site denisty [mol/cm2]
cat_loading = 1000    # Catalyst loading [cm2/cm3]
cat_moles = cat_density*cat_loading  # mol/cm3

E_sr_low = 0*c.convert_unit(initial='eV/molecule', final='kcal/mol')
E_sr_high = 3*c.convert_unit(initial='eV/molecule', final='kcal/mol')
E_sr = np.linspace(E_sr_low, E_sr_high, 1000)
T_low = 300
T_high = 1500
T_sr = np.linspace(T_low, T_high, 1000)

xv, yv = np.meshgrid(T_sr, E_sr, indexing='xy')
TOF = np.log10(get_TOF(xv, yv, A_sr, cat_moles,
                       K_A_ref, H_A, T_A_ref, p_A,
                       K_R_ref, H_R, T_R_ref, p_R,
                       K_rxn_ref, H_rxn, T_rxn_ref))
TOF[np.logical_or((TOF >= 1), (TOF <= -3))] = np.NaN


plt.figure(1)
plt.contourf(xv, yv, TOF, 50, cmap='cool', alpha=1.0)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Temperature [K]', fontsize=18)
plt.ylabel('Ea$_{app}$ [kcal/mol]', fontsize=18)
plt.title('A --> R\nEstimated Reaction Rate', fontsize=18)
atm = plt.colorbar(ticks=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
atm.set_label('Log$_{10}$(TOF)', fontsize=14)
plt.tight_layout()
