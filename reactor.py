"""
Reactor and numerical integration functions
"""
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import odeint, ode

#%% ODE functions for numerical integration
def dcdt(t, concentrations, stoichiometry, rate_expression,  para_dict, temperature=None):
    """Compute the derivatives
    """
    cur_rate = rate_expression(concentrations, para_dict, temperature)
    n_spec = len(stoichiometry)

    # dC/dt for each species
    cur_dcdt = np.zeros(n_spec)
    for i in range(n_spec):
        cur_dcdt[i] = stoichiometry[i] * cur_rate
 
    return cur_dcdt
    

def ode_solver(func, y0, t0, tf, *args):
    """Set up the ode solver 
    """
    # Construct the ode solver, ode45 with varying step size
    ans = []
    def get_ans(t, y):
        ans.append([t, *y])
        
    solver = ode(func).set_integrator('dopri5', rtol  = 1e-6, method='bdf')
    solver.set_solout(get_ans)
    #feed in argumenrs and initial conditions for odes
    solver.set_initial_value(y0, t0).set_f_params(*args) 
    solver.integrate(tf)
    ans = np.array(ans)
    
    return ans
    

class Reactor():
    """Reaction ODEs class"""
    
    def __init__(self, stoichiometry, P0, feed_composition, tf, name = 'simple reaction', temperature=None):
        """Initialize the constants"""
        self.stoichiometry = stoichiometry
        self.name = name
        self.P0 = P0
        self.feed_composition = feed_composition
        self.t0 = 0
        self.tf = tf
        self.temperature = temperature
        
        self.C0 = P0 * np.array(feed_composition)/np.sum(feed_composition)
    
    def get_profile(self, rate_expression, para_dict):
        """Numerical integration of the rate expression given the parameters"""
        
        tC_profile = ode_solver(dcdt, self.C0, self.t0, self.tf, self.stoichiometry, rate_expression, para_dict, self.temperature)
        
        return tC_profile
        
    def get_conversion(self, rate_expression, para_dict, species_index = 0):
        """Get the final conversion of a specie"""
        
        # Get the profile
        tC_profile = self.get_profile(rate_expression, para_dict)
        # Extract the final concentrations
        Cf = tC_profile[-1, species_index+1:] 
        # Compute the final percentage conversion
        xf = (self.C0[species_index] - Cf[species_index])/self.C0[species_index] * 100
        
        # Compute the final rates
        dcdt_f = dcdt(self.tf, Cf, self.stoichiometry, rate_expression, para_dict, self.temperature)
        
        return xf, dcdt_f
                 