"""
Reactor and numerical integration functions
"""
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import odeint, ode, solve_ivp


# Assume we have 
# n_rxn reactions and m_spec species  

#%% ODE functions for numerical integration
# def dcdt(t, concentrations, stoichiometry, rate_expression, para_dict, temperature=None):
#     """Compute the derivatives
#     """
#     cur_rate = rate_expression(concentrations, para_dict, temperature)
#     n_spec = len(stoichiometry)

#     # dC/dt for each species
#     cur_dcdt = np.zeros(n_spec)
#     for i in range(n_spec):
#         cur_dcdt[i] = stoichiometry[i] * cur_rate
 
#     return cur_dcdt
    
def dcdt(t, concentrations, stoichiometry, rate_expressions, para_dict, temperature=None):
    """Compute the derivatives
    """
    # stoichiometry matrix is in the shape of n_rxn * m_spec
    if not isinstance(stoichiometry[0], list):
        stoichiometry = [stoichiometry]
    stoichiometry = np.array(stoichiometry)   
    
    # expand expressions to a list
    if not isinstance(rate_expressions, list):
        rate_expressions = [rate_expressions]
    
    # rates are in the shape of 1 * n_rxn
    n_rxn = len(rate_expressions)
    cur_rate = np.zeros(n_rxn)
    for i in range(n_rxn):    
        cur_rate[i] = rate_expressions[i](concentrations, para_dict, temperature)
    
    # dC/dt for each species
    # dcdt is in the shape of 1 * m_spec 
    # use matrix multiplication
    cur_dcdt = np.matmul(cur_rate, stoichiometry)

    return cur_dcdt
    

def ode_solver(func, y0, t0, tf, *args):
    """
    Set up the ode solver 
    Older ode wrapper in scipy
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


def ode_solver_ivp(func, y0, t0, tf, t_eval, method, *options):
    """
    Set up the ode solver 
    Use solve_ivp
    """
    print(method)
    sol = solve_ivp(func, t_span=[t0, tf], y0=y0, method=method, t_eval=t_eval, args=(*options,))
    # Extract t and C from sol
    t_vec = sol.t
    C_vec = sol.y
    # ans is a matrix for tC_profile
    # 0th column is the time and ith column is the concentration of species i
    n_species, n_points = sol.y.shape
    ans = np.zeros((n_points, n_species+1))
    ans[:, 0] = t_vec
    ans[:, 1:] = C_vec.T
        
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

    def get_profile(self, rate_expression, para_dict, t_eval=None, method='LSODA'):
        """Numerical integration of the rate expression given the parameters"""
        print(method)
        tC_profile = ode_solver_ivp(dcdt, self.C0, self.t0, self.tf, t_eval, method, \
                                    self.stoichiometry, rate_expression, para_dict, self.temperature)
        
        return tC_profile
    
    def get_conversion(self, rate_expression, para_dict, species_index = 0, t_eval=None, method='LSODA'):
        """Get the final conversion of a specie"""
        
        # Get the profile
        tC_profile = self.get_profile(rate_expression, para_dict, t_eval, method)
        # Extract the final concentrations
        Cf = tC_profile[-1, species_index+1:] 
        # Compute the final percentage conversion
        xf = (self.C0[species_index] - Cf[species_index])/self.C0[species_index] * 100
        
        # Compute the final rates
        dcdt_f = dcdt(self.tf, Cf, self.stoichiometry, rate_expression, para_dict, self.temperature)
        
        return xf, dcdt_f
    
                 