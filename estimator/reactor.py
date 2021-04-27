"""
Reactor and numerical integration functions
"""
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import odeint, ode, solve_ivp


# Assume we have 
# n_rxn reactions and m_spec species  

#%% ODE functions for numerical integration

# Older version
def dcdt_1d(t, concentrations, stoichiometry, rate_expression, para_dict, temperature):
    """
    Compute the derivatives
    """
    cur_rate = rate_expression(concentrations, para_dict, temperature)
    n_spec = len(stoichiometry)

    # dC/dt for each species
    cur_dcdt = np.zeros(n_spec)
    for i in range(n_spec):
        cur_dcdt[i] = stoichiometry[i] * cur_rate
 
    return cur_dcdt
    

def ode_solver(func, y0, t0, tf, *func_inputs):
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
    solver.set_initial_value(y0, t0).set_f_params(*func_inputs) 
    solver.integrate(tf)
    ans = np.array(ans)
    
    return ans


# Current verison
def dcdt(t, concentrations, stoichiometry, rate_expressions, para_dict, names=None, temperature=None, *rate_inputs):
    """
    Compute the derivatives for multiple parallel reactions
    """
    # stoichiometry matrix is in the shape of n_rxn * m_spec
    if not isinstance(stoichiometry[0], list):
        stoichiometry = [stoichiometry]
        n_rxn = 1
    else:
        n_rxn = len(stoichiometry)
            
    # expand expressions to a list
    if not isinstance(rate_expressions, list):
        rate_expressions = n_rxn * [rate_expressions]
    if not isinstance(names, list):
        names = n_rxn * [names]
    
    if (n_rxn != len(rate_expressions)) or n_rxn != len(names):
        raise ValueError("Input stoichiometry matrix must equal to the number of input rate expressions or names")
    
    # rates are in the shape of 1 * n_rxn
    cur_rate = np.zeros(n_rxn)
    for i in range(n_rxn):    
        cur_rate[i] = rate_expressions[i](concentrations,  para_dict, stoichiometry[i],  names[i], temperature, *rate_inputs)
    
    # dC/dt for each species
    # dcdt is in the shape of 1 * m_spec 
    # use matrix multiplication
    cur_dcdt = np.matmul(cur_rate, np.array(stoichiometry))

    return cur_dcdt
    

def ode_solver_ivp(func, y0, t0, tf, t_eval, method, *func_inputs):
    """
    Set up the ode solver 
    Use solve_ivp
    """
    sol = solve_ivp(func, t_span=[t0, tf], y0=y0, method=method, t_eval=t_eval, args=(*func_inputs,))
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
    
    def __init__(self, stoichiometry, tf, P0=None, feed_composition=None, C0=None, names=None, temperature=None):
        """Initialize the constants"""
        self.stoichiometry = stoichiometry
        self.names = names
        self.P0 = P0
        self.feed_composition = feed_composition
        self.t0 = 0
        self.tf = tf
        self.temperature = temperature
        
        if (P0 is not None) and (feed_composition is not None):
            self.C0 = P0 * np.array(feed_composition)/np.sum(feed_composition)
        elif C0 is not None: 
            self.C0 = C0
        else:
            raise ValueError("Must input P0, feed composition or C0")

    def get_profile(self, rate_expressions, para_dict, t_eval=None, method='LSODA'):
        """Numerical integration of the rate expression given the parameters"""
        #print(method)
        tC_profile = ode_solver_ivp(dcdt, self.C0, self.t0, self.tf, t_eval, method, \
                                    self.stoichiometry, 
                                    rate_expressions, 
                                    para_dict, 
                                    self.names,
                                    self.temperature)
        
        return tC_profile
    
    def get_conversion(self, rate_expressions, para_dict, species_index = 0, t_eval=None, method='LSODA'):
        """Get the final conversion of a specie"""
        
        # Get the profile
        tC_profile = self.get_profile(rate_expressions, para_dict, t_eval, method)
        # Extract the final concentrations
        Cf = tC_profile[-1, species_index+1:] 
        # Compute the final percentage conversion
        xf = (self.C0[species_index] - Cf[species_index])/self.C0[species_index] * 100
        
        # Compute the final rates
        dcdt_f = dcdt(self.tf, Cf, self.stoichiometry, rate_expressions, para_dict, self.names, self.temperature)
        
        return xf, dcdt_f
    
                 