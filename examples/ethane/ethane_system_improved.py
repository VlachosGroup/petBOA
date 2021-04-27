import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(3)

#Inlet Concentration Vals in M
C2H6_in = 0.0005
C2H4_in = 0
CH4_in = 0
H2_in = 0
CO2_in = 0.0005
CO_in = 0
H2O_in = 0
y0 = [C2H6_in, C2H4_in, CH4_in, H2_in, CO2_in, CO_in, H2O_in]

#Noise Param
noise = 0.00003

#Set parameter values
A0_EDH = 2.5E6
Ea_EDH = 125 #kJ/mol
A0_Hyd = 3.8E8
Ea_Hyd = 110 #kJ/mol
A0_RWGS = 1.9E6
Ea_RWGS = 70 #kJ/mol
T = 873 #K, note: only works for temperatures of 835, 848, 861, and 873K
                 #if a different temperature is desired, add the Kp values
                 #at that temperature to the Kp catalogs
P = 1 #atm, not used in code, just confirming that it runs at atmospheric pressure

#rxns:
#EDH: ethane dehydrogenation: C2H6 -> C2H4 + H2

#Hyd: hydrogenolysis: C2H6 + H2 -> CH4

#RWGS: Reverse water-gas shift: CO2 + H2 -> CO + H2O

#Rate equation form: rate_net = rate_fwd*(1-(Qp/Kp))
#Where the forward rate equation is rate_fwd = A0*exp(-Ea/RT)*[reactant]
#the (1-(Qp/Kp)) term correct for equilibrium
#Qp is the reaction quotiet: https://en.wikipedia.org/wiki/Reaction_quotient
#When the system is far from equilibrium, Qp -> 0, so rate_net -> rate_fwd
#When they system is close to equilibrium, Qp -> Kp, so rate_net -> 0

#Kp values calculated at different T using thermosolver
#Dictionary of temperature and associated Kp value
Kp_EDH = {'835': 0.0114,
          '848': 0.0157,
          '861': 0.0214,
          '873': 0.0281}
Kp_Hyd = {'835': 28700,
          '848': 24500,
          '861': 21100,
          '873': 18400}
Kp_RWGS = {'835': 0.296,
           '848': 0.321,
           '861': 0.347,
           '873': 0.372}

R = 8.31446261815324*(1/1000) #kJ/mol/K
R_bar = 8.31446261815324*(1/100) #(L*bar)/(mol*K)
#y = [C2H6, C2H4, CH4, H2, CO2, CO, H2O]
#p = [A0_EDH, Ea_EDH, A0_Hyd, Ea_Hyd, A0_RWGS, Ea_RWGS, T, Kp_EDH, Kp_Hyd, Kp_RWGS]
#parameters to estimate: [A0_EDH, Ea_EDH, A0_Hyd, Ea_Hyd, A0_RWGS, Ea_RWGS]

#Batch reactor simulation - d_concentration/d_time (units of M/s)
def dConcdt(t, y, p):
    #set Kp values
    Kp_EDH = p[7]
    Kp_Hyd = p[8]
    Kp_RWGS = p[9]
    
    #Calculate Qp values
    #Qp is ratio of pressures, to calculate pressure assume ideal gas law
    #P = Conc*RT
    #Assume standard pressure of P0 is 1 atm
    
    # p[6] is temperature 
    if y[0]!=0:
        Qp_EDH = ((y[1]*R_bar*p[6])*(y[3]*R_bar*p[6]))/(y[0]*R_bar*p[6])
    else:
        Qp_EDH = 0
    if y[0]!=0 and y[3]!=0:
        Qp_Hyd = ((y[2]*R_bar*p[6])*(y[2]*R_bar*p[6]))/((y[0]*R_bar*p[6])*(y[3]*R_bar*p[6]))
    else:
        Qp_Hyd = 0
    if y[3]!=0 and y[4]!=0:
        Qp_RWGS = ((y[5]*R_bar*p[6])*(y[6]*R_bar*p[6]))/((y[3]*R_bar*p[6])*(y[4]*R_bar*p[6]))
    else:
        Qp_RWGS = 0
    
    #Calculate rate values
    rate_net_EDH = (p[0]*np.exp(-p[1]/(R*p[6]))*y[0])*(1-(Qp_EDH/Kp_EDH))
    rate_net_Hyd = (p[2]*np.exp(-p[3]/(R*p[6]))*y[0]*y[3])*(1-(Qp_Hyd/Kp_Hyd))
    rate_net_RWGS = (p[4]*np.exp(-p[5]/(R*p[6]))*y[3]*y[4])*(1-(Qp_RWGS/Kp_RWGS))
    
    #Calculate concentration derivatives from rates and stoichiometry:
    #EDH: C2H6 -> C2H4 + H2
    #Hyd: C2H6 + H2 -> CH4
    #RWGS: CO2 + H2 -> CO + H2O
    dC2H6dt = -rate_net_EDH-rate_net_Hyd
    dC2H4dt = rate_net_EDH
    dCH4dt = 2*rate_net_Hyd
    dH2dt = rate_net_EDH-rate_net_Hyd-rate_net_RWGS
    dCO2dt = -rate_net_RWGS
    dCOdt = rate_net_RWGS
    dH2Odt = rate_net_RWGS
    
    return [dC2H6dt, dC2H4dt, dCH4dt, dH2dt, dCO2dt, dCOdt, dH2Odt]



#ODE solver - simulate batch system
t_lower = 0
t_upper = 10
t_span = [t_lower, t_upper]
num_points_sim = 101
t_eval_no_noise = np.linspace(t_lower, t_upper, num_points_sim)
t_eval_with_noise = [1,2.5,4,5.5,7,10]
num_points_collect = len(t_eval_with_noise)

#Set parameter vector
p_Kp_EDH = Kp_EDH[str(T)]
p_Kp_Hyd = Kp_Hyd[str(T)]
p_Kp_RWGS = Kp_RWGS[str(T)]
p = [A0_EDH, Ea_EDH, A0_Hyd, Ea_Hyd, A0_RWGS, Ea_RWGS, T, p_Kp_EDH, p_Kp_Hyd, p_Kp_RWGS]

sol_no_noise = solve_ivp(dConcdt, t_span=t_span, y0=y0, method='LSODA', 
                t_eval=t_eval_no_noise, args=(p, ))
sol_with_noise = solve_ivp(dConcdt, t_span=t_span, y0=y0, method='LSODA', 
                t_eval=t_eval_with_noise, args=(p, ))

#Plot the data without noise as smooth lines
f,ax = plt.subplots(1) 
ax.plot(sol_no_noise['t'],sol_no_noise['y'][0],label='C2H6')
ax.plot(sol_no_noise['t'],sol_no_noise['y'][1],label='C2H4')
ax.plot(sol_no_noise['t'],sol_no_noise['y'][2],label='CH4')
ax.plot(sol_no_noise['t'],sol_no_noise['y'][3],label='H2')
ax.plot(sol_no_noise['t'],sol_no_noise['y'][4],label='CO2')
ax.plot(sol_no_noise['t'],sol_no_noise['y'][5],label='CO')
ax.plot(sol_no_noise['t'],sol_no_noise['y'][6],label='H2O')

ax.set_xlabel('Time (sec)')
ax.set_ylabel('Concentration (mol/L)')
ax.legend()

#generate noisy data
data = np.zeros([num_points_collect,7])
time = np.zeros([num_points_collect])
for j in range(num_points_collect):
    time[j] = sol_with_noise['t'][j]
    data[j,0] = sol_with_noise['y'][0][j] + np.random.normal(loc=0,scale=noise)
    data[j,1] = sol_with_noise['y'][1][j] + np.random.normal(loc=0,scale=noise)
    data[j,2] = sol_with_noise['y'][2][j] + np.random.normal(loc=0,scale=noise)
    data[j,3] = sol_with_noise['y'][3][j] + np.random.normal(loc=0,scale=noise)
    data[j,4] = sol_with_noise['y'][4][j] + np.random.normal(loc=0,scale=noise)
    data[j,5] = sol_with_noise['y'][5][j] + np.random.normal(loc=0,scale=noise)
    data[j,6] = sol_with_noise['y'][6][j] + np.random.normal(loc=0,scale=noise)
    
#Ensure no concentrations are <0, if so set that point to 0
for i in range(num_points_collect):
    for j in range(7):
        if data[i,j]<0:
            data[i,j] = 0
 
#Plot the noisy data as a scatter plot        
ax.scatter(time,data[:,0])        
ax.scatter(time,data[:,1])        
ax.scatter(time,data[:,2])        
ax.scatter(time,data[:,3])        
ax.scatter(time,data[:,4])        
ax.scatter(time,data[:,5])        
ax.scatter(time,data[:,6])        

plt.show()
