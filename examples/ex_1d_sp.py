
# Contact model example: 1d bouncing body with single contact point

# Standard imports
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add workspace and package directories to the path
import os
import sys
sim_pkg_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sim_pkg_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                             '../..'))
sys.path.insert(0, sim_pkg_path1)
sys.path.insert(0, sim_pkg_path2)

# Workspace package imports
import contact
from amls_sim import simulator

# Configure contact object for bouncing point
contact_debug_ctl = { # Default debug settings in dict
    "main": False, # contact_main method
    "contact": False, # contact_calc methods
    "phase": False, # compression/extension/restitution methods
    "deriv": False, # restitution_deriv/compression_deriv methods
    "ancillary": False, # Ancillary helper methods
    "file": False # Save debug prints to a text file
}
cont_points = np.array([[0, 0, 0]])
m = 1
I =  np.eye(3)
e1 = 0.5
e2 = 1
sp_1d_cont =  contact.ImpulseContact(cont_points, m, I, e=e1, 
                                     debug_ctl=contact_debug_ctl)

# Example 1d system dynamics function
def sysdyn_1d_sp(t, x):

    g = 9.8

    dx_dt = np.array([x[1], g]) # Initial system dynamics

    return dx_dt # Return change in states

# Direct state modification from contact function
def state_mod_cont(t, x):
    '''
    Function for direct change of system states in sim due to contact model
    '''

    debug = False

    # Apply contact model to system dynamics
    pos_cont = np.array([0, 0, x[0]])
    rot_cont = state_mod_cont.rot # Passed in from numerical integrator
    lin_vel_cont = np.array([0, 0, x[1]])
    ang_vel_cont = np.zeros(3) # Not relevant in 1d case
    del_lin_vel, del_ang_vel, df_cont_tracking =  sp_1d_cont.contact_main(
        pos_cont, rot_cont, lin_vel_cont, ang_vel_cont, t=t)
    
    if debug: # Debug prints
        print('t = %f, del_vel = %f' % (t, del_lin_vel[2]))
    
    x_out = x + np.array([0, del_lin_vel[2]])

    return x_out

state_mod_cont.rot = R.from_matrix(np.eye(3)) # Rotation doesnt matter

# Setup simulation control parameters
output_ctl1 = { # Output file control
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'ex_1d_sp_bounce.csv', # Name for output file
    'file': True, # Save output file
    'plots': True # Save output plots
}

tspan = (0, 5)
x0 = np.array([-1, 0])
timestep = 0.01
state_names = ['position', 'velocity']

# Run the simulation
sim_test = simulator.Simulator(tspan, x0, sysdyn_1d_sp, timestep=timestep,
                               state_mod_fun=state_mod_cont,
                               state_names=state_names, output_ctl=output_ctl1)
sim_test.compute()
sim_test.plot_states()






