
# Contact model example: 3d bouncing body with multiple contact points

# Standard imports
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add workspace and package directories to the path
import os
import sys
sim_pkg_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sim_pkg_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
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

r = 0.1
cont_points1 = np.array([[r, 0, 0], [0, r, 0], [-r, 0, 0], [0, -r, 0]])
m = 1
I =  np.eye(3)
e1 = 0.5
e2 = 1
mp_3d_cont =  contact.ImpulseContact(cont_points1, m, I, e=e2, mu=1.5,
                                     debug_ctl=contact_debug_ctl)

# Example 3d system dynamics function
def sysdyn_3d_4p(t, x):
    '''
    System dynamics for 3d 4-point system

    State order: x, y, z, dx, dy, dz
    '''

    g = 9.8

    dx_dt = np.array([x[3], x[4], x[5], 0, 0, g])

    return dx_dt # Return change in states

# Direct state modification from contact function
def state_mod_cont(t, x):
    '''
    Function for direct change of system states in sim due to contact model
    '''

    debug = False

    # Apply contact model to system dynamics
    pos_cont = x[:3]
    rot_cont = state_mod_cont.rot # Passed in from numerical integrator
    lin_vel_cont = x[3:]
    ang_vel_cont = np.zeros(3) # Not modeled
    del_lin_vel, del_ang_vel, df_cont_tracking =  mp_3d_cont.contact_main(
        pos_cont, rot_cont, lin_vel_cont, ang_vel_cont, t=t)
    
    if debug: # Debug prints
        print('t = %f, del_vel = %f' % (t, del_lin_vel[2]))
    
    x_out = x + np.concatenate((np.zeros(3), del_lin_vel))

    return x_out

state_mod_cont.rot = R.from_matrix(np.eye(3)) # Rotation not modeled

# Setup simulation control parameters
output_ctl1 = { # Output file control
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'ex_3d_4p.csv', # Name for output file
    'file': True, # Save output file
    'plots': True # Save output plots
}

tspan = (0, 5)
x0 = np.array([0, 0, -1.0, 0.1, 0.2, 0])
timestep = 0.01
state_names = ['x', 'y', 'z', 'dx', 'dy', 'dz']

# Run the simulation
sim_test = simulator.Simulator(tspan, x0, sysdyn_3d_4p, timestep=timestep,
                               state_mod_fun=state_mod_cont,
                               state_names=state_names, output_ctl=output_ctl1)
sim_test.compute()
sim_test.plot_states()
