
'''
Example script of a 2d rod falling through space, meant to demonstrate both 
single-point and multi-point collisions in a simple environment
'''

# Standard imports
import numpy as np
from numpy import linalg as LA

import pandas as pd

from scipy.spatial.transform import Rotation as Rot

from math import pi

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = ["Latin Modern Roman"]

from types import MethodType

# Add package and workspace directories to the path
import os
import sys
sim_pkg_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sim_pkg_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0, sim_pkg_path1)
sys.path.insert(0, sim_pkg_path2)


# Workspace package imports
from amls_impulse_contact import contact
from amls_impulse_contact import utility_functions as uf_cont

from amls_sim import simulator

# ----------------------- Contact Object Configuration ----------------------- #

contact_debug_ctl = { # Debug settings dict
    "main": False, # contact_main method
    "contact": False, # contact_calc methods
    "count_every": False, # contact_count method every call
    "count_pos": False, # contact_count method positive counts
    "phase": False, # compression/extension/restitution methods
    "deriv": False, # restitution_deriv/compression_deriv methods
    "ancillary": False, # Ancillary helper methods
    "file": False # Save debug prints to a text file
}

rod_len = 1 # Unit length (m)
rod_contact_points_R = np.array([[0, rod_len/2, 0], # x,y,z by row
                               [0, -rod_len/2, 0]]) 
m = 1 # Unit mass (kg)
r = 0.01 # Radius (m)
I_para = 0.5*(m*r**2)
I_perp = (1/12)*m*(3*r**2 + rod_len**2)
I = np.array([[I_perp, 0, 0],
              [0, I_para, 0],
              [0, 0, I_perp]])
I_inv = LA.inv(I)
e1 = 0.5 # Coefficient of restitution
mu1 = 1.5 # Coefficient of friction
# contact_size = np.array([0.762, 0.762])
rod_contact = contact.ImpulseContact(rod_contact_points_R, m, I, e=e1, 
                                     debug_ctl=contact_debug_ctl, mu=mu1)

# ------------------- Overall System Dynamics - Rod in 2d -------------------- #

def sysdyn_rod(self, t: float, state_vec: np.ndarray):
    '''
    Mixin method that will override the default simulation system dynamics. 
    Will give the system dynamics for a rod falling and spinning. Full states
    are included, even though the intention is to simulate 2d mostly.

    States: x, y, z, qx, qy, qz, qw, dx, dy, dz, omega_x, omega_y, omega_z
    '''

    # Assign state vector to variables for readability
    x = state_vec[0]
    y = state_vec[1]
    z = state_vec[2]
    qx = state_vec[3]
    qy = state_vec[4]
    qz = state_vec[5]
    qw = state_vec[6]
    vel_x = state_vec[7]
    vel_y = state_vec[8]
    vel_z = state_vec[9]
    om_x = state_vec[10]
    om_y = state_vec[11]
    om_z = state_vec[12]

    omega_B = np.array([om_x, om_y, om_z])

     # Compile system dynamics output
    dstates_dt = np.zeros(13, dtype=float) # Init return deriv state vec

    # Unforced system dynamics
    dstates_dt[0] = vel_x # x velocity
    dstates_dt[1] = vel_y # y velocity
    dstates_dt[2] = vel_z # z velocity
    dstates_dt[3] = 0.5*(om_x*qw + om_z*qy - om_y*qz) # dqx_dt [6]
    dstates_dt[4] = 0.5*(om_y*qw - om_z*qx + om_x*qz) # dqy_dt
    dstates_dt[5] = 0.5*(om_z*qw + om_y*qx - om_x*qy) # dqz_dt
    dstates_dt[6] = 0.5*(-om_x*qx - om_y*qy - om_z*qz) # dqw_dt
    dstates_dt[9] = 9.8 # z acceleration from grav

    cross_term = np.cross(omega_B, I@omega_B) 
    alpha_B = I_inv@cross_term
    dstates_dt[10:] = alpha_B # Inertial effects on angular acceleration

    return dstates_dt

# --------------------- Direct State Modification Method --------------------- #

def state_mod(self, t: float, x: np.ndarray, i: float):
    '''
    Mixin method that will override the default simulation state modification.
    This method will primarily serve to implement the contact model
    '''

    # Isolate out needed states and variables for clarity
    pos_W = x[:3] # Rod position states
    R_R_W = Rot.from_quat(x[3:7]) # Rod to world rotation
    lin_vel_W = x[7:10] # Rod linear velocity states
    ang_vel_R = x[10:] # Body-fixed R-frame angular velocities

    # Apply contact model to system dynamics
    del_lin_vel_W, del_ang_vel_R, del_z_pos_W, cont_check_vec, \
        df_cont_tracking = rod_contact.contact_main(pos_W, R_R_W, lin_vel_W, 
            ang_vel_R, t=t)
    
    # Add changes of velocity due to contact to rod states
    x_out = x + np.concatenate((del_z_pos_W, np.zeros(4), del_lin_vel_W, 
                                del_ang_vel_R))
    
    return x_out

# --------------------- Configure and Perform Simulation --------------------- #

# Setup simulation control parameters
output_ctl1 = { # Output file control
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'rod_test.csv', # Name for output file
    'file': True, # Save output file
    'plots': True # Save output plots
}

# Simulation time parameters
tspan = (0, 1.5)
timestep = 0.0005

# State Vector: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, 
#   omega_x, omega_y, omega_z
state_names = ['x', 'y', 'z', 
               'qx', 'qy', 'qz', 'qw', 
               'dx', 'dy', 'dz', 
               'om_x', 'om_y', 'om_z']
pos0 = np.array([0.0, 0.0, -2.0])
R_init = Rot.from_euler('xyz', [15, 0, 0], degrees=True)
q0 = R_init.as_quat()
vel0 = np.array([0, 0, 0])
omega0 = np.array([0.0, 0, 0])
x0 = np.concatenate((pos0, q0, vel0, omega0))

# Initialize simulation
sim_rod = simulator.Simulator(tspan, x0, timestep=timestep, 
                               state_names=state_names, output_ctl=output_ctl1)

# Overwrite default state modification and system dynamics methods
sim_rod.state_mod_fun = MethodType(state_mod, sim_rod)
sim_rod.sysdyn = MethodType(sysdyn_rod, sim_rod)

# Compute the simulation
sim_rod.compute() 

rod_results_df = sim_rod.state_traj

# ------------------------------ Post Processing ----------------------------- #

# Add euler angles to the trajectory dataframe
eul_conv = lambda row: pd.Series(Rot.from_quat([row.qx, row.qy, 
                                row.qz, row.qw]).as_euler('xyz', degrees=True))
euler_df = rod_results_df.apply(eul_conv, axis=1)

euler_df.columns = ['phi', 'theta', 'psi'] # Rename the cols in the Euler df
rod_results_df = pd.concat([rod_results_df, euler_df], axis=1)

# Create dataframe with contact point positions and velocities
name_list_points = uf_cont.point_df_name(rod_contact_points_R.shape[0])
point_df = pd.DataFrame(columns=name_list_points, index=rod_results_df.index)
point_df.t = rod_results_df.t

# Loop through results df and generate contact data
for i in range(len(rod_results_df.t)): # For each rod position

    # Position and orientation
    rod_pos_W = rod_results_df.iloc[i, 1:4].to_numpy().astype(float)
    rod_q = rod_results_df.iloc[i, 4:8].to_numpy().astype(float)
    R_R_W = Rot.from_quat(rod_q)

    # Linear and angular velocity
    rod_vel_W = rod_results_df.iloc[i, 8:11].to_numpy().astype(float)
    rod_angvel_R = rod_results_df.iloc[i, 11:14].to_numpy().astype(float)

    for ii in range(rod_contact_points_R.shape[0]): # For each contact point

        # Compute contact point position
        point_ii_R = rod_contact_points_R[ii].astype(float)
        point_ii_pos_W = rod_pos_W + R_R_W.apply(point_ii_R)
        point_df.iloc[i, (6*ii + 1):(6*ii + 4)] =  point_ii_pos_W

        # Compute contact point velocity
        cross_term = np.cross(rod_angvel_R, point_ii_R)
        vel_ii_R = rod_vel_W + R_R_W.apply(cross_term)

        if ii < rod_contact_points_R.shape[0]:
            point_df.iloc[i, (6*ii + 4):(6*ii + 7)] = vel_ii_R
        else:
            point_df.iloc[i, (6*ii + 4):] = vel_ii_R

# --------------------------------- Plotting --------------------------------- #

# Overall plot settings
fig_scale = 1.0
fig_dpi = 96 # PowerPoint default
font_suptitle = 28
font_subtitle = 22
font_axlabel = 20
font_axtick = 18
ax_lw = 2
plt_lw = 3
setpoint_lw = 1
fig_x_size = 13.33
fig_y_size = 9.075
save_fig = False

# Plot position states and setpoints
fig_pos, (ax_x_pos, ax_y_pos, ax_z_pos) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_pos.suptitle('Rod Position States', fontsize=font_suptitle, 
                 fontweight='bold')

# x position
ax_x_pos.plot(rod_results_df.t, rod_results_df.x, linewidth=plt_lw) 
ax_x_pos.set_title('X Position', fontsize=font_subtitle, fontweight='bold')
ax_x_pos.set_ylabel('Position (m)', fontsize=font_axlabel, fontweight='bold')
ax_x_pos.grid()
# ax_x_pos.set_ylim(-2.5, 2.5)
ax_x_pos.legend(['position'], fontsize=18)

# y position
ax_y_pos.plot(rod_results_df.t, rod_results_df.y, linewidth=plt_lw) 
ax_y_pos.set_title('Y Position', fontsize=font_subtitle, fontweight='bold')
ax_y_pos.set_ylabel('Position (m)', fontsize=font_axlabel, fontweight='bold')
ax_y_pos.grid()
# ax_y_pos.set_ylim(-2.5, 2.5)
ax_y_pos.legend(['position'], fontsize=18)

# z position
ax_z_pos.plot(rod_results_df.t, rod_results_df.z, linewidth=plt_lw) 
ax_z_pos.set_title('Z Position', fontsize=font_subtitle, fontweight='bold')
ax_z_pos.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_z_pos.set_ylabel('Position (m)', fontsize=font_axlabel, fontweight='bold')
ax_z_pos.grid()
ax_z_pos.invert_yaxis()
ax_z_pos.legend(['position'], fontsize=18)
ax_z_pos.set_ylim(0.5, -5)


# Plot velocity states and setpoints
fig_vel, (ax_x_vel, ax_y_vel, ax_z_vel) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_vel.suptitle('Rod Velocity States', fontsize=font_suptitle, 
                 fontweight='bold')

# x velocity
ax_x_vel.plot(rod_results_df.t, rod_results_df.dx, linewidth=plt_lw) 
ax_x_vel.set_title('X Velocity', fontsize=font_subtitle, fontweight='bold')
ax_x_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, fontweight='bold')
ax_x_vel.grid()
# ax_x_vel.set_ylim(-1.0, 1.0)
ax_x_vel.legend(['velocity'], fontsize=18)

# y velocity
ax_y_vel.plot(rod_results_df.t, rod_results_df.dy, linewidth=plt_lw) 
ax_y_vel.set_title('Y Velocity', fontsize=font_subtitle, fontweight='bold')
ax_y_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, fontweight='bold')
ax_y_vel.grid()
# ax_y_vel.set_ylim(-1.0, 1.0)
ax_y_vel.legend(['velocity'], fontsize=18)

# z velocity
ax_z_vel.plot(rod_results_df.t, rod_results_df.dz, linewidth=plt_lw) 
ax_z_vel.set_title('Z Velocity', fontsize=font_subtitle, fontweight='bold')
ax_z_vel.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_z_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, fontweight='bold')
ax_z_vel.grid()
# ax_z_vel.set_ylim(-1.0, 1.0)
ax_z_vel.invert_yaxis()
ax_z_vel.legend(['velocity'], fontsize=18)


# Plot attitude quaternion and setpoints
fig_quat, (ax_qx, ax_qy, ax_qz, ax_qw) = \
    plt.subplots(4, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_quat.suptitle('Rod Quaternion States', fontsize=font_suptitle, 
                 fontweight='bold')

# qx
ax_qx.plot(rod_results_df.t, rod_results_df.qx, linewidth=plt_lw) 
ax_qx.set_title('X Quaternion', fontsize=font_subtitle, fontweight='bold')
ax_qx.grid()
# ax_qx.set_ylim(-1.1, 1.1)
ax_qx.legend(['$q_x$'], fontsize=18)

# qy
ax_qy.plot(rod_results_df.t, rod_results_df.qy, linewidth=plt_lw) 
ax_qy.set_title('Y Quaternion', fontsize=font_subtitle, fontweight='bold')
ax_qy.grid()
# ax_qy.set_ylim(-1.1, 1.1)
ax_qy.legend(['$q_y$'], fontsize=18)

# qz
ax_qz.plot(rod_results_df.t, rod_results_df.qz, linewidth=plt_lw) 
ax_qz.set_title('Z Quaternion', fontsize=font_subtitle, fontweight='bold')
ax_qz.grid()
# ax_qz.set_ylim(-1.1, 1.1)
ax_qz.legend(['$q_z$'], fontsize=18)

# qw
ax_qw.plot(rod_results_df.t, rod_results_df.qw, linewidth=plt_lw) 
ax_qw.set_title('W Quaternion', fontsize=font_subtitle, fontweight='bold')
ax_qw.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_qw.grid()
# ax_qw.set_ylim(-1.1, 1.1)
ax_qw.legend(['$q_w$'], fontsize=18)


# Plot attitude euler angles and setpoints
fig_eul, (ax_phi, ax_theta, ax_psi) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_eul.suptitle('Rod Euler Angles', fontsize=font_suptitle, fontweight='bold')
ax_eul_list = [ax_phi, ax_theta, ax_psi]

# Phi (roll) angle
ax_phi.plot(rod_results_df.t, rod_results_df.phi, linewidth=plt_lw) 
ax_phi.set_title('Roll Angle ($\Phi$)', fontsize=font_subtitle, 
                 fontweight='bold')
ax_phi.set_ylabel('Angle (deg)', fontsize=font_axlabel, fontweight='bold')
ax_phi.grid()
# ax_phi.set_ylim(-60, 60)
ax_phi.legend(['$\Phi$'], fontsize=18)

# Theta (pitch) angle
ax_theta.plot(rod_results_df.t, rod_results_df.theta, linewidth=plt_lw) 
ax_theta.set_title('Pitch Angle ($\Theta$)', fontsize=font_subtitle, 
                   fontweight='bold')
ax_theta.set_ylabel('Angle (deg)', fontsize=font_axlabel, fontweight='bold')
ax_theta.grid()
# ax_theta.set_ylim(-60, 60)
ax_theta.legend(['$\Theta$'], fontsize=18)

# Psi (yaw) angle
ax_psi.plot(rod_results_df.t, rod_results_df.psi, linewidth=plt_lw) 
ax_psi.set_title('Yaw Angle ($\Psi$)', fontsize=font_subtitle, 
                 fontweight='bold')
ax_psi.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_psi.set_ylabel('Angle (deg)', fontsize=font_axlabel, fontweight='bold')
ax_psi.grid()
# ax_psi.set_ylim(-60, 60)
ax_psi.legend(['$\Psi$'], fontsize=18)


# Plot angular rates and setpoints
fig_om, (ax_om_x, ax_om_y, ax_om_z) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_om.suptitle('Rod Angular Velocity States', fontsize=font_suptitle, 
                 fontweight='bold')
ax_om_list = [ax_om_x, ax_om_y, ax_om_z]

# x angular velocity
ax_om_x.plot(rod_results_df.t, (180/pi)*rod_results_df.om_x, linewidth=plt_lw) 
ax_om_x.set_title('$\omega_x$', fontsize=font_subtitle, fontweight='bold')
ax_om_x.set_ylabel('Ang. Vel. (deg/s)', fontsize=font_axlabel, 
                   fontweight='bold')
ax_om_x.grid()
# ax_om_x.set_ylim(-60, 60)
ax_om_x.legend(['$\omega_x$'], fontsize=18)

# y angular velocity
ax_om_y.plot(rod_results_df.t, (180/pi)*rod_results_df.om_y, linewidth=plt_lw) 
ax_om_y.set_title('$\omega_y$', fontsize=font_subtitle, fontweight='bold')
ax_om_y.set_ylabel('Ang. Vel. (deg/s)', fontsize=font_axlabel, 
                   fontweight='bold')
ax_om_y.grid()
# ax_om_y.set_ylim(-60, 60)
ax_om_y.legend(['$\omega_y$'], fontsize=18)

# z angular velocity
ax_om_z.plot(rod_results_df.t, (180/pi)*rod_results_df.om_z, 
             linewidth=plt_lw) 
ax_om_z.set_title('$\omega_z$', fontsize=font_subtitle, fontweight='bold')
ax_om_z.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_om_z.set_ylabel('Ang. Vel. (deg/s)', fontsize=font_axlabel, 
                   fontweight='bold')
ax_om_z.grid()
# ax_om_z.set_ylim(-60, 60)
ax_om_z.legend(['$\omega_z$'], fontsize=18)

# Plot the contact point positions 
point_cols = point_df.columns.values.tolist()
point_cols = ['${0}$'.format(col) for col in point_cols]
fig_cont_pos, (ax_x_cont_pos, ax_y_cont_pos, ax_z_cont_pos) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_cont_pos.suptitle('Contact Point Position States', fontsize=font_suptitle, 
                 fontweight='bold')

# x position
ax_x_cont_pos.plot(point_df.t, point_df.iloc[:, 1::6], linewidth=plt_lw) 
ax_x_cont_pos.set_title('X Position', fontsize=font_subtitle, fontweight='bold')
ax_x_cont_pos.set_ylabel('Position (m)', fontsize=font_axlabel, 
                         fontweight='bold')
ax_x_cont_pos.grid()
# ax_x_pos.set_ylim(-2.5, 2.5)
ax_x_cont_pos.legend(point_cols[1::6], fontsize=18)

# y position
ax_y_cont_pos.plot(point_df.t, point_df.iloc[:, 2::6], linewidth=plt_lw) 
ax_y_cont_pos.set_title('Y Position', fontsize=font_subtitle, fontweight='bold')
ax_y_cont_pos.set_ylabel('Position (m)', fontsize=font_axlabel, 
                         fontweight='bold')
ax_y_cont_pos.grid()
# ax_y_pos.set_ylim(-2.5, 2.5)
ax_y_cont_pos.legend(point_cols[2::6], fontsize=18)

# z position
ax_z_cont_pos.plot(point_df.t, point_df.iloc[:, 3::6], linewidth=plt_lw) 
ax_z_cont_pos.set_title('Z Position', fontsize=font_subtitle, fontweight='bold')
ax_z_cont_pos.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_z_cont_pos.set_ylabel('Position (m)', fontsize=font_axlabel, 
                         fontweight='bold')
ax_z_cont_pos.grid()
ax_z_cont_pos.invert_yaxis()
ax_z_cont_pos.legend(point_cols[3::6], fontsize=18)
# ax_z_cont_pos.set_ylim(0.5, -5)


# Plot the contact point velocities
fig_cont_vel, (ax_x_cont_vel, ax_y_cont_vel, ax_z_cont_vel) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_cont_vel.suptitle('Contact Point Velocity States', fontsize=font_suptitle, 
                 fontweight='bold')

# x velocity
ax_x_cont_vel.plot(point_df.t, point_df.iloc[:, 4::6], linewidth=plt_lw) 
ax_x_cont_vel.set_title('X Velocity', fontsize=font_subtitle, fontweight='bold')
ax_x_cont_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, 
                         fontweight='bold')
ax_x_cont_vel.grid()
# ax_x_vel.set_ylim(-2.5, 2.5)
ax_x_cont_vel.legend(point_cols[4::6], fontsize=18)

# y velocity
ax_y_cont_vel.plot(point_df.t, point_df.iloc[:, 5::6], linewidth=plt_lw) 
ax_y_cont_vel.set_title('Y Velocity', fontsize=font_subtitle, fontweight='bold')
ax_y_cont_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, 
                         fontweight='bold')
ax_y_cont_vel.grid()
# ax_y_vel.set_ylim(-2.5, 2.5)
ax_y_cont_vel.legend(point_cols[5::6], fontsize=18)

# z velocity
ax_z_cont_vel.plot(point_df.t, point_df.iloc[:, 6::6], linewidth=plt_lw) 
ax_z_cont_vel.set_title('Z Velocity', fontsize=font_subtitle, fontweight='bold')
ax_z_cont_vel.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_z_cont_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, 
                         fontweight='bold')
ax_z_cont_vel.grid()
ax_z_cont_vel.invert_yaxis()
ax_z_cont_vel.legend(point_cols[6::6], fontsize=18)
# ax_z_cont_vel.set_ylim(0.5, -5)

# Save and show figures
if save_fig:
    trial_name = 'rod'
    fig_pos.savefig(trial_name + '_pos.png')
    fig_vel.savefig(trial_name + '_vel.png')
    fig_quat.savefig(trial_name + '_quat.png')
    fig_eul.savefig(trial_name + '_eul.png')
    fig_om.savefig(trial_name + '_om.png')
    fig_cont_pos.savefig(trial_name + '_cont_pos.png')
    fig_cont_vel.savefig(trial_name + '_cont_vel.png')
else:
    plt.show()




