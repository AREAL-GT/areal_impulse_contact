
# --------------------------------- Imports ---------------------------------- #

import numpy as np
from numpy import linalg as LA

import pandas as pd

from math import ceil

import warnings

from scipy.spatial.transform import Rotation as Rot

# Add workspace directory to the path
import os
import sys
sim_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, sim_pkg_path)

# Workspace package imports
from amls_sim.simulator import rk4_step_fun

from amls.dynamics.quadcopter_params import iris_params, x500_exp_params
from amls.dynamics.ground_vehicle import ground_vehicle_1

from amls_impulse_contact.single_point import SinglePoint
from amls_impulse_contact.multi_point import MultiPoint
import amls_impulse_contact.utility_functions as uf

# ----------------------- Default Class Init Arguments ----------------------- #

default_debug_ctl = { # Default debug settings in dict
    "warn_mute": False, # Mute warning messages
    "main": False, # contact_main method
    "contact": False, # contact_calc methods
    "count_every": False, # contact_count method every call
    "count_pos": False, # contact_count method positive counts
    "phase": False, # compression/extension/restitution methods
    "deriv": False, # restitution_deriv/compression_deriv methods
    "ancillary": False, # Ancillary helper methods
    "file": False # Save debug prints to a text file
}

# Default contact zone states: 
#   x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, omega_x, omega_y, omega_z 
default_cont_states = np.zeros(13) 
default_cont_states[6] = 1 # qw to 1
default_cont_size = -np.ones(2) # Default size -1 means ground plane

# ----------------------------- Class Definition ----------------------------- #

class ImpulseContact(SinglePoint, MultiPoint):
    '''
    Class used to organize and structure checking and resolving contacts in
    simulation using impulse-based methods

    Notes:
    - x-forward, y-right, z-down coordinate system
    - Assuming a single body with multiple body-fixed contact points
    - Assumed body 1 is the contact zone, with functionally infinite mass and
        inertia
    
    Frames:
        B: Body-fixed frame to the moving object
        W: NED inertial frame
        G: Fixed to the origin of the collision zone
        C: Collision frame
    '''

    # Class-wide variables
    g = 9.81 # Gravitational acceleration

    # Constructor method
    def __init__(self, contact_points_B: np.ndarray, m: float, I_B: np.ndarray, 
                 epsilon_c: float = 0.001, epsilon_t: float = 1e-2, 
                 uz_step: float = 0.001, Wz_step: float = 0.001, 
                 mu: float = 1.4, e: float = 0.46, 
                 cont_size: np.ndarray = default_cont_size,
                 micro_override: bool = True, 
                 debug_ctl: dict = default_debug_ctl) -> None:
        '''
        Constructor method

        Required Inputs:
            contact_points_B: np array of body-fixed contact points, each row
                containing (x,y,z) for a single point
            m: mass of body
            I_B: Inertia matrix of body expressed in the frame B
        Optional Inputs:
            epsilon_c: double contact zone tolerance
            epsilon_t: double lateral velocity tolerance for slip/stick fric
            uz_step: double step size for uz numerical integration
            Wz_step: double step size for Wz numerical integration
            mu: double friction sliding coefficient
            e: double contact coefficient of restitution
            cont_size: np 1d vec of x, y dimensions of contact area
            debug_ctl: dict controling debug print behavior
        '''

        self.points_B = contact_points_B # Body-frame cont points
        self.num_points = self.points_B.shape[0] # Number of contact points

        self.m = m # Mass of the body
        self.I_B = I_B # Inertia matrix of the body in frame B
        self.I_inv_B = LA.inv(self.I_B)
        self.mu = mu # Friction sliding coefficient
        self.e = e # Coefficient of resitution (default value steel-aluminum)

        self.epsilon_c = epsilon_c # Contact zone tolerance
        self.epsilon_t = epsilon_t # Slip/stick friction tolerance
        self.uz_step = uz_step # Step size for uz numerical integration
        self.Wz_step = Wz_step # Step size for Wz numerical integration

        self.cont_size = cont_size # Size of the contact area x and y

        self.debug_ctl = debug_ctl # Debug control dictionary
        self.debug_statements = [] # Init empty list to track debug prints

        self.micro_override = micro_override # Override to turn off ucollisions

        # Instance variables unset at initialization
        self.pos_W: np.ndarray # Moving body position in W frame
        self.R_B_W: Rot # Rotation object from B to W frame
        self.R_B_C: Rot # Rotation object from B to C frame
        self.points_G: np.ndarray # Contact points relative to contact zone

        self.min_num_steps = 10


    # ------------------------ General Contact Methods ----------------------- #

    # Main contact management method
    def contact_main(self, pos_W: np.ndarray, R_B_W: Rot , 
                     lin_vels_W: np.ndarray, ang_vels_B: np.ndarray, 
                     cont_states: np.ndarray = default_cont_states,
                     t: float = 0):
        '''
        Primary method call for contact. This will organize the calls to check
        and resolve contact.

        Required inputs:
            pos: np 1d vec of x,y,z inertial position
            rot: scipy rotation object of body-fixed frame
            lin_vel: np 1d vec of x,y,z inertial-frame linear velocity
            ang_vel: np 1d vec of body-fixed angular velocity 
        Optional inputs:
            t: time given , only used for debug statements currently
            cont_states: np 1d vec of contact area states:
                x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, 
                omega_x, omega_y, omega_z 
        Outputs:

        Notes:
        - All units meters, seconds, radians
        '''

        # Enforce types on inputs
        lin_vels_W = lin_vels_W.astype(float)
        ang_vels_B = ang_vels_B.astype(float)
        cont_states = cont_states.astype(float)

        # Define needed rotations for collision check
        R_G_W = Rot.from_quat(cont_states[3:7]) # Contact zone to inertial
        R_W_G = R_G_W.inv() # Inertial to contact zone
        R_C_W = R_G_W # Collision frame to inertial
        R_W_C = R_C_W.inv() # Inertial to collision frame
        R_B_C = R_B_W*R_W_C # Moving body to collision frame
        R_C_B = R_B_C.inv() # Collision to moving body frame

        pos_cont_W = cont_states[:3] # Position of the contact zone in inertial
        vel_cont_W = cont_states[7:10] # Vel of the contact zone in inertial
        ang_vel_cont_G = cont_states[10:] # Ang vel of the contact zone in G

        # Translate collision points from B to G
        mat1 = R_B_W.apply(self.points_B) + pos_W - pos_cont_W
        mat1 = mat1.astype(float)
        points_G = R_W_G.apply(mat1)
        self.points_G = points_G # Save to instance variable for external use

        # Compute collision point velocities in inertial frame
        point_vels_W = uf.vel_calc(self.points_B, R_B_W, lin_vels_W, ang_vels_B)

        # Translate collision point velocities to contact zone G frame
        point_vels_G = R_W_G.apply(point_vels_W - vel_cont_W) 

        # Run the collision check algorithm on all potential points
        num_contacts, contact_ids, points_G, cont_check_vec = \
            self.collision_check(points_G, point_vels_G, t) 
        
        # Debug print for collision check
        if self.debug_ctl['main']:
            msg = "[Cont-contact_main] t = %f: contact_ids = " % t + \
                np.array2string(contact_ids)
            self.debug_statements.append(msg)
            print(msg)
        
        # Initialize output vectors to zeros for no change
        del_lin_vel_W = np.zeros(3) # Change in linear velocity
        del_ang_vel_B = np.zeros(3) # Change in angular velocity
        del_z_pos_W = 0 # Position adjustment from penetration
        del_lin_vel_C = np.zeros(3) # Change in linear velocity
        del_ang_vel_C = np.zeros(3) # Change in angular velocity
        del_z_pos_G = 0 # Position adjustment from penetration

        # Init blank tracking mat
        tracking_mat = pd.DataFrame(columns=['ux', 'uy', 'uz', 'Wz'])

        # Call contact solving methods depending on number of contact points
        if num_contacts == 1: # If a single contact point

            # Define translation in W between origins of C and W
            # pos_C_W = pos_cont_W + R_G_W.apply(points_G)

            # Rotate I2 to the C frame - R*I*R^T
            I2_C = R_B_C.as_matrix()@self.I_B@R_C_B.as_matrix()

            # Rotate r2 to the C frame
            r2_B = self.points_B[contact_ids, :].flatten()
            r2_C = R_B_C.apply(r2_B)

            # Transfer u1 and u2 to the C frame
            vel_cont_C = R_W_C.apply(vel_cont_W)
            u1_C = uf.vel_calc(points_G[contact_ids].flatten(), Rot.identity(), 
                               vel_cont_C, ang_vel_cont_G) 
            u2_C = point_vels_G[contact_ids].flatten() + vel_cont_C 

            # Relative contact point velocity u0 in C
            u0_C = u1_C - u2_C 

            # Call the single point collision calculation method
            del_lin_vel_C, del_ang_vel_C, tracking_mat = \
                self.contact_calc_sp(r2_C, I2_C, u0_C, t)

        elif num_contacts > 1: # If multiple contact points

            # Isolate active in contact points
            act_points_B = self.points_B[contact_ids]
            act_points_G = points_G[contact_ids]
            act_vels_G = point_vels_G[contact_ids]

            # Rotate I2 to the C frame - R*I*R^T
            I2_C = R_B_C.as_matrix()@self.I_B@R_C_B.as_matrix()

            # Rotate r vectors on body 2 to the C frame
            r2_C = R_B_C.apply(act_points_B)

            # Transfer u1 and u2 to the C frame
            vel_cont_C = R_W_C.apply(vel_cont_W)
            u1_C = uf.vel_calc(act_points_G, Rot.identity(), vel_cont_C, 
                               ang_vel_cont_G) 
            u2_C = act_vels_G + vel_cont_C 

            # Relative contact point velocities u0 in C
            u0_C = u1_C - u2_C 

            # Call multi-point contact method
            del_lin_vel_C, del_ang_vel_C, tracking_mat = \
                self.contact_calc_mp(r2_C, I2_C, u0_C, t)

        # Convert linear velocity change to inertial frame
        del_lin_vel_W = R_C_W.apply(del_lin_vel_C)

        # Convert angular velocity change to body-fixed frame
        del_ang_vel_B = R_C_B.apply(del_ang_vel_C)

        # Compute z micro-adjustment to stop penetration
        del_z_pos_W = np.zeros(3)
        pen_check = np.all([points_G[:, 2] > 0, cont_check_vec], axis=0)
        if np.any(pen_check): # Any penetration
            z_offset_G = np.max(points_G[contact_ids, 2])
            del_z_pos_G = -z_offset_G
            del_z_pos_W = R_G_W.apply(np.array([0, 0, del_z_pos_G]))
        
        return del_lin_vel_W, del_ang_vel_B, del_z_pos_W, cont_check_vec, \
            tracking_mat

    # Collision check method
    def collision_check(self, points_G: np.ndarray, point_vels_G: np.ndarray, 
                        t: float = 0):
        '''
        Method to check for the collision state of all potential points

        Required Inputs:
            points_G:
            point_vels_G:

        Notes:
        - Positions and velocities are in the contact-zone fixed frame G
        '''

        # Check if z-component of point positions are within the contact bound
        z_check = points_G[:, 2] > -self.epsilon_c

        # Check if z-component of point velocities are towards zons (positive)
        dz_check = point_vels_G[:, 2] > 0

        # Check if points are withing contact zone x & y bounds (if defined)
        if self.cont_size[0] != -1: # If there is a defined cont size

            # Contact zone sizing variables
            cs_x = self.cont_size[0] # Size in x-dimension
            cs_y = self.cont_size[1] # Size in y-dimension
            
            # Check zone dimensions
            x_check_neg = points_G[:, 0] < 0.5*cs_x
            x_check_pos = points_G[:, 0] > -0.5*cs_x
            y_check_neg = points_G[:, 1] < 0.5*cs_y
            y_check_pos = points_G[:, 1] > -0.5*cs_y

            zone_check = np.all([x_check_neg, x_check_pos, 
                                 y_check_neg, y_check_pos], axis=0)

        else: 

            zone_check = np.full(self.num_points, True, dtype=bool)

        # Combine the three checks
        all_check = np.all([z_check, zone_check, dz_check], axis=0) 

        # Find counts and indicies of points in contact
        cont_count = sum(bool(x) for x in all_check) # Sum active points

        # Find index/indecies of active points
        cont_idx = np.ma.masked_where(np.invert(all_check), 
                                      np.arange(self.num_points))
        cont_idx = np.ma.compressed(cont_idx)

        # Debug prints
        if self.debug_ctl['count_every']:

            # Section for prints on every call
            print('[contact_count_every]')
            print('t = %0.5f' % t)
            print('pos_G')
            print(points_G)
            print('vel_G')
            print(point_vels_G)
            print('checks')
            print(z_check)
            print(zone_check)
            print(dz_check)
            print(all_check)
            print('index')
            print(cont_count)
            print(cont_idx)

        if self.debug_ctl['count_pos']:

            if cont_count > 0: # Section for prints if in contact

                print('[contact_count_pos]')
                print('t = %0.5f' % t)
                print('pos_G')
                print(points_G)
                print('vel_G')
                print(point_vels_G)
                print('checks')
                print(z_check)
                print(zone_check)
                print(dz_check)
                print(all_check)
                print('index')
                print(cont_count)
                print(cont_idx)

        return cont_count, cont_idx, points_G, all_check

# ------------------------ Configured Collision Models ----------------------- #

debug_ctl_mute = { # Debug settings in dict
    "warn_mute": True, # Mute warning messages
    "main": False, # contact_main method
    "contact": False, # contact_calc methods
    "count_every": False, # contact_count method every call
    "count_pos": False, # contact_count method positive counts
    "phase": False, # compression/extension/restitution methods
    "deriv": False, # restitution_deriv/compression_deriv methods
    "ancillary": False, # Ancillary helper methods
    "file": False # Save debug prints to a text file
}

# Collision model based on Iris quadcopter
leg_dim = iris_params.d # 9.5 inches to m
iris_contact_points = np.array([[leg_dim/2, leg_dim/2, leg_dim/3],
                                [-leg_dim/2, leg_dim/2, leg_dim/3], 
                                [-leg_dim/2, -leg_dim/2, leg_dim/3],
                                [leg_dim/2, -leg_dim/2, leg_dim/3]]) 
m = iris_params.m
I = iris_params.I
e1 = 0.1 # Coefficient of restitution
mu1 = 1.0 # Coefficient of friction
# contact_size = np.array([0.762, 0.762]) # 30 inches to m
uav_contact_iris = ImpulseContact(iris_contact_points, m, I, e=e1, mu=mu1,
                                  cont_size=ground_vehicle_1.contact_size, 
                                  debug_ctl=debug_ctl_mute)

# Collision model based on Iris quadcopter with contact in line with z com
# leg_dim = iris_params.d # 9.5 inches to m
# iris_contact_points_noz = np.array([[leg_dim/2, leg_dim/2, 0],
#                                     [-leg_dim/2, leg_dim/2, 0], 
#                                     [-leg_dim/2, -leg_dim/2, 0],
#                                     [leg_dim/2, -leg_dim/2, 0]])
iris_contact_points_noz = iris_params.D
m = iris_params.m
I = iris_params.I
e1 = 0.1 # Coefficient of restitution
mu1 = 1.0 # Coefficient of friction
uav_contact_iris_noz = ImpulseContact(iris_contact_points_noz, m, I, e=e1, 
    mu=mu1, cont_size=ground_vehicle_1.contact_size, debug_ctl=debug_ctl_mute)

# Collision model based on modified Holybro x500v2
xdim = 0.2381
ydim = 0.2413
zdim = 0.2286
x500_cont_points = np.array([[xdim/2, ydim/2, zdim],
                             [-xdim/2, ydim/2, zdim], 
                             [-xdim/2, -ydim/2, zdim],
                             [xdim/2, -ydim/2, zdim]]) 
m = x500_exp_params.m
I = x500_exp_params.I
# e1 = 0.6 # Coefficient of restitution
# mu1 = 0.293 # Coefficient of friction
e2 = 0.55 # Coefficient of restitution - Look good as of 8/30/23
mu2 = 0.293 # Coefficient of friction
e_use = e2
mu_use = mu2
uav_contact_x500_exp = ImpulseContact(x500_cont_points, m, I, e=e_use, 
    mu=mu_use,cont_size=ground_vehicle_1.contact_size, debug_ctl=debug_ctl_mute)

