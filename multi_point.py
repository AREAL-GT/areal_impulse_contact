
# Imports
import numpy as np
from numpy import linalg as LA

import pandas as pd

from math import floor

import copy
import string

# Add workspace directory to the path
import os
import sys
sim_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, sim_pkg_path)

# Workspace package imports
from amls_sim.simulator import rk4_step_fun

import amls_impulse_contact.utility_functions as uf
from amls_impulse_contact.utility_functions import pad

class MultiPoint():
    '''
    Mixin class containing the multi-point contact methods.
    This class is methods only

    Methods:
        contact_calc_mp
        compression_calc_mp
        extension_calc_mp
        restitution_calc_mp
        compression_deriv_mp
        restitution_deriv_mp
        microcollision_compute_mp
        ext_limit_calc_mp
        micro_check_mp
        cont_mat_calc_mp
        track_mat_name_mp
    '''

    def contact_calc_mp(self, r2_C: np.ndarray, I_C: np.ndarray, 
                        u0_C: np.ndarray, t: float=0):
        '''
        Method to carry out contact computations with multiple simultaneous
        collision points

         Required Inputs:
            r2_C: np 1d vec of C-frame active contact-point x,y,z positions
            I_C:
            u0_C: 
            primary_idx:

        Optional Inputs:
            t: time given , only used for debug statements currently

        Notes:
        '''

        num_points = r2_C.shape[0] # Number of active contact points

        # Solve for the collision matrix K
        K = self.cont_mat_calc_mp(r2_C, I_C)
        
        # Initialize the collision tracking matrices to 0
        track_names = self.track_mat_name_mp(num_points)
        track_mat_comp = pd.DataFrame(columns=track_names)
        track_mat_ext = pd.DataFrame(columns=track_names)
        track_mat_rest = pd.DataFrame(columns=track_names)

        # Check if collision is in microcollision range
        #   microcollision conditions: 
        #       - all points in sticking tangential velocity
        #       - all points normal velocity in prescribed bound
        micro_check = self.micro_check_mp(u0_C)

        # Compute collision impulses from normal or microcollisions
        if micro_check: # If in microcollision

            # Compute microcollision impulses
            p_C = self.microcollision_compute_mp(u0_C, K)

            if self.debug_ctl['contact']:
                msg = ("[Cont-contact_calc_mp] t = %f: microcollision " \
                        "impulse (p) = ") % (t) + np.array2string(p_C)
                self.debug_statements.append(msg)
                print(msg)

        else: # If in normal collision

            # Call the collision computation methods
            if self.debug_ctl['contact']:
                msg = ("[Cont-contact_calc_mp] t = %f: Contact start") % (t) + \
                    "\n\tu0_C = " + np.array_str(u0_C[0]) + '\n' + \
                        pad(u0_C[1:], sep='\t\t')
                self.debug_statements.append(msg)
                print(msg)

            u1_C, Wz1, track_mat_comp = self.compression_calc_mp(u0_C, K, t)

            if self.debug_ctl['contact']:
                msg = ("[Cont-contact_calc_mp] t = %f: Pre extension") % (t) + \
                    "\n\tWz1 = " + np.array_str(Wz1) + \
                    '\n\tu0_C = ' + np.array_str(u0_C[0]) + '\n' + \
                        pad(u0_C[1:], sep='\t\t') + \
                    '\n\tu1_C = ' + np.array_str(u1_C[0]) + '\n' + \
                        pad(u1_C[1:], sep='\t\t')
                self.debug_statements.append(msg)
                print(msg)

            u2_C, Wz2, track_mat_ext = \
                self.extension_calc_mp(u0_C, u1_C, Wz1, K, t)
            
            if self.debug_ctl['contact']:
                msg = ("[Cont-contact_calc_mp] t = %f: Pre restitution") % \
                    (t) + \
                    '\n\tWz2 = ' + np.array_str(Wz2) + \
                    '\n\tu0_C = ' + np.array_str(u0_C[0]) + '\n' + \
                        pad(u0_C[1:], sep='\t\t') + \
                    '\n\tu1_C = ' + np.array_str(u1_C[0]) + '\n' + \
                        pad(u1_C[1:], sep='\t\t') + \
                    '\n\tu2_C = ' + np.array_str(u2_C[0]) + '\n' + \
                        pad(u2_C[1:], sep='\t\t')
                self.debug_statements.append(msg)
                print(msg)

            u3_C, Wz3, track_mat_rest = \
                self.restitution_calc_mp(u2_C, Wz1, Wz2, K, t)
            
            if self.debug_ctl['contact']:
                msg = ("[Cont-contact_calc_mp] t = %f: After restitution") % \
                    (t) + \
                    '\n\tWz3 = ' + np.array_str(Wz3) + \
                    '\n\tu0_C = ' + np.array_str(u0_C[0]) + '\n' + \
                        pad(u0_C[1:], sep='\t\t') + \
                    '\n\tu1_C = ' + np.array_str(u1_C[0]) + '\n' + \
                        pad(u1_C[1:], sep='\t\t') + \
                    '\n\tu2_C = ' + np.array_str(u2_C[0]) + '\n' + \
                        pad(u2_C[1:], sep='\t\t') + \
                    '\n\tu3_C = ' + np.array_str(u3_C[0]) + '\n' + \
                        pad(u3_C[1:], sep='\t\t')
                self.debug_statements.append(msg)
                print(msg)

            # Solve for the impulses from changes in velocity
            del_u = u3_C.flatten() - u0_C.flatten()
            del_u = del_u.astype(float)
            K_reorder = copy.deepcopy(K)
            K_reorder = K_reorder.transpose((0, 2, 1, 3))
            K_uni = K_reorder.reshape(3*num_points, 3*num_points)

            # Solve for dpaz_duaz
            p_C = LA.lstsq(K_uni, del_u, rcond=None)[0]    

        # Compute needed terms to apply impulses
        px_C = p_C[range(0, len(p_C), 3)].sum() # Sum x,y,z
        py_C = p_C[range(1, len(p_C), 3)].sum()
        pz_C = p_C[range(2, len(p_C), 3)].sum()
        p_sum_C = np.array([px_C, py_C, pz_C])

        cross_term = np.zeros(3) # Compute the cross term
        for i in range(num_points):
            j = i*3
            r_i = r2_C[i]
            if i < num_points:
                p_j = p_C[j:(j+3)]
            else:
                p_j = p_C[j:]
                
            cross_term += np.cross(r_i, p_j)

        # Apply impulses to the rigid body for lin and ang vel
        del_lin_vel_C = -(1/self.m)*p_sum_C 
        del_ang_vel_C = -LA.inv(I_C)@cross_term

        if self.debug_ctl['contact']:
            msg = ("[Cont-contact_calc_mp] t = %f: Contact end") % (t) + \
                '\n\tdel_lin_vel = ' + np.array_str(del_lin_vel_C) + '\n' + \
                '\n\tdel_ang_vel = ' + np.array_str(del_ang_vel_C)
            print(msg)

        # Compile tracking matrices for export
        track_mat = pd.concat([track_mat_comp, track_mat_ext, track_mat_rest])

        return del_lin_vel_C, del_ang_vel_C, track_mat

    def compression_calc_mp(self, u0_C: np.ndarray, K: np.ndarray, t: float):
        '''
        Method to carry out the compression phase of multi-point contact
        integration
        '''

        if self.debug_ctl['phase']:
            msg = ("[Cont-compression_calc_mp] t = %f: Compression " + \
                   "Start") % (t) + "\n\tu0_C = " + np.array_str(u0_C[0]) + \
                    '\n' + pad(u0_C[1:], sep='\t\t')
            self.debug_statements.append(msg)
            print(msg)

        # Initial sanity checks:
        #   u0_C < 0
        check_init1 = all(u0_C[:, 2] < 0)
        if not check_init1: # Raise exception if initial checks not passed
            msg = ("[compression_calc_mp] t = %f: Initial check failed, " + \
                   "uz0_C greater than 0") % t + "uZ0_C = " + \
                   np.array_str(u0_C[:, 2])
            raise ValueError(msg)

        num_points = u0_C.shape[0] # Number of active contact points

        # Setup relative step size parameters
        primary_idx = np.argmin(u0_C[:, 2]) # Used to set primary point
        num_steps_idx = np.argmax(u0_C[:, 2]) # Used to set number of steps
        # num_steps_idx = np.argmin(u0_C[:, 2]) # Used to set number of steps
        num_step_uz = abs(u0_C[num_steps_idx, 2])

        # Setup integration parameters based on compression limit
        step_check = num_step_uz > self.min_num_steps*self.uz_step
        if step_check: # If limit larger than step size
            num_steps = floor(step_check/self.uz_step) 
        else: # If limit less than step size
            num_steps = self.min_num_steps # Single step
        
        # Generate the uz integration vectors, also getting step size
        uz_mat, stepsize_vec = np.linspace(u0_C[:, 2], np.zeros(num_points), 
                                            num_steps, retstep=True)
        uz_mat = uz_mat.T

        # Setup step size vec variables
        uz_step_size = stepsize_vec[primary_idx] # Primary stepsize
        h_vec = stepsize_vec/uz_step_size # h values for compression phase
        self.h_vec_comp = h_vec # Save to class for use by other phases

        # Initialize tracking variables and matrix
        track_names = self.track_mat_name_mp(num_points)
        tracking_mat = pd.DataFrame(columns=track_names, index=range(num_steps))
        u_int = copy.deepcopy(u0_C) # Relative velocity integration tracker
        Wz_int = np.zeros(num_points) # Normal work 

        # Fill out tracking matrix with known values
        for i in range(num_points):
            tracking_mat.iloc[0, 4*i] = u_int[i, 0] # ux values
            tracking_mat.iloc[0, 4*i + 1] = u_int[i, 1] # uy values
            tracking_mat[tracking_mat.columns[4*i + 1]] = uz_mat[i] # uz vecs
            # tracking_mat.iloc[:, 4*i + 2] = uz_mat[i] # uz vectors
            # df[df.columns[i]] = newvals
            tracking_mat.iloc[0, 4*i + 3] = 0  # Wz values
            
        # First percent of steps check
        pct_exemp = 0.1 # Percent of steps not allowed to stick
        min_step_check = 1/pct_exemp # Number of steps needed for pct

        # Loop through all uz values 
        for i in range(1, num_steps):

            # Exclusion on sticking in initial compression phase check
            if num_steps > 1: # Multiple steps
                # If enough steps total to allow percentage check
                if num_steps > min_step_check: 
                    pct_check = i/num_steps > pct_exemp # Check percentage
                else: # Skip first step
                    pct_check = i > 0
            else: # Single step
                pct_check = True # Allow sticking

            # Tangential velocity check
            uT_mag = LA.norm(u_int[:, :2].astype(float), axis=1)
            uT_check = all(uT_mag < self.epsilon_t)

            check_stick1 = False

            # If past init pct of steps and all points in tan bounds
            if uT_check and pct_check:

                K_stick = copy.deepcopy(K)

                # Form unified "A" matrix of K matrices
                K_stick = K_stick.transpose((0, 2, 1, 3))
                K_uni = K_stick.reshape(3*num_points, 3*num_points)

                # Form unified dun_uaz vector
                du_dua_uni = np.zeros(3*num_points)
                for ii in range(num_points): # Loop through all points
                    du_dua_uni[3*ii + 2] = h_vec[ii]

                # Solve for dpaz_duaz
                dp_duz_sol = LA.lstsq(K_uni, du_dua_uni, rcond=None)[0]    
                dpaz_duaz = dp_duz_sol[3*primary_idx + 2]

                # Sticking sanity check:
                #   dpaz_duaz > 0
                check_stick1 = dpaz_duaz > 0

            if check_stick1: # If in tan, pct, and valid deriv parameter
                
                # Form lumped constant vec
                c_vec = np.square(h_vec)*dpaz_duaz

                # Compute final Wz values
                uz_vec = u_int[:, 2]
                Wz_old = Wz_int
                Wz_int = Wz_int + 0.5*np.multiply(c_vec, -np.square(uz_vec))
                
                # Assign 0 to u integration variables
                u_int = np.zeros((num_points, 3))

                # Fill out tracking matrix
                for ii in range(num_points):
                    tracking_mat.iloc[i:, 4*ii] = 0 # ux values
                    tracking_mat.iloc[i:, 4*ii + 1] = 0 # uy values
                    tracking_mat.iloc[i:, 4*ii + 3] =  Wz_old[ii] # Wz old
                    tracking_mat.iloc[-1, 4*ii + 3] =  Wz_int[ii]
                
                if self.debug_ctl['phase']:
                    msg = ("[Cont-compression_calc_mp] t = %f: " + \
                           "stick") % (t) + "\n\tu_int = " + \
                            np.array_str(u_int[0]) + '\n' + \
                            pad(u_int[1:], sep='\t\t')
                    self.debug_statements.append(msg)
                    print(msg)

                break # Break the for loop

            else: # In slipping

                # Form state vector for numerical integration
                state_vec = np.zeros(3*num_points)
                for ii in range(num_points):
                    state_vec[3*ii:(3*ii + 2)] = u_int[ii, :2]
                    state_vec[3*ii + 2] = Wz_int[ii]

                # Step states forward with numerical integration
                uaz_curr = u_int[primary_idx, 2]
                state_step = rk4_step_fun(self.compression_deriv_mp, state_vec, 
                    uaz_curr, uz_step_size, K, uaz_curr, h_vec, primary_idx, t)
                
                for ii in range(num_points): # Loop through each point
                    # Save states to integration variables
                    u_int[ii, :2] = state_step[3*ii:(3*ii + 2)] # ux, uy
                    u_int[ii, 2] = uz_mat[ii, i] # uz
                    Wz_int[ii] = state_step[3*ii + 2] # Wz

                # Save new states to tracking matrix
                tracking_mat.iloc[i, 4*ii:(4*ii + 3)] = u_int[ii] # u
                tracking_mat.iloc[i, 4*ii + 3] = Wz_int[ii] # Wz

                if self.debug_ctl['phase']:
                    msg = ("[Cont-compression_calc_mp] t = %f: " + \
                           "slip") % (t) + "\n\tu_int = " + \
                            np.array_str(u_int[0]) + '\n' + \
                            pad(u_int[1:], sep='\t\t')
                    self.debug_statements.append(msg)
                    print(msg)

        if self.debug_ctl['phase']:
            msg = ("[Cont-compression_calc_mp] t = %f: " + \
                   "Compression End") % (t) + "\n\tWz1 = " + \
                    np.array_str(Wz_int) + "\n\tu0_C = " + \
                    np.array_str(u0_C[0]) + "\n" + pad(u0_C[1:], sep="\t\t") + \
                    "\n\tu1_C = " + np.array_str(u_int[0]) + "\n" + \
                    pad(u_int[1:], sep="\t\t")
            self.debug_statements.append(msg)
            print(msg)

        # Set output variables
        u1_C = u_int
        Wz1 = Wz_int

        # Output sanity checks:
        #   uz1_C = 0
        #   Wz1 < 0
        check_final1 = all(u1_C[:, 2] == 0)
        check_final2 = all(Wz1 < 0)
        if not check_final1:
            msg = ("[compression_calc_mp] t = %f: Final check failed, " + \
                   "uz1_C not equal to 0, uz1_C = ") % (t) + \
                   np.array_str(u1_C[:, 2])
            raise ValueError(msg)
        if not check_final2:
            msg = ("[compression_calc_mp] t = %f: Final check failed, " + \
                   "Wz1 greater than or equal to 0, Wz1 = ") % (t) + \
                   np.array_str(Wz1)
            raise ValueError(msg)

        return u1_C, Wz1, tracking_mat

    def extension_calc_mp(self, u0_C: np.ndarray, u1_C: np.ndarray, 
                          Wz1: np.ndarray, K: np.ndarray, t: float):
        '''
        Method to carry out the extension phase of multi-point contact
        integration
        '''

        num_points = u1_C.shape[0] # Number of active contact points

        # Compute limits of restitution integration
        Wz3 = Wz1*(1 - self.e**2)
        del_Wz = Wz3 - Wz1 

        if self.debug_ctl['phase']:
            msg = ("[Cont-extension_calc_mp] t = %f Extension Start") % (t) + \
                '\n\tu1_C = ' + np.array_str(u1_C[0]) + '\n' + \
                    pad(u1_C[1:], sep='\t\t') + "\n\tWz1 = " + \
                    np.array_str(Wz1) + "\n\tWz3 = " + np.array_str(Wz3)
            self.debug_statements.append(msg)
            print(msg)

        # Initial sanity checks (after Wz3 is calculated):
        #   uz1_C = 0
        #   0 > Wz3 > Wz1
        #   K33 > 0
        check_init1 = all(u1_C[:, 2] == 0)
        check_init2 = all(K[0, :, 2, 2] > 0)
        check_init3 = all(0 >= Wz3)
        check_init4 = all(Wz3 > Wz1)
        if not check_init1:
            msg = ("[extension_calc_mp] t = %f: Initial check failed, " + \
                   "uz1_C not equal to 0, uz1_C = ") % (t) + \
                   np.array_str(u1_C[:, 2])
            raise ValueError(msg)
        if (not check_init2) and (not self.debug_ctl['warn_mute']):
            msg = ("[extension_calc_mp] t = %f: Initial check warning, " + \
                   "K33 less than or equal to 0, K33 = ") % (t) + \
                   np.array_str(K[0, :, 2, 2])
            print(msg)
        if not check_init3:
            msg = ("[extension_calc_mp] t = %f: Initial check failed, " + \
                   "Wz3 greater than 0, Wz3 = ") % (t) + np.array_str(Wz3)
            raise ValueError(msg)
        if not check_init4:
            msg = ("[extension_calc_mp] t = %f: Initial check failed, " + \
                   "Wz3 less than or equal to Wz1, Wz3 = ") % (t) + \
                   np.array_str(Wz3) + ", Wz1 = " + np.array_str(Wz1)
            raise ValueError(msg)
        
        # Initialize control parameters for extension integration
        ext_complete = False # While loop control variable
        min_range = 0.5 # Initialize to 50% of range minimum
        max_range = 0.8 # Initialize to 80% of range maximum
        target_range = (min_range + max_range)/2 # Middle of range (65%)
        loop_level1 = 30 # First increment of loop count for ranges
        loop_level2 = 50 # Second increment of loop count for ranges
        uz2_C = -0.5*self.e*u0_C[:, 2] # Naive initial guess
        min_ext_steps = 30
        ext_uz_step = 0.1*self.uz_step
        loop_count = 0
        track_names = self.track_mat_name_mp(num_points)

        # Iterative loop for extension integration
        while not ext_complete:

            loop_count += 1 # Increment count of loops

            # Setup relative step size parameters
            primary_idx = np.argmax(uz2_C) # Max value primary point
            num_steps_idx = np.argmax(uz2_C) # Num steps from max value
            # num_steps_idx = np.argmin(uz2_C) # Num steps from min value
            step_uz2 = abs(uz2_C[num_steps_idx])

            # Setup integration parameters based on extension limits
            step_check = step_uz2 > min_ext_steps*ext_uz_step
            if step_check: # If limit larger than step size x num steps
                num_steps = floor(step_uz2/ext_uz_step)
            else: # If limit less than step size
                num_steps = min_ext_steps

            # Generate the uz integration vectors, also getting step size
            uz_mat, stepsize_vec = \
                np.linspace(u1_C[:, 2], uz2_C, num_steps, retstep=True)
            uz_mat = uz_mat.T

            # Setup step size vec variables
            uz_step_size = stepsize_vec[primary_idx] # Primary stepsize
            h_vec = stepsize_vec/uz_step_size # h values for ext phase

            # Initialize tracking variables and matrix
            tracking_mat = pd.DataFrame(columns=track_names, 
                                        index=range(num_steps))
            u_int = copy.deepcopy(u1_C) # Relative velocity
            Wz_int = copy.deepcopy(Wz1) # Normal work 

            # Fill out tracking matrix with known values
            for i in range(num_points):
                tracking_mat.iloc[0, 4*i] = u_int[i, 0] # ux values
                tracking_mat.iloc[0, 4*i + 1] = u_int[i, 1] # uy values
                tracking_mat.iloc[:, 4*i + 2] = uz_mat[i] # uz vectors
                tracking_mat.iloc[0, 4*i + 3] = Wz_int[i]  # Wz values

            # Loop through all uz values 
            for i in range(1, num_steps):

                # Form state vector for numerical integration
                state_vec = np.zeros(3*num_points)
                for ii in range(num_points):
                    state_vec[3*ii:(3*ii + 2)] = u_int[ii, :2]
                    state_vec[3*ii + 2] = Wz_int[ii]

                # Step states forward with numerical integration
                uaz_curr = u_int[primary_idx, 2]
                state_step = rk4_step_fun(self.compression_deriv_mp, state_vec, 
                    uaz_curr, uz_step_size, K, uaz_curr, h_vec, primary_idx, t)
                
                for ii in range(num_points): # Loop through each point
                    # Save states to integration variables
                    u_int[ii, :2] = state_step[3*ii:(3*ii + 2)] # ux, uy
                    u_int[ii, 2] = uz_mat[ii, i] # uz
                    Wz_int[ii] = state_step[3*ii + 2] # Wz

                # Save new states to tracking matrix
                tracking_mat.iloc[i, 4*ii:(4*ii + 3)] = u_int[ii] # u
                tracking_mat.iloc[i, 4*ii + 3] = Wz_int[ii] # Wz

            # Set ranges based on loop count
            if loop_count <= loop_level1:
                min_range = 0.5 # Initialize to 50% of range minimum
                max_range = 0.8 # Initialize to 80% of range maximum
            elif loop_count > loop_level1 and loop_count <= loop_level2:
                min_range = 0.2 # 20% of range minimum
                max_range = 0.9 # 90% of range maximum
            else: # If loop count above level 2
                min_range = 0.01 # 1% of range minimum
                max_range = 0.99 # 99% of range maximum

            # Check and modify each point for extension range
            ext_check_vec = np.full(num_points, False, dtype=bool)
            for i in range(num_points):

                # Set value limits from ranges for this point
                Wz_max = Wz1[i] + max_range*del_Wz[i]
                Wz_min = Wz1[i] + min_range*del_Wz[i]

                # Check bounds on final normal work values
                if Wz_int[i] < Wz_max: # If below upper bound
                
                    if Wz_int[i] > Wz_min: # If above lower bound

                        ext_check_vec[i] = True # Set check to true

                    else: # If below lower bound as well

                        scale = (Wz_int[i] - Wz1[i])/(target_range*del_Wz[i])
                        scale = 1.2
                        uz2_C[i] = uz2_C[i]*scale

                else: # If above upper bound

                    scale = (target_range*del_Wz[i])/(Wz_int[i] - Wz1[i])
                    scale = 0.8
                    uz2_C[i] = scale*uz2_C[i]

            if all(ext_check_vec): # If all points in range
                ext_complete = True # End the loop

        # Set output variables
        u2_C = u_int
        Wz2 = Wz_int

        if self.debug_ctl['phase']:
            msg = ("[Cont-extension_calc_mp] t = %f: Extension " + \
                   "End") % (t) +  "\n\tWz1 = " + np.array_str(Wz1) + \
                   "\n\tWz2 = " + np.array_str(Wz2) + "\n\tWz3 = " + \
                    np.array_str(Wz3) + '\n\tu1_C = ' + \
                    np.array_str(u1_C[0]) + '\n' + \
                    pad(u1_C[1:], sep='\t\t') + '\n\tu2_C = ' + \
                    np.array_str(u_int[0]) + '\n' + pad(u_int[1:], sep='\t\t')
            self.debug_statements.append(msg)
            print(msg)

        # Output sanity checks
        #   uz2_C > 0
        #   Wz2 < Wz3
        #   Wz2 > Wz1
        check_final1 = all(u2_C[:, 2] > 0)
        check_final2 = all(Wz2 <= Wz3)
        check_final3 = all(Wz2 >= Wz1)
        if not check_final1:
            msg = ("[extension_calc_mp] t = %f: Final check failed, " + \
                   "uz2_C less than or equal to 0, uz2_C = ") % (t)+ \
                   np.array_str(u2_C[:, 2])
            raise ValueError(msg)
        if not check_final2:
            msg = ("[extension_calc_mp] t = %f: Final check failed, " + \
                   "Wz3 less than Wz2, Wz3 = ") % (t) + \
                   np.array_str(Wz3) + ", Wz2 = " + np.array_str(Wz2)
            print(Wz1)
            raise ValueError(msg)
        if not check_final3:
            msg = ("[extension_calc_mp] t = %f: Final check failed, " + \
                   "Wz2 less than Wz1, Wz2 = ") % (t) + \
                   np.array_str(Wz2) + ", Wz1 = " + np.array_str(Wz1)
            raise ValueError(msg)
        
        return u2_C, Wz2, tracking_mat

    def restitution_calc_mp(self, u2_C: np.ndarray, Wz1: np.ndarray, 
                            Wz2: np.ndarray, K: np.ndarray, t: float):
        '''
        Method to carry out the restitution phase of multi-point contact
        integration
        '''

        if self.debug_ctl['phase']:
            msg = ("[Cont-restitution_calc_mp] t = %f " + \
                   "Restitution Start") % (t) + '\n\tu2_C = ' + \
                    np.array_str(u2_C[0]) + '\n' + pad(u2_C[1:], sep='\t\t')
            self.debug_statements.append(msg)
            print(msg)

        num_points = u2_C.shape[0] # Number of active contact points

        # Compute limits of restitution integration
        Wz3 = Wz1*(1 - self.e**2)
        del_Wz = Wz3 - Wz2

        # Initial sanity checks (after Wz3 is calculated):
        #   uz2_C > 0
        #   0 > Wz3 > Wz2 > Wz1
        check_init1 = all(u2_C[:, 2] > 0)
        check_init2 = all(0 >= Wz3)
        check_init3 = all(Wz3 >= Wz2)
        check_init4 = all(Wz2 >= Wz1)
        if not check_init1:
            msg = ("[restitution_calc_mp] t = %f: Initial check failed, " + \
                   "uz2_C less than or equal to 0, uz2_C = ") % (t) + \
                   np.array_str(u2_C[:, 2])
            raise ValueError(msg)
        if not check_init2:
            msg = ("[restitution_calc_mp] t = %f: Initial check failed, " + \
                   "Wz3 greater than 0, Wz3 = ") % (t) + np.array_str(Wz3)
            raise ValueError(msg)
        if not check_init3:
            msg = ("[restitution_calc_mp] t = %f: Initial check failed, " + \
                   "Wz3 less than Wz2, Wz3 = ") % (t) + \
                   np.array_str(Wz3) + ", Wz2 = " + np.array_str(Wz2)
            raise ValueError(msg)
        if not check_init4:
            msg = ("[restitution_calc_mp] t = %f: Initial check failed, " + \
                   "Wz2 less than Wz1, Wz2 = ") % (t) + \
                   np.array_str(Wz2) + ", Wz1 = " + np.array_str(Wz1)
            raise ValueError(msg)

        # Determine the primary point - minimum Wz extension limit
        primary_idx = np.argmax(del_Wz) # Used to set primary point (h=1)
        num_steps_idx = np.argmin(del_Wz)  # Used to set number of steps
        # num_steps_idx = np.argmax(del_Wz)  # Used to set number of steps
        num_step_Wz = del_Wz[num_steps_idx]

        # Setup integration parameters based on restitution limits
        step_check = num_step_Wz > self.min_num_steps*self.Wz_step
        if step_check: # If limit larger than step size
            num_steps = floor(num_step_Wz/self.Wz_step)
        else: # If limit less than step size
            num_steps = self.min_num_steps

        # Generate the Wz integration vectors, also getting step size
        Wz_mat, stepsize_vec = np.linspace(Wz2, Wz3, num_steps, retstep=True)
        Wz_mat = Wz_mat.T

        Wz_step_size = stepsize_vec[primary_idx]
        h_vec = stepsize_vec/Wz_step_size # h values for resitution phase

        # Initialize tracking variables and matrix
        track_names = self.track_mat_name_mp(num_points)
        tracking_mat = pd.DataFrame(columns=track_names, index=range(num_steps))
        u_int = copy.deepcopy(u2_C) # Relative velocity integration tracker
        Wz_int = copy.deepcopy(Wz2) # Normal work vector

        # Fill out tracking matrix with known values
        for i in range(num_points):
            tracking_mat.iloc[0, 4*i] = u_int[i, 0] # ux values
            tracking_mat.iloc[0, 4*i + 1] = u_int[i, 1] # uy values
            tracking_mat.iloc[0, 4*i + 2] = u_int[i, 2] # uz values
            tracking_mat.iloc[:, 4*i + 3] = Wz_mat[i]  # Wz vectors

        # Loop through all Wz values 
        for i in range(1, num_steps):

            # Tangential velocity check
            uT_mag = LA.norm(u_int[:, :2].astype(float), axis=1)
            uT_check = all(uT_mag < self.epsilon_t)

            check_stick1 = False

            if uT_check: # Stick if all points in tan bounds

                K_stick = copy.deepcopy(K)

                # Form unified "A" matrix of K matrices
                K_stick = K_stick.transpose((0, 2, 1, 3))
                K_uni = K_stick.reshape(3*num_points, 3*num_points)

                # Form unified dun_uaz vector
                du_dua_uni = np.zeros(3*num_points)
                for ii in range(num_points): # Loop through all points
                    du_dua_uni[3*ii + 2] = h_vec[ii]

                # Solve for dpaz_duaz
                dp_duz_sol = LA.lstsq(K_uni, du_dua_uni, rcond=None)[0]    
                dpaz_duaz = dp_duz_sol[3*primary_idx + 2]

                # Sticking sanity check:
                #   dpaz_duaz > 0
                check_stick1 = dpaz_duaz > 0

            if check_stick1: # If in tan and valid deriv parameter

                # Form lumped constant vec
                c_vec = h_vec/dpaz_duaz

                # Compute final uz values
                uz_vec_old = u_int[:, 2]
                sub_vec = 2*np.multiply(c_vec, Wz3 - Wz_int)
                sub_vec += np.square(uz_vec_old)
                uz_vec = np.sqrt(sub_vec.astype(np.float64))

                # Assign 0 to u integration variables for ux and uy
                u_int[:, 0] = np.zeros(num_points)
                u_int[:, 1] = np.zeros(num_points)
                u_int[:, 2] = uz_vec

                # Fill out tracking matrix
                for ii in range(num_points):
                    tracking_mat.iloc[i:, 4*ii] = 0 # ux values
                    tracking_mat.iloc[i:, 4*ii + 1] = 0 # uy values
                    tracking_mat.iloc[i:, 4*ii + 2] =  uz_vec_old[ii]
                    tracking_mat.iloc[-1, 4*ii + 2] =  uz_vec[ii]

                if self.debug_ctl['phase']:
                    msg = ("[Cont-restitution_calc_mp] t = %f: " + \
                           "stick") % (t) + "\n\tu_int = " + \
                            np.array_str(u_int[0]) + '\n' + \
                            pad(u_int[1:], sep='\t\t')
                    self.debug_statements.append(msg)
                    print(msg)
                
                break # End integration loop

            else: # In sliding

                # Form state vector for numerical integration
                state_vec = u_int.flatten()

                # Step states forward with numerical integration
                Waz_curr = Wz_int[primary_idx]
                state_step = rk4_step_fun(self.restitution_deriv_mp, state_vec, 
                    Waz_curr, Wz_step_size, K, h_vec, primary_idx, t)
                
                for ii in range(num_points): # Loop through each point
                    # Save states to integration variables
                    if ii < num_points:
                        u_int[ii] = state_step[3*ii:(3*ii + 3)] # ux, uy, uz
                    else:
                        u_int[ii] = state_step[3*ii:]

                Wz_int[ii] = Wz_mat[ii, i] # Wz

                # Save new states to tracking matrix
                tracking_mat.iloc[i, 4*ii:(4*ii + 3)] = u_int[ii] # u
                tracking_mat.iloc[i, 4*ii + 3] = Wz_int[ii] # Wz

                if self.debug_ctl['phase']:
                    msg = ("[Cont-restitution_calc_mp] t = %f: " + \
                           "slip") % (t) + "\n\tu_int = " + \
                            np.array_str(u_int[0]) + '\n' + \
                            pad(u_int[1:], sep='\t\t')
                    self.debug_statements.append(msg)
                    print(msg)

        if self.debug_ctl['phase']:
            msg = ("[Cont-restitution_calc_mp] t = %f: " + \
                   "Restitution End") % (t) + "\n\tWz3 = " + \
                    np.array_str(Wz_int) + "\n\tu2_C = " + \
                    np.array_str(u2_C[0]) + "\n" + pad(u2_C[1:], sep="\t\t") + \
                    "\n\tu3_C = " + np.array_str(u_int[0]) + "\n" + \
                    pad(u_int[1:], sep="\t\t")
            self.debug_statements.append(msg)
            print(msg)

        # Set output variables
        u3_C = u_int

        # Output sanity checks
        #   uz3_C > uz2_C
        check_final1 = all(u3_C[:, 2] >= u2_C[:, 2])
        if not check_final1:
            msg = ("[restitution_calc_mp] t = %f: Final check failed, " + \
                   "uz3_C less than uz2_C, uz2_C = ") % (t) + \
                    np.array_str(u2_C[:, 2]) + ", uz3_C = " + \
                    np.array_str(u3_C[:, 2])
            raise ValueError(msg)

        return u3_C, Wz3, tracking_mat

    def compression_deriv_mp(self, uz_in: np.ndarray, x: np.ndarray, 
                             args: list):
        '''
        Method to carry out the compression-phase derivative for multi-point
        contact

        State vector: x = [uax, uay, Waz, ubx, uby, Wbz, ..., uNx, uNy, WNz]
        Returns dx: dx_duaz

        Notes:
            - Can't compute h from uz_in b/c that is single "timestep" input
              from the rk4 solver
        '''

        # Additional arguments
        K_in = args[0]
        K = copy.deepcopy(K_in)
        uaz = args[1]
        h_vec = args[2]
        primary_idx = args[3]
        t = args[4] # For warning print
        
        num_points = int(x.size/3)

        # Form unified vector of sliding vectors
        xi_uni = np.zeros(3*num_points)

        for i in range(num_points): # Loop through each contact point

            ux_i = x[3*i]
            uy_i = x[3*i + 1]
            uT_mag = LA.norm(np.array([ux_i, uy_i])) # Tangential magnitude

            if uT_mag == 0: # If numerically zero
                uT_mag = 0.01*self.epsilon_t # Make a very small number

            xi_i = np.array([-self.mu*ux_i/uT_mag, -self.mu*uy_i/uT_mag, 1])

            if i < num_points: # Assign to unified vec
                xi_uni[(3*i):(3*i + 3)] = xi_i*h_vec[i] 
            else:
                xi_uni[(3*i):] = xi_i*h_vec[i]

        # Form unified "A" matrix of K matrices
        K = K.transpose((0, 2, 1, 3))
        K_uni = K.reshape(3*num_points, 3*num_points)

        # Calculate the velocity derivatives
        du_dpaz = K_uni@xi_uni

        if du_dpaz[3*primary_idx + 2] < 0: # If negative make a small positive

            deriv_value = 1

            if not self.debug_ctl['warn_mute']:
                msg = ("[Cont-compression_deriv_mp] t = %f: Compression " + \
                       "deriv substitution. duaz_dpaz  = %0.3f, new " + \
                       "duaz_dpaz = %0.3f") % (t, du_dpaz[3*primary_idx + 2], \
                                               deriv_value)
                print(msg)

            du_dpaz[3*primary_idx + 2] = deriv_value

        du_duaz = du_dpaz/(du_dpaz[3*primary_idx + 2])

        # Calculate the work derivatives (dWz_duaz)
        dWz_duaz = np.square(h_vec)*(uaz/(du_dpaz[3*primary_idx + 2]))

        # Interweave velocity and work derivatives into an output
        dx_duaz = np.zeros(3*num_points)
        for i in range(num_points):
            dx_duaz[3*i:(3*i + 2)] = du_duaz[3*i:(3*i + 2)]
            dx_duaz[3*i + 2] = dWz_duaz[i]

        return dx_duaz

    def restitution_deriv_mp(self, Wz_in: np.ndarray, x: np.ndarray, 
                             args: list):
        '''
        Method to carry out the restitution-phase derivative for multi-point
        contact

        State vector: x = [uax, uay, uaz, ubx, uby, ubz, ..., uNx, uNy, uNz]
        Returns dx: dx_dWaz = du_dWaz
        '''

        # Additional arguments
        K_in = args[0]
        K = copy.deepcopy(K_in)
        h_vec = args[1]
        primary_idx = args[2]
        t = args[3]

        num_points = int(x.size/3)

        # Form unified vector of sliding vectors
        xi_uni = np.zeros(3*num_points)

        for i in range(num_points): # Loop through each contact point

            ux_i = x[3*i]
            uy_i = x[3*i + 1]
            uT_mag = LA.norm(np.array([ux_i, uy_i])) # Tangential magnitude

            if uT_mag == 0: # If numerically zero
                uT_mag = 0.01*self.epsilon_t # Make a very small number

            xi_i = np.array([-self.mu*ux_i/uT_mag, -self.mu*uy_i/uT_mag, 1])

            if i < num_points: # Assign to unified vec
                xi_uni[(3*i):(3*i + 3)] = xi_i*h_vec[i] 
            else:
                xi_uni[(3*i):] = xi_i*h_vec[i]

        # Form unified "A" matrix of K matrices
        K = K.transpose((0, 2, 1, 3))
        K_uni = K.reshape(3*num_points, 3*num_points)

        # If any z values negative, make small positive
        K_xi_uni = K_uni@xi_uni
        z_vec = K_xi_uni[2::3]
        zvec_nonzero = np.nonzero(z_vec < 0)[0]
        neg_idx = 3*zvec_nonzero + 2
        if len(neg_idx) > 0:

            deriv_value = 1

            if not self.debug_ctl['warn_mute']:
                msg = ("[Cont-crestitution_deriv_mp] t = %f: Restitution " +  
                       "deriv substitution. K_xi_uni negative indices = ") % \
                        (t) + np.array_str(neg_idx) + "\nnegative values = " + \
                        np.array_str(K_xi_uni[neg_idx]) + (", replaced " + 
                        "with %0.3f") % deriv_value
                print(msg)

            K_xi_uni[neg_idx] = 1

        # Calculate the velocity derivatives
        uaz = x[3*primary_idx + 2]
        du_dWaz = (1/uaz)*K_xi_uni

        if self.debug_ctl['deriv']:
            msg = ("[Cont-restitution_deriv_mp] t = %f") % (t) + \
                "\n\tK_xi_uni = " + np.array_str(K_xi_uni) + \
                "\n\tdu_dWaz = " + np.array_str(du_dWaz) + \
                "\n\tx = " + np.array_str(x) + "\n\tuaz = %f" % uaz
            self.debug_statements.append(msg)
            print(msg)

        return du_dWaz

    def microcollision_compute_mp(self, u0_C: np.ndarray, K_in: np.ndarray):
        '''
        Method to compute microcollision impulses for multi-point collisions
        '''

        num_points = u0_C.shape[0]

        # Compute and form change in velocity vector
        del_u = -2*u0_C.flatten()
        del_u = del_u.astype(float)

        # Form A matrix for impulse-momentum system
        K = copy.deepcopy(K_in)
        K = K.transpose((0, 2, 1, 3))
        A = K.reshape(3*num_points, 3*num_points)

        # Solve system of equations for impulse vector
        p = LA.lstsq(A, del_u, rcond=None)[0]

        return p

    def ext_limit_calc_mp(self, del_Wz: np.ndarray, K: np.ndarray, 
                          h1_comp: np.ndarray):
        '''
        Method to compute the limits of integration for the extension phase 
        in the multi-point contact case
        '''

        num_points = h1_comp.size

        # Loop through all points to compute bounding denomentator
        den = 0
        for i in range(num_points):

            # Isolate the variables to use for this point
            K_i = K[0, i] 
            h_i = h1_comp[i]

            K_tan = np.array([K_i[2, 0], K_i[2, 1]])
            K_check1 = K_i[2, 2] > self.mu*np.sum(K_tan)
            K_check2 = K_i[2, 2] > 0

            if all([K_check1, K_check2]): # If bound going to give a valid val
                den += h_i*(K_i[2, 2] - self.mu*np.sum(K_tan))
            elif K_check2: # If K33 is at least positive
                den += h_i*0.05*K_i[2, 2]
            else: # Use dummy value
                Kval = 0.2
                den += h_i*0.05*Kval

        # Form bounding constant vectors
        g_1 = (1/den)*np.square(h1_comp)

        # Compute the extension limits
        vec_1 = np.divide(del_Wz, g_1).astype(float)
        uz2_C = np.sqrt(2*vec_1)

        return uz2_C

    def micro_check_mp(self, u0_C: np.ndarray):
        '''
        Method to check if all points meet the microcollision bounds

        Conditions to check for microcollisions:
            - Within sticking bounds for tangential velocity
            - Within sqrt(2*g*epsilon_c) of normal velocity
        '''

        # Check tangent velocities
        uT_mag = LA.norm(u0_C[:, :2].astype(float), axis=1)
        uT_check = all(uT_mag < self.epsilon_t)
        
        # Check normal velocities
        uz_vec = u0_C[:, 2].flatten()
        uz_check = all(-uz_vec < np.sqrt(2*self.g*self.epsilon_c))

        check_vec = [uT_check, uz_check, self.micro_override]
        
        return all(check_vec)

    def cont_mat_calc_mp(self, r2_C: np.ndarray, I_C: np.ndarray):
        '''
        Method to calculate the constant collision matrices required for
        multi-point contact
        '''

        I_C_inv = LA.inv(I_C)
        num_points = r2_C.shape[0]

        K = np.zeros([num_points, num_points, 3, 3])

        for i in range(num_points): # Iterate over rows

            r_row = uf.make_skew(r2_C[i])

            for j in range(num_points): # Iterate over columns

                r_col = uf.make_skew(r2_C[j])

                K[i, j, :, :] = (1/self.m)*np.eye(3) - r_row@I_C_inv@r_col

        return K

    def track_mat_name_mp(self, num_points: int):
        '''
        Method to form the list of column names for the collision tracking
        dataframe
        '''

        name_list = []

        for i in range(num_points): # Loop through all points

            letter = string.ascii_lowercase[i]
            sub_list = ['u%sx' % letter, 'u%sy' % letter, 'u%sz' % letter, \
                        'W%sz' % letter]
            name_list.extend(sub_list)

        return name_list
