
# Imports
import numpy as np
from numpy import linalg as LA

import pandas as pd

from math import ceil, floor

import copy

# Add workspace directory to the path
import os
import sys
sim_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, sim_pkg_path)

# Workspace package imports
from amls_sim.simulator import rk4_step_fun

import amls_impulse_contact.utility_functions as uf
from amls_impulse_contact.utility_functions import pad

class SinglePoint:
    '''
    Mixin class containing the single-point contact methods.
    This class is methods only

    Methods:
        contact_calc_sp
        compression_calc_sp
        extension_calc_sp
        restitution_calc_sp
        compression_deriv_sp
        restitution_deriv_sp
    '''

    def contact_calc_sp(self, r2_C: np.ndarray, I_C: np.ndarray, 
                        u0_C: np.ndarray, t: float=0):
        '''
        Method to carry out the computation of contact for a given point

        Required Inputs:
            r2_C: np 1d vec of C-frame contact-point x,y,z positions
            I_C:
            u0_C: 
        Optional Inputs:
            t: time given , only used for debug statements currently

        '''

        # Solve for the collision matrix K
        r2_C_skew = uf.make_skew(r2_C)
        I_C_inv = LA.inv(I_C)
        K = (1/self.m)*np.eye(3) - r2_C_skew@I_C_inv@r2_C_skew
        K_inv = LA.inv(K) # Take inverse of K

        if self.debug_ctl['contact']:
            msg = "[Cont-contact_calc_sp] t = %f: \n\tK = " % (t) + \
                np.array_str(K[0]) + '\n' + pad(K[1:], sep='\t   ')
            self.debug_statements.append(msg)
            print(msg)

        # # Perform sticking stability check
        # (5/16/23) - Implement if needed
        # if (K_inv[0,2]**2+K_inv[1,2]**2) <= (self.mu**2)*K_inv[2,2]**2:
        #     stick_stable = 'stable'
        # else:
        #     stick_stable = 'unstable'

        # Initialize tracking matrices to 0
        track_mat_comp = pd.DataFrame(columns=['ux', 'uy', 'uz', 'Wz'])
        track_mat_ext = pd.DataFrame(columns=['ux', 'uy', 'uz', 'Wz'])
        track_mat_rest = pd.DataFrame(columns=['ux', 'uy', 'uz', 'Wz'])
        
        # Compute impulse on body from normal or micro collisions
        uT_mag = LA.norm(u0_C[0:2]) # Tangential velocity mag
        uT_check = uT_mag < self.epsilon_t # If in sliding contact
        uz_if0 = u0_C[2] # Isolate z component
        uz_check = -uz_if0 < np.sqrt(2*self.g*self.epsilon_c)
        micro_check = all([uT_check, uz_check])

        if micro_check: # In microcollision

            p_r = -2*K_inv@u0_C # Calculate microcollision impulse
            p_C = p_r

            if self.debug_ctl['contact']:
                msg = ("[Cont-contact_calc_sp] t = %f: microcollision " + \
                        "impulse (p) = ") % (t) + np.array2string(p_C) + "^T"
                self.debug_statements.append(msg)
                print(msg)

        else: # Normal collision

            if self.debug_ctl['contact']:
                msg = "[Cont-contact_calc_sp] t = %f: u0_C = " % (t) + \
                    np.array2string(u0_C)
                self.debug_statements.append(msg)
                print(msg)

            u1_C, Wz1, track_mat_comp = self.compression_calc_sp(u0_C, K, t)

            if self.debug_ctl['contact']:
                msg = "[Cont-contact_calc_sp] t = %f: u1_C = " % (t) + \
                    np.array2string(u1_C) + " Wz1 = " + str(Wz1)
                self.debug_statements.append(msg)
                print(msg)

            u2_C, Wz2, track_mat_ext = self.extension_calc_sp(u1_C, Wz1, K, t)

            if self.debug_ctl['contact']:
                msg = "[Cont-contact_calc_sp] t = %f: u2_C = " % (t) + \
                    np.array2string(u2_C) + " Wz2 = " + str(Wz2)
                self.debug_statements.append(msg)
                print(msg)

            u3_C, Wz3, track_mat_rest = \
                self.restitution_calc_sp(u2_C, Wz1, Wz2, K, t)

            if self.debug_ctl['contact']:
                msg = "[Cont-contact_calc_sp] t = %f: u3_C = " % (t) + \
                    np.array2string(u3_C)
                self.debug_statements.append(msg)
                print(msg)

            p_C = (LA.inv(K))@(u3_C - u0_C)
            p_C = p_C.astype(float)

            if self.debug_ctl['contact']:
                msg = ("[Cont-contact_calc_sp] t = %f: normal collision "
                        "impulse (p) = ") % (t) + np.array2string(p_C) + "^T"
                self.debug_statements.append(msg)
                print(msg)

        # Apply impulse to velocities (negative because changing body 2)
        del_lin_vel_C = -(1/self.m)*p_C # Linear velocity
        del_ang_vel_C = -I_C_inv@np.cross(r2_C, p_C)

        if self.debug_ctl['contact']:
            msg = ("[Cont-contact_calc_sp] t = %f: del_v = ") % (t) + \
                np.array2string(del_lin_vel_C) + "^T, del_ang_vel = " + \
                np.array2string(del_ang_vel_C) + "^T"
            self.debug_statements.append(msg)
            print(msg)

        # Compile tracking matrix for export
        track_mat = pd.concat([track_mat_comp, track_mat_ext, track_mat_rest])

        return del_lin_vel_C, del_ang_vel_C, track_mat

    # Single-point compression calculation method
    def compression_calc_sp(self, u0_C: np.ndarray, K: np.ndarray, t: float):
        '''
        Method used to carry out the compression phase of contact integration
        for single point
        '''

        # Divide input variables out
        K33 = K[2, 2]

        # Initial sanity check:
        #   uz0_C < 0
        check_init1 = u0_C[2] < 0
        if not check_init1: # Raise exception if initial checks not passed
            msg = ("[compression_calc_sp] t = %f: Initial check failed, " + \
                   "uz0_C greater than 0") % t
            raise ValueError(msg)

        # Setup integration parameters based on compression limit
        step_check = abs(u0_C[2]) > self.min_num_steps*self.uz_step
        if step_check: # If limit larger than step size
            num_steps = floor(abs(u0_C[2])/self.uz_step) 
        else: # If limit less than step size
            num_steps = self.min_num_steps # Single step

        # Generate the uz integration vectors, also getting step size
        uz_vec, uz_step_size = np.linspace(u0_C[2], 0, num_steps, retstep=True)

        # Initialize tracking variables and matrix
        Wz_int = 0
        u_int = copy.deepcopy(u0_C).astype(float)
        tracking_mat = pd.DataFrame(columns=['ux', 'uy', 'uz', 'Wz'], 
                                    index=range(uz_vec.size))

        # Fill out tracking matrix with known values
        tracking_mat['uz'] = uz_vec
        tracking_mat.iloc[0, 3] = Wz_int
        tracking_mat.iloc[0, 0:3] = u0_C

        # First percent of steps check
        pct_exemp = 0.1 # Percent of steps not allowed to stick
        min_step_check = 1/pct_exemp # Number of steps needed for pct

        for i, uz in enumerate(uz_vec[1:]): # Loop uz values after the first

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
            uT_mag = LA.norm(u_int[0:2]) # Tangential velocity mag
            uT_check = uT_mag < self.epsilon_t # Within tangential bounds

            # Stick if past init pct of steps and all points in tan bounds
            if uT_check and pct_check:

                if self.debug_ctl['phase']:
                    msg = ("[Cont-compression_calc_sp] t = %f: in " + \
                        "sticking, u = ") % (t) + np.array2string(u_int) + \
                        "^T, Wz = %0.3f" % Wz_int
                    self.debug_statements.append(msg)
                    print(msg)

                # Sticking sanity check:
                #   K33 > 0
                check_stick1 = K33 > 0
                if not check_stick1: # Excep if stick checks failed
                    msg = ("[compression_calc_sp] t = %f: Sticking check " + \
                           "failed, K33 less than or equal to 0") % t
                    raise ValueError(msg)
                
                # Fill out previous Wz on tracking mat
                tracking_mat['Wz'].values[i:] = Wz_int

                # Resolve the compression phase algebraically
                Wz_int += 0.5*(1/K33)*(-u_int[2]**2)
                u_int = np.zeros(3) # Set sticking values at mc

                # Fill out remainder of tracking matrix
                tracking_mat['ux'].values[i:] = 0
                tracking_mat['uy'].values[i:] = 0
                tracking_mat.iloc[-1, 3] = Wz_int
                
                break # Break the for loop

            else: # In sliding

                if self.debug_ctl['phase']:
                    msg = ("[Cont-compression_calc_sp] t = %f: in " + \
                        "sliding, u = ") % (t) + np.array2string(u_int) + \
                        "^T, Wz = %0.3f" % Wz_int
                    self.debug_statements.append(msg)
                    print(msg)

                # Step ux, uy, Wz forward with numerical integration
                state_vec = np.array([u_int[0], u_int[1], Wz_int])
                state_step = rk4_step_fun(self.compression_deriv_sp, state_vec,
                                          u_int[2], uz_step_size, K)

                # Save new states to variables
                u_int[0] = state_step[0]
                u_int[1] = state_step[1]
                u_int[2] = uz
                Wz_int = state_step[2]

                # Save to tracking matrix
                tracking_mat.iloc[i, 0:3] = u_int
                tracking_mat.iloc[i, 3] = Wz_int

        if self.debug_ctl['phase']:
            msg = ("[Cont-compression_calc_sp] t = %f: compression " + \
                "complete, u = ") % (t) +  np.array2string(u_int) + \
                "^T, Wz = %0.10f" % Wz_int
            self.debug_statements.append(msg)
            print(msg)

        # Set output variables
        u1_C = u_int
        Wz1 = Wz_int

        # Output sanity checks:
        #   uz1_C = 0
        #   Wz1 < 0
        check_final1 = u1_C[2] == 0
        check_final2 = Wz1 < 0
        if not check_final1:
            msg = ("[compression_calc_sp] t = %f: Final check failed, " + \
                   "uz1_C not equal to 0, uz1_C = %f") % (t, u1_C[2])
            raise ValueError(msg)
        if not check_final2:
            msg = ("[compression_calc_sp] t = %f: Final check failed, " + \
                   "Wz1 greater than 0, Wz1 = %0.10f") % (t, Wz1)
            raise ValueError(msg)

        return u1_C, Wz1, tracking_mat

    # Single-point compression extension calculation method
    def extension_calc_sp(self, u1_C: np.ndarray, Wz1: float, K: np.ndarray, 
                          t: float):
        '''
        Method used to carry out the extension phase of the contact integration
        for single point
        '''

        # Divide out input variables
        K31 = K[2, 0]
        K32 = K[2, 1]
        K33 = K[2, 2]

        # Set uz upper bound for extension integration
        Wz3 = Wz1*(1 - self.e**2)
        del_Wz = Wz3 - Wz1

        # Initial sanity checks (after Wz3 is calculated):
        #   uz1_C = 0
        #   0 > Wz3 > Wz1
        #   K33 > 0
        check_init1 = u1_C[2] == 0
        check_init2 = K33 > 0
        check_init3 = Wz3 < 0
        check_init4 = Wz3 > Wz1
        if not check_init1:
            msg = ("[extension_calc_sp] t = %f: Initial check failed, " + \
                   "uz1_C not equal to 0, uz1_C = %f") % (t, u1_C[2])
            raise ValueError(msg)
        if not check_init2:
            msg = ("[extension_calc_sp] t = %f: Initial check failed, " + \
                   "K33 less than or equal to 0, K33 = %f") % (t, K33)
            raise ValueError(msg)
        if not check_init3:
            msg = ("[extension_calc_sp] t = %f: Initial check failed, " + \
                   "Wz3 greater than or equal to 0, Wz3 = %f") % (t, Wz3)
            raise ValueError(msg)
        if not check_init4:
            msg = ("[extension_calc_sp] t = %f: Initial check failed, " + \
                   "Wz3 less than or equal to Wz1, Wz3 = %f, Wz1 = %f") % \
                   (t, Wz3, Wz1)
            raise ValueError(msg)

        # Compute extension limits given K to make sure they are valid
        K_check = self.mu*np.sqrt(K31**2 + K32**2) < K33
        if K_check: # If bound going going to give a valid value  
            uz2_C = 0.25*np.sqrt(2*del_Wz*(K33 - self.mu*\
                                           np.sqrt(K31**2+K32**2)))
        else: # If bound going to give an invalid number
            uz2_C = 0.25*np.sqrt(2*del_Wz*(0.05*K33))

        # Set up integration parameters based on extension limit
        step_check = uz2_C > self.min_num_steps*self.uz_step
        if step_check: # If limit larger than step size
            num_steps = floor(uz2_C/self.uz_step)
        else: # If limit less than step size
            num_steps = self.min_num_steps

        # Generate the uz integration vectors, also getting step size
        uz_vec, uz_step_size = np.linspace(u1_C[2], uz2_C, num_steps, 
                                           retstep=True)
        
        # Initialize tracking variables and matrix
        u_int = copy.deepcopy(u1_C)
        Wz_int = copy.deepcopy(Wz1)
        tracking_mat = pd.DataFrame(columns=['ux', 'uy', 'uz', 'Wz'], 
                                    index=range(uz_vec.size))

        # Fill out tracking matrix with known values
        tracking_mat['uz'] = uz_vec
        tracking_mat.iloc[0, 3] = Wz1
        tracking_mat.iloc[0, 0:3] = u1_C

        for i, uz in enumerate(uz_vec[1:]): # Loop uz values after the first

            # Tangential velocity check
            uT_mag = LA.norm(u_int[0:2]) # Tangential velocity mag
            uT_check = uT_mag < self.epsilon_t # Within tangential bounds

            # In sticking if within tangential bounds
            if uT_check: 

                if self.debug_ctl['phase']:
                    msg = ("[Cont-extension_calc_sp] t = %f: in " + \
                        "sticking, u = ") % (t) + np.array2string(u_int) + \
                        "^T, Wz = %0.3f" % Wz_int
                    self.debug_statements.append(msg)
                    print(msg)

                # Fill out previous Wz on tracking mat
                tracking_mat['Wz'].values[i:] = Wz_int

                # Resolve the compression phase algebraically
                Wz_int += 0.5*(1/K33)*(uz2_C**2 - u_int[2]**2)
                u_int = np.array([0, 0, uz2_C]) # Set sticking values at mc

                # Fill out remainder of tracking matrix
                tracking_mat['ux'].values[i:] = 0
                tracking_mat['uy'].values[i:] = 0
                tracking_mat.iloc[-1, 3] = Wz_int

                break # Break the for loop

            else: # In sliding

                if self.debug_ctl['phase']:
                    msg = ("[Cont-extension_calc_sp] t = %f: in " + \
                        "sliding, u = ") % (t) + np.array2string(u_int) + \
                        "^T, Wz = %0.3f" % Wz_int
                    self.debug_statements.append(msg)
                    print(msg)

                # Step ux, uy, Wz forward with numerical integration
                state_vec = np.array([u_int[0], u_int[1], Wz_int])
                state_step = rk4_step_fun(self.compression_deriv_sp, state_vec,
                                          u_int[2], uz_step_size, K)

                # Save new states to variables
                u_int[0] = state_step[0]
                u_int[1] = state_step[1]
                u_int[2] = uz
                Wz_int = state_step[2]   

                # Save to tracking matrix
                tracking_mat.iloc[i, 0:3] = u_int
                tracking_mat.iloc[i, 3] = Wz_int

        # Set output variables
        u2_C = u_int
        Wz2 = Wz_int

        # Output sanity checks
        #   uz2_C > 0
        #   Wz2 < Wz3
        #   Wz2 > Wz1
        check_final1 = u2_C[2] > 0
        check_final2 = Wz2 <= Wz3
        check_final3 = Wz2 > Wz1
        if not check_final1:
            msg = ("[extension_calc_sp] t = %f: Final check failed, " + \
                   "uz2_C less than or equal to 0, uz2_C = %f") % (t, u2_C[2])
            raise ValueError(msg)
        if not check_final2:
            msg = ("[extension_calc_sp] t = %f: Final check failed, " + \
                   "Wz2 greater than Wz3, Wz2 = %f, Wz3 = %f") % \
                   (t, Wz2, Wz3)
            raise ValueError(msg)
        if not check_final3:
            msg = ("[extension_calc_sp] t = %f: Final check failed, " + \
                   "Wz2 less than or equal to Wz1, Wz2 = %f, Wz1 = %f") % \
                   (t, Wz2, Wz1)
            raise ValueError(msg)

        return u2_C, Wz2, tracking_mat

    # Single-point restitution calculation method
    def restitution_calc_sp(self, u2_C: np.ndarray, Wz1: float, Wz2: float, 
                            K: np.ndarray, t: float):
        '''
        Method used to handle the restitution phase of contact integration
        for single point
        '''

        # Divide out input variables
        K33 = K[2, 2]

        # Set Wz upper bound for restitution integration
        Wz3 = Wz1*(1 - self.e**2)
        del_Wz = Wz3 - Wz2

        # Initial sanity checks (after Wz3 is calculated):
        #   uz2_C > 0
        #   0 > Wz3 > Wz2 > Wz1
        check_init1 = u2_C[2] > 0
        check_init2 = 0 > Wz3
        check_init3 = Wz3 > Wz2
        check_init4 = Wz2 > Wz1
        if not check_init1:
            msg = ("[restitution_calc_sp] t = %f: Initial check failed, " + \
                   "uz2_C less than or equal to 0, uz2_C = %f") % (t, u2_C[2])
            raise ValueError(msg)
        if not check_init2:
            msg = ("[restitution_calc_sp] t = %f: Initial check failed, " + \
                   "Wz3 greater than or equal to 0, Wz3 = %f") % (t, Wz3)
            raise ValueError(msg)
        if not check_init3:
            msg = ("[restitution_calc_sp] t = %f: Final check failed, " + \
                   "Wz3 less than or equal to Wz2, Wz3 = %f, Wz2 = %f") % \
                   (t, Wz3, Wz2)
            raise ValueError(msg)
        if not check_init4:
            msg = ("[restitution_calc_sp] t = %f: Final check failed, " + \
                   "Wz2 less than or equal to Wz1, Wz2 = %f, Wz1 = %f") % \
                   (t, Wz2, Wz1)
            raise ValueError(msg)

        # Setup integration parameters based on restitution limits
        step_check = del_Wz > self.min_num_steps*self.Wz_step
        if step_check: # Wz greater than step size
            num_steps = floor(del_Wz/self.Wz_step)
        else: # Wz greater than step size
            num_steps = self.min_num_steps

        # Generate the Wz integration vectors, also getting step size
        Wz_vec, Wz_step_size = np.linspace(Wz2, Wz3, num_steps, retstep=True)

        # Initialize tracking variables and matrix
        u_int = copy.deepcopy(u2_C)
        tracking_mat = pd.DataFrame(columns=['ux', 'uy', 'uz', 'Wz'], 
                                    index=range(Wz_vec.size))

        # Fill out tracking matrix with known values
        tracking_mat['Wz'] = Wz_vec
        tracking_mat.iloc[0, 0:3] = u2_C

        # Loop through all Wz values after the first
        for i, Wz in enumerate(Wz_vec[1:]): 

            # Tangential velocity check
            uT_mag = LA.norm(u_int[0:2]) # Tangential velocity mag
            uT_check = uT_mag < self.epsilon_t # Within tangential bounds

            # In sticking if within tangential bounds
            if uT_check: 

                if self.debug_ctl['phase']:
                    msg = ("[Cont-restitution_calc_sp] t = %f: in " + \
                        "sticking, u = ") % (t) + np.array2string(u_int) + \
                        "^T, Wz = %0.3f" % Wz
                    self.debug_statements.append(msg)
                    print(msg)

                # Sticking sanity check:
                #   K33 > 0
                check_stick1 = K33 > 0
                if not check_stick1: # Excep if stick checks failed
                    msg = ("[resitution_calc_sp] t = %f: Sticking check " + \
                           "failed, K33 less than or equal to 0") % t
                    raise ValueError(msg)
                
                # Fill out previous uz on tracking mat
                tracking_mat['uz'].values[i:] = u_int[2]

                # Resolve the restitution phase algebraically
                uz3_C = np.sqrt(u_int[2]**2 + (2*K33)*(Wz3 - Wz))
                u_int = np.array([0, 0, uz3_C])

                # Fill out remainder of tracking matrix
                tracking_mat['ux'].values[i:] = 0
                tracking_mat['uy'].values[i:] = 0  
                tracking_mat.iloc[-1, 2] = uz3_C

                break # Break the for loop

            else: # In sliding

                if self.debug_ctl['phase']:
                    msg = ("[Cont-restitution_calc_sp] t = %f: in " + \
                        "sliding, u = ") % (t) + np.array2string(u_int) + \
                        "^T, Wz = %0.3f" % Wz
                    self.debug_statements.append(msg)
                    print(msg)

                # Step ux, uy, uz forward with numerical integration
                u_int = rk4_step_fun(self.restitution_deriv_sp, u_int, 
                                     Wz_vec[i-1], Wz_step_size, K)

                # Save to tracking matrix
                tracking_mat.iloc[i, 0:3] = u_int

        if self.debug_ctl['phase']:
            msg = ("[Cont-restitution_calc_sp] t = %f: restitution " + \
                "complete, u = ") % (t) + np.array2string(u_int) + \
                "^T, Wz = %0.3f" % Wz
            self.debug_statements.append(msg)
            print(msg)

        # Set output variables
        u3_C = u_int

        # Output sanity checks
        #   uz3_C > uz2_C
        check_final1 = u3_C[2] >= u2_C[2]
        if not check_final1:
            msg = ("[restitution_calc_sp] t = %f: Final check failed, " + \
                   "uz3_C less than uz2_C, uz2_C = %f, uz3_C = %f") % \
                   (t, u2_C[2], u3_C[2])
            raise ValueError(msg)

        return u3_C, Wz3, tracking_mat

    # Single-point compression derivative calculation method
    def compression_deriv_sp(self, uz_in: float, x: np.ndarray, args):
        '''
        Method to evaluate the sliding vector and compression differential terms

        State vector x = [ux, uy, Wz]
        '''

        K = args[0]

        # Divide out input variables
        ux = x[0]
        uy = x[1]
        uz = uz_in
        u = np.array([ux, uy, uz])
        Kx = K[0, :]
        Ky = K[1, :]
        Kz = K[2, :]

        uT_mag = LA.norm(u[:2]) # Magnitude of ux and uy vector

        if uT_mag == 0: # If numerically zero
            uT_mag = 0.01*self.epsilon_t # Make a very small number

        # Compute sliding vector
        xi = np.array([(-self.mu*ux / uT_mag), (-self.mu*uy / uT_mag), 1])

        if self.debug_ctl['deriv']:
            msg = "[Cont-compression_deriv_sp] sliding_vec = " + \
                np.array2string(xi) + "^T"
            self.debug_statements.append(msg)
            print(msg)

        # Calculate differential ux, uy, Wz
        Kx_xi = Kx@xi
        Ky_xi = Ky@xi
        Kz_xi = Kz@xi

        if Kz_xi < 0: # If negative, make small positive
            Kz_xi = 1e3

        deriv_comp = (1 / Kz_xi)*np.array([Kx_xi, Ky_xi, uz])
        dux_duz = deriv_comp[0]
        duy_duz = deriv_comp[1]
        dWz_duz = deriv_comp[2]

        if self.debug_ctl['deriv']:
            msg = ("[Cont-compression_deriv]: uz = %f, Kx_xi = %f, " + \
                   " Ky_xi = %f, Kz_xi = %f, dux_duz = %0.3f, " + \
                   "duy_duz = %0.3f, dWz_duz = %0.3f,") % \
                    (uz, Kx_xi, Ky_xi, Kz_xi, dux_duz, duy_duz, dWz_duz)
            self.debug_statements.append(msg)
            print(msg)

        return deriv_comp # Return vector

    # Single-point restitution derivative calculation method
    def restitution_deriv_sp(self, Wz_in: float, u: np.ndarray, args):
        '''
        Method to evaluate the sliding vector and restitution differential terms
        '''

        K = args[0]

        # Divide out input variables
        ux = u[0]
        uy = u[1]
        uz = u[2]
        Kx = K[0, :]
        Ky = K[1, :]
        Kz = K[2, :]
        
        uT_mag = LA.norm(u[:2]) # Magnitude of ux and uy vector

        if uT_mag == 0: # If numerically zero
            uT_mag = 0.01*self.epsilon_t # Make a very small number

        # Compute sliding vector
        xi = np.array([(-self.mu*ux/uT_mag), (-self.mu*uy/uT_mag), 1])

        if self.debug_ctl['deriv']:
            msg = "[Cont-restitution_deriv] xi = " + \
                np.array2string(xi) + "^T"
            self.debug_statements.append(msg)
            print(msg)

        # Calculate differential ux, uy, uz
        Kx_xi = Kx@xi
        Ky_xi = Ky@xi
        Kz_xi = Kz@xi

        if Kz_xi < 0: # If negative, make small positive
            Kz_xi = 1e3

        deriv_rest = (1 / uz)*np.array([Kx_xi, Ky_xi, Kz_xi])
        dux_dWz = deriv_rest[0]
        duy_dWz = deriv_rest[1]
        duz_dWz = deriv_rest[2]

        if self.debug_ctl['deriv']:
            msg = ("Cont-restitution_deriv]: dux_dWz = %0.3f, " + \
                   "duy_dWz = %0.3f, duz_dWz = %0.3f,") % (dux_dWz, duy_dWz, 
                                                           duz_dWz)
            self.debug_statements.append(msg)
            print(msg)

        return deriv_rest # Return vector













