
# Imports
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import string

# Single-point intertial-frame velocity calculation method
def vel_calc(points_B: np.ndarray, R: Rot, lin_vel_W: np.ndarray, 
             ang_vel_B: np.ndarray):
    '''
    Method to compute the inertial frame velocity of a body-fixed point

    Required Inputs:
        points_B: 1d numpy vec of body-fixed point coordinates
        R: Scipy rotation object, "apply" is body to inertial rotation
        lin_vel_if: 1d numpy vec of body linear velocity terms in inertial
        ang_vel_bf: 1d numpy vec of body angular velocity terms body fixed

    Notes:
    - Velocity of a point on a translating and rotating body [1, eq.3.5.6]
    - All coordinates follow x-forward, y-right, z-down
    '''

    # Compute angular contribution in body-fixed coords
    ang_cont_B = np.cross(ang_vel_B.astype(float), points_B.astype(float))
    ang_conts_W = R.apply(ang_cont_B)

    # Sum velocity components in inertial frame
    vels_W = lin_vel_W + ang_conts_W

    return vels_W # Return the point velocities

# Skew-symetric translation method
def make_skew(vec):
    '''
    Helper method to take a 1D numpy vector or a 2D numpy row/column vector
    and turn it into a 2D numpy array in skew symetric format

    See section 1: 
    https://lcvmwww.epfl.ch/teaching/modelling_dna/index.php?dir=exercises&file=corr02.pdf
    '''

    # Transfer into skew-symetric
    skew = np.array([[0, -vec[2], vec[1]], 
                        [vec[2], 0, -vec[0]], 
                        [-vec[1], vec[0], 0]])
    
    # Handeling the 2D row and column vectors is not currently implemented

    return skew

def pad(a, sep='\t'):
    from itertools import repeat
    t = repeat(sep)
    return '\n'.join(map(''.join, zip(t, str(a).split('\n'))))

def point_df_name(num_points: np.ndarray) -> list:
    '''
    Function to form the list of column names for the dataframe of collision
    point position and velocity
    '''

    name_list = ['t']

    for i in range(num_points): # Loop through all points

        letter = string.ascii_lowercase[i]
        sub_list = ['p%s_x' % letter, 'p%s_y' % letter, 'p%s_z' % letter, \
                    'p%s_{dx}' % letter, 'p%s_{dy}' % letter, \
                    'p%s_{dz}' % letter]
        name_list.extend(sub_list)

    return name_list


