import math

import numpy as np

from . import dcm_functions 

def quaternion_to_dcm(beta_vec):

    b0, b1, b2, b3 = beta_vec[0], beta_vec[1], beta_vec[2], beta_vec[3] 

    return np.array([[ b0**2 + b1**2 - b2**2 - b3**2,
                       2 * ( b1*b2 + b0*b3 ),
                       2 * ( b1*b3 - b0*b2 ) ],
                     [ 2 * ( b1*b2 -b0*b3 ),
                       b0**2 - b1**2 + b2**2 - b3**2,
                       2 * ( b2*b3 + b0*b1 ) ],
                     [ 2 * ( b1*b3 + b0*b2 ),
                       2 * ( b2*b3 - b0*b1 ),
                       b0**2 - b1**2 - b2**2 +b3**2 ]])

def shepperd_method_0(dcm, b0):

    b1 = ( dcm[1][2] - dcm[2][1] ) / ( 4*b0 )
    b2 = ( dcm[2][0] - dcm[0][2] ) / ( 4*b0 )
    b3 = ( dcm[0][1] - dcm[1][0] ) / ( 4*b0 )

    return np.array([b0, b1, b2, b3])

def shepperd_method_1(dcm, b1):

    b0 = ( dcm[1][2] - dcm[2][1] ) / ( 4*b1 )
    b2 = ( dcm[0][1] + dcm[1][0] ) / ( 4*b1 )
    b3 = ( dcm[2][0] + dcm[0][2] ) / ( 4*b1 )

    return np.array([b0, b1, b2, b3])

def shepperd_method_2(dcm, b2):

    b0 = ( dcm[2][0] - dcm[0][2] ) / ( 4*b2 )
    b1 = ( dcm[0][1] + dcm[1][0] ) / ( 4*b2 )
    b3 = ( dcm[1][2] + dcm[2][1] ) / ( 4*b2 )

    return np.array([b0, b1, b2, b3])

def shepperd_method_3(dcm, b3):

    b0 = ( dcm[0][1] - dcm[1][0] ) / ( 4*b3 )
    b1 = ( dcm[2][0] + dcm[0][2] ) / ( 4*b3 )
    b2 = ( dcm[1][2] + dcm[2][1] ) / ( 4*b3 )

    return np.array([b0, b1, b2, b3])

def is_short_way_quaternion(b0):

    if b0 > 0:
        return True

    else:
        return False

shepperd_hash = { 0: shepperd_method_0,
                  1: shepperd_method_1,
                  2: shepperd_method_2,
                  3: shepperd_method_3 }

def dcm_to_quaternions(dcm):

    beta_vec_squared = [ (1/4) * ( 1 + np.trace(dcm) ),
                         (1/4) * ( 1 + 2*dcm[0][0] - np.trace(dcm) ),
                         (1/4) * ( 1 + 2*dcm[1][1] - np.trace(dcm) ),
                         (1/4) * ( 1 + 2*dcm[2][2] - np.trace(dcm) ) ]

    beta_idx = beta_vec_squared.index(max(beta_vec_squared))

    beta_vec = shepperd_hash[beta_idx](dcm, math.sqrt(beta_vec_squared[beta_idx]))

    if is_short_way_quaternion(beta_vec[0]):
        return beta_vec

    else:
        return shepperd_hash[beta_idx](dcm, -math.sqrt(beta_vec_squared[beta_idx]))

def invert_quaternion(beta):

    return np.array([ beta[0], -beta[1], -beta[2], -beta[3] ])

def quaternion_addition(beta_vec_1, beta_vec_2):

    # example of what this does: q_FB = [{q_FN}] * q_BN

    beta_FN_matrix = np.array([[ beta_vec_1[0], -beta_vec_1[1], -beta_vec_1[2], -beta_vec_1[3] ],
                            [ beta_vec_1[1], beta_vec_1[0], beta_vec_1[3], -beta_vec_1[2] ],
                            [ beta_vec_1[2], -beta_vec_1[3], beta_vec_1[0], beta_vec_1[1] ],
                            [ beta_vec_1[3], beta_vec_1[2], -beta_vec_1[1], beta_vec_1[0] ]])

    beta_FB = np.dot(beta_FN_matrix, beta_vec_2)

    if beta_FB[0] > 0:
        return beta_FB

    else:
        return -1 * beta_FB
