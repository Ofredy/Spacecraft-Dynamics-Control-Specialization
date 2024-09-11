import numpy as np

from . import quaternions


def crp_to_dcm(q_vec):

    q1, q2, q3 = q_vec[0], q_vec[1], q_vec[2]

    return ( 1 / ( 1 + np.dot(np.transpose(q_vec), q_vec) ) ) * np.array([[ 1 + q1**2 - q2**2 - q3**2, 2*( q1*q2 + q3 ), 2*( q1*q3 - q2 ) ],
                                                                  [ 2*( q2*q1 - q3 ), 1 - q1**2 + q2**2 - q3**2, 2*( q2*q3 + q1 ) ],
                                                                  [ 2*( q3*q1 + q2 ), 2*( q3*q2 - q1 ), 1 - q1**2 - q2**2 + q3**2 ]])

def quaternions_to_crp(beta_vec):
    
    return (1/beta_vec[0]) * np.array([beta_vec[1], beta_vec[2], beta_vec[3]])

def dcm_to_crp(dcm):

    beta_vec = quaternions.dcm_to_quaternions(dcm)

    return quaternions_to_crp(beta_vec)

def invert_crp(q_vec):

    return -1 * q_vec

def crp_addition(q_vec_1, q_vec_2):

    # example of what this does: [{q_FB}] = [{q_FN}] * [{q_BN}}
    return ( q_vec_1 + q_vec_2 - np.cross(q_vec_1, q_vec_2) ) / ( 1 - np.dot(q_vec_1, q_vec_2) )