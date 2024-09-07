import numpy as np

from . import quaternions, helper_functions


def mrp_is_long(mrp):

    if np.linalg.norm(mrp) >= 1:
        return True

    else:
        return False

def get_shadow_mrp(mrp):

    return ( -1 * mrp ) / np.linalg.norm(mrp)**2

def invert_mrp(mrp):

    return -1 * mrp

def mrp_to_dcm(mrp):

    mrp_norm = np.linalg.norm(mrp)
    mrp_tilde = helper_functions.get_tilde_matrix(mrp)

    return np.eye(3) + ( 8 * mrp_tilde @ mrp_tilde - 4 * ( 1 - mrp_norm**2 ) * mrp_tilde ) / ( 1 + mrp_norm**2 )**2

def quaternions_to_mrp(beta):

    mrp = np.zeros(shape=(3))

    mrp[0] = beta[1] / ( 1 + beta[0] )
    mrp[1] = beta[2] / ( 1 + beta[0] )
    mrp[2] = beta[3] / ( 1 + beta[0] )

    return mrp

def dcm_to_mrp(dcm):

    beta = quaternions.dcm_to_quaternions(dcm)

    return quaternions_to_mrp(beta)

def mrp_addition(mrp_2, mrp_1):

    # {mrp_FB} = {mrp_FN} * {mrp_BN}
    mrp_1_abs_squared = np.dot(np.abs(mrp_1), np.abs(mrp_1))
    mrp_2_abs_squared = np.dot(np.abs(mrp_2), np.abs(mrp_2))

    mrp =  ( ( 1 - mrp_1_abs_squared) * mrp_2 + ( 1 - mrp_2_abs_squared ) * mrp_1 - 2 * np.cross(mrp_2, mrp_1) ) \
           / ( 1 + mrp_1_abs_squared * mrp_2_abs_squared - 2 * np.dot(mrp_1, mrp_2) )

    if mrp_is_long(mrp):
        return get_shadow_mrp(mrp)

    else:
        return mrp
