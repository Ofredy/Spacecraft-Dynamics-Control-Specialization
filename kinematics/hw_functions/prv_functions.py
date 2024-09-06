import numpy as np


def get_prv_params_from_dcm(dcm):

    phi_upper = np.arccos( float( 0.5 * (np.trace(dcm) - 1) ) )

    e_vec = np.zeros(shape=3)

    # need to add singularity checker here -> divided by 0
    e_vec[0] = (1 / ( 2 * np.sin(phi_upper) )) * (dcm[1][2] - dcm[2][1])
    e_vec[1] = (1 / ( 2 * np.sin(phi_upper) )) * (dcm[2][0] - dcm[0][2])
    e_vec[2] = (1 / ( 2 * np.sin(phi_upper) )) * (dcm[0][1] - dcm[1][0])

    return e_vec, phi_upper
