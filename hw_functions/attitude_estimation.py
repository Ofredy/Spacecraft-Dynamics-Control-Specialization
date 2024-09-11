import math

import numpy as np

from . import prv_functions


def triad_method(sensor_1, sensor_2, inertial_1, inertial_2):

    # norming inputs
    sensor_1 = sensor_1 / np.linalg.norm(sensor_1)
    sensor_2 = sensor_2 / np.linalg.norm(sensor_2)
    inertial_1 = inertial_1 / np.linalg.norm(inertial_1)
    inertial_2 = inertial_2 / np.linalg.norm(inertial_2)

    # t frame for b & n
    B_t1 = sensor_1
    B_t2 = np.cross(sensor_1, sensor_2) / np.linalg.norm(np.cross(sensor_1, sensor_2))
    B_t3 = np.cross(B_t1, B_t2)

    N_t1 = inertial_1 / np.linalg.norm(inertial_1)
    N_t2 = np.cross(inertial_1, inertial_2) / np.linalg.norm(np.cross(inertial_1, inertial_2))
    N_t3 = np.cross(N_t1, N_t2)

    Bbar_T_dcm = np.stack((B_t1, B_t2, B_t3), axis=1)
    NT_dcm = np.stack((N_t1, N_t2, N_t3), axis=1) 

    return np.dot(Bbar_T_dcm, np.transpose(NT_dcm))

def estimate_error(true_dcm, estimated_dcm):

    BbarB_dcm = np.dot(estimated_dcm, np.transpose(true_dcm))

    _, phi_upper = prv_functions.get_prv_params_from_dcm(BbarB_dcm)

    return math.degrees(phi_upper)
