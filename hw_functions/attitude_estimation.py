import math

import numpy as np

from . import prv_functions, quaternions, crps


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

def devenport_q_method(sensor_measurements, inertials, weights=None):

    if weights == None:
        weights = np.ones(shape=sensor_measurements.shape[0])

    sensor_norms = np.linalg.norm(sensor_measurements, axis=1, keepdims=True)
    sensor_measurements /= sensor_norms

    B = np.zeros(shape=(3, 3))

    for idx, measurement in enumerate(sensor_measurements):

        measurement = np.expand_dims(measurement, axis=-1)
        inertial = np.expand_dims(inertials[idx], axis=-1)

        B += weights[idx] * measurement @ np.transpose(inertial)

    S = B + np.transpose(B)

    sigma = np.trace(B)

    Z = np.array([[ B[1][2]-B[2][1] ], [ B[2][0]-B[0][2] ], [ B[0][1]-B[1][0] ]])  # 3x1 vector

    # Create the 4x4 matrix
    K = np.block([[sigma, Z.T], [Z, S-sigma*np.eye(3)]])

    eigenvalues, eigenvectors = np.linalg.eig(K)

    beta = eigenvectors[:, np.argmax(eigenvalues)]

    if not quaternions.is_short_way_quaternion(beta[0]):
        beta = beta * -1

    return quaternions.quaternion_to_dcm(beta)

def f_opt_eig(opt_eig, K):

    return np.linalg.det( K - opt_eig*np.eye(4) )

def f_p_opt_eig(opt_eig, K):

    return -1 * np.linalg.det( K - opt_eig*np.eye(4) ) * np.trace( np.linalg.inv( K - opt_eig*np.eye(4) ) )

def opt_eig_newton(opt_eig, K):

    for _ in range(3):

        opt_eig = opt_eig - f_opt_eig(opt_eig, K) / f_p_opt_eig(opt_eig, K)

    return opt_eig

def quest_method(sensor_measurements, inertials, weights=None):

    if weights == None:
        weights = np.ones(shape=sensor_measurements.shape[0])

    opt_eig = np.sum(weights)

    sensor_norms = np.linalg.norm(sensor_measurements, axis=1, keepdims=True)
    sensor_measurements /= sensor_norms

    B = np.zeros(shape=(3, 3))

    for idx, measurement in enumerate(sensor_measurements):

        measurement = np.expand_dims(measurement, axis=-1)
        inertial = np.expand_dims(inertials[idx], axis=-1)

        B += weights[idx] * measurement @ np.transpose(inertial)

    S = B + np.transpose(B)

    sigma = np.trace(B)

    Z = np.array([[ B[1][2]-B[2][1] ], [ B[2][0]-B[0][2] ], [ B[0][1]-B[1][0] ]])  # 3x1 vector

    # Create the 4x4 matrix
    K = np.block([[sigma, Z.T], [Z, S-sigma*np.eye(3)]])

    opt_eig = opt_eig_newton(opt_eig, K)

    # solving for crps
    q_bar = np.linalg.inv( ( opt_eig + sigma )*np.eye(3) - S ) @ Z

    return crps.crp_to_dcm(q_bar)

def estimate_error(true_dcm, estimated_dcm):

    BbarB_dcm = np.dot(estimated_dcm, np.transpose(true_dcm))

    _, phi_upper = prv_functions.get_prv_params_from_dcm(BbarB_dcm)

    return math.degrees(phi_upper)
