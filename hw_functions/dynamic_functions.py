# library imports
import numpy as np


def find_cm(masses, positions):

    total_mass = np.sum(masses)

    return total_mass, np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass

def find_velocity_of_cm(masses, velocities):

    total_mass = np.sum(masses)

    return total_mass, np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass

def find_total_ke(masses, velocities):

    total_mass, velocity_cm = find_velocity_of_cm(masses, velocities)

    # solving for translational ke
    translational_ke = (1/2) * total_mass * np.vdot(velocity_cm, velocity_cm)

    # solving for rotational ke
    relative_velocities = velocities - velocity_cm

    rotational_ke = 0
    for idx in range(relative_velocities.shape[0]):

        rotational_ke += masses[idx] * np.vdot(relative_velocities[idx], relative_velocities[idx])

    rotational_ke = (1/2) * np.sum(rotational_ke)

    return translational_ke, rotational_ke

def find_linear_momentum(masses, velocities):

    total_mass, velocity_cm = find_velocity_of_cm(masses, velocities)

    return total_mass * velocity_cm

def find_angular_momentum(masses, positions, velocities):

    # angular momentum of origin
    origin_am = np.zeros(shape=positions[0].shape)

    for idx in range(positions.shape[0]):

        origin_am += np.cross(positions[idx], masses[idx] * velocities[idx]) 

    # angular momentum about the center of mass
    _, center_of_mass = find_cm(masses, positions)
    _, velocity_cm = find_velocity_of_cm(masses, velocities)

    relative_positions = positions - center_of_mass
    relative_velocities = velocities - velocity_cm

    cm_am = np.zeros(shape=relative_positions[0].shape)
    for idx in range(relative_positions.shape[0]):

        cm_am += np.cross(relative_positions[idx], masses[idx] * relative_velocities[idx])

    return origin_am, cm_am
