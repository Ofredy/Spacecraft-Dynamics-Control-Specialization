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
