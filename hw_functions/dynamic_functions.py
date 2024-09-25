# library imports
import numpy as np

# our import
from . import helper_functions


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

def find_total_angular_momentum(masses, positions, velocities):

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

def find_rot_angular_momentum(inertia_tensor, angular_velocity):

    return inertia_tensor @ angular_velocity
    
def parallel_axis_theorem(total_mass, inertia_tensor_cm, inertia_offset):

    offset_tilde = helper_functions.get_tilde_matrix(inertia_offset)

    return inertia_tensor_cm + total_mass * offset_tilde @ np.transpose(offset_tilde)

def inertia_tensor_cordinate_transform(dcm, inertia_tensor):

    return dcm @ inertia_tensor @ np.transpose(dcm)

def find_principal_inertia_tensor(inertia_tensor):

    eigen_values, eigen_vectors = np.linalg.eig(inertia_tensor)
    eigen_vectors = np.transpose(eigen_vectors)

    eigen_vectors = eigen_vectors / np.linalg.norm(eigen_vectors)
    
    if np.dot( np.cross(eigen_vectors[0], eigen_vectors[1]), eigen_vectors[2] ) != 1:
        eigen_vectors[1] = -1 * eigen_vectors[1]

    return eigen_values, eigen_vectors

def sort_principal_inertia_tensor(eigen_values, eigen_vectors):

    # Combine eigenvalues and eigenvectors for sorting
    eigen_pairs = [(eigen_values[i], eigen_vectors[i]) for i in range(len(eigen_values))]

    # Sort by eigenvalue in descending order
    sorted_eigen_pairs = sorted(eigen_pairs, key=lambda x: x[0], reverse=True)

    # Separate sorted eigenvalues and eigenvectors
    sorted_eigenvalues = np.array([pair[0] for pair in sorted_eigen_pairs])
    sorted_eigenvectors = np.array([pair[1] for pair in sorted_eigen_pairs])

    return sorted_eigenvalues, sorted_eigenvectors
