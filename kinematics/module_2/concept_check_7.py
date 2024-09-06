import numpy as np

# Define the 3-2-1 Euler angles
psi_321, theta_321, phi_321 = 20, 10, -10

# Rotation matrices for 3-2-1 sequence
R_z_321 = np.array([
    [np.cos(psi_321), np.sin(psi_321), 0],
    [-np.sin(psi_321), np.cos(psi_321), 0],
    [0, 0, 1]
])

R_y_321 = np.array([
    [np.cos(theta_321), 0, -np.sin(theta_321)],
    [0, 1, 0],
    [np.sin(theta_321), 0, np.cos(theta_321)]
])

R_x_321 = np.array([
    [1, 0, 0],
    [0, np.cos(phi_321), np.sin(phi_321)],
    [0, -np.sin(phi_321), np.cos(phi_321)]
])

# Overall rotation matrix
R = R_z_321 @ R_y_321 @ R_x_321

# Now extract 3-1-3 Euler angles
psi_313 = np.arctan2(R[0, 2], -R[1, 2])
phi_313 = np.arccos(R[1, 1])
psi_prime_313 = np.arctan2(R[2, 0], -R[2, 1])

# Convert back to degrees if needed
yaw_313, roll_313, yaw_prime_313 = np.radians([psi_313, phi_313, psi_prime_313])

print(f"3-1-3 Euler angles: {yaw_313}, {roll_313}, {yaw_prime_313}")