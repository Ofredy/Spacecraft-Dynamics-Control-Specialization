# library imports
import numpy as np

# our imports
from hw_functions import attitude_estimation


########## problem 6 ##########
sensor_measurements = np.array([[ 0.8273, 0.5541, -0.0920 ],
                                [ -0.8285, 0.5522, -0.0955 ]])

inertials = np.array([[ -0.1517, -0.9669, 0.2050 ],
                      [ -0.8393, 0.4494, -0.3044 ]])


print(attitude_estimation.devenport_q_method(sensor_measurements, inertials)) 

########## example problem ##########
sensor_measurements = np.array([[ 0.8190, -0.5282, 0.2242 ],
                                [ -0.3138, -0.1584, 0.9362 ]])

inertials = np.array([[ 1.0, 0.0, 0.0 ],
                      [ 0.0, 0.0, 1.0 ]])


print(attitude_estimation.devenport_q_method(sensor_measurements, inertials)) 