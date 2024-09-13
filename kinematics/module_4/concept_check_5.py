# library imports
import numpy as np

from hw_functions import attitude_estimation


########### problem 5 ###########
sensor_measurements = np.array([[ 0.8273, 0.5541, -0.0920 ],
                                [ -0.8285, 0.5522, -0.0955 ]])

inertials = np.array([[ -0.1517, -0.9669, 0.2050 ],
                      [ -0.8393, 0.4494, -0.3044 ]])

print(attitude_estimation.quest_method(sensor_measurements, inertials)) 
