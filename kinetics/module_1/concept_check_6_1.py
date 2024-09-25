# library imports
import math
import numpy as np

# our imports
from hw_functions import mrp_functions, dynamic_functions


######## problem 1 ########
mrps_D_B = np.array([0.1, 0.2, 0.3])
dcm_D_B = mrp_functions.mrp_to_dcm(mrps_D_B)

inertia_tensor_B = np.array([[ 10, 1, -1 ],
                             [ 1, 5, 1 ],
                             [ -1, 1, 8 ]]) 

print(dynamic_functions.inertia_tensor_cordinate_transform(dcm_D_B, inertia_tensor_B))


######## problem 2 ########
inertia_tensor_B = np.array([[ 10, 1, -1 ],
                             [ 1, 5, 1 ],
                             [ -1, 1, 8 ]]) 

principal_inertia_values, principal_inertia_dcm = dynamic_functions.find_principal_inertia_tensor(inertia_tensor_B)

sorted_principal_inertia_values, _ = dynamic_functions.sort_principal_inertia_tensor(principal_inertia_values, principal_inertia_dcm)

print(sorted_principal_inertia_values)


######## problem 4 ########
inertia_tensor_B = np.array([[ 10, 1, -1 ],
                             [ 1, 5, 1 ],
                             [ -1, 1, 8 ]]) 

principal_inertia_values, principal_inertia_dcm = dynamic_functions.find_principal_inertia_tensor(inertia_tensor_B)

_, sorted_principal_inertia_dcm = dynamic_functions.sort_principal_inertia_tensor(principal_inertia_values, principal_inertia_dcm)

print(sorted_principal_inertia_dcm)
