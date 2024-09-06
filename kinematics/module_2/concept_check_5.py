#### File Descriptor ####

import numpy as np


BN = np.array([[-0.87097, 0.45161, 0.19355], 
               [-0.19355, -0.67742, 0.70968],
                [ 0.45161, 0.58065, 0.67742]])

tilde_W_b_n = np.array([[ 0, -0.3, 0.2 ],
                        [ 0.3, 0, -0.1 ],
                        [ -0.2, 0.1, 0 ]]) 

print( np.dot(-1*tilde_W_b_n, BN) )