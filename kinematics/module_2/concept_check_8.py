# system imports
import math

# library imports
import sympy as sp

# our imports
import hw_functions.dcm_functions as dcm_funcs


########## problem 1 ##########
x1, x2, x3 = math.radians(10), math.radians(20), math.radians(30)

dcm_321 = dcm_funcs.get_dcm("3-2-1", x1, x2, x3)
print('3-2-1 matrix')
print(dcm_321)

x1, x2, x3 = sp.symbols('x1 x2 x3')
symbolic_dcm_313 = dcm_funcs.get_dcm("3-1-3", x1, x2, x3)
print('3-1-3 symbolic matrix')
print(symbolic_dcm_313)

# solving for the angles
x2 = sp.acos(dcm_321[2, 2])

x1 = sp.atan(dcm_321[2, 0] / -dcm_321[2, 1])

x3 = sp.atan(dcm_321[0, 2] / dcm_321[1, 2])

x1, x2, x3 = math.degrees(x1), math.degrees(x2), math.degrees(x3)

print('%.5f, %.5f, %.5f' % (x1, x2, x3))


########## problem 2 ##########
print("\n\n\nPROBLEM 2")

bn_x1, bn_x2, bn_x3 = math.radians(10), math.radians(20), math.radians(30)

bn_dcm_321 = dcm_funcs.get_dcm("3-2-1", bn_x1, bn_x2, bn_x3)

rn_x1, rn_x2, rn_x3 = math.radians(-5), math.radians(5), math.radians(5)

rn_dcm_321 = dcm_funcs.get_dcm("3-2-1", rn_x1, rn_x2, rn_x3)

br_dcm_321 = bn_dcm_321 * sp.transpose(rn_dcm_321)

# printing the symbolic matrix
x1, x2, x3 = sp.symbols('x1 x2 x3')
symbolic_dcm_321 = dcm_funcs.get_dcm("3-2-1", x1, x2, x3)
print('3-2-1 symbolic matrix')
print(symbolic_dcm_321)

x2 = -sp.asin(br_dcm_321[0, 2])

x1 = sp.atan( br_dcm_321[0, 1] / br_dcm_321[0, 0] )

x3 = sp.atan( br_dcm_321[1, 2] / br_dcm_321[ 2, 2] )

x1, x2, x3 = math.degrees(x1), math.degrees(x2), math.degrees(x3)

print('%.5f, %.5f, %.5f' % (x1, x2, x3))
