import sympy as sp


# rotation matrix across axis 1
def m1(x):

    return sp.Matrix([[ 1, 0, 0 ],
                      [ 0, sp.cos(x), sp.sin(x) ],
                      [ 0, -sp.sin(x), sp.cos(x) ]])

# rotation matrix across axis 2
def m2(x):

    return sp.Matrix([[ sp.cos(x), 0, -sp.sin(x) ],
                      [ 0, 1, 0 ],
                      [ sp.sin(x), 0, sp.cos(x) ]])

# rotation matrix across axis 3
def m3(x):

    return sp.Matrix([[ sp.cos(x), sp.sin(x), 0 ],
                      [ -sp.sin(x), sp.cos(x), 0 ],
                      [ 0, 0, 1 ]])

rotation_hash = { "1": m1,
                  "2": m2,
                  "3": m3 }

# create dcm for the respective euler angle
def get_dcm(euler_angle_set: str, x1, x2, x3):

    euler_angle_set = euler_angle_set.split('-')

    dcm = rotation_hash[euler_angle_set[0]](x1)
    dcm = rotation_hash[euler_angle_set[1]](x2) * dcm
    return rotation_hash[euler_angle_set[2]](x3) * dcm


if __name__ == "__main__":

    # creating a numerical matrix
    print('straight numerical solution')
    print(get_dcm("3-2-1", 10, 20, 30).evalf())

    # creating a symbolic matrix
    x1, x2, x3 = sp.symbols('x1, x2, x3')

    symbolic_dcm = get_dcm("3-2-1", x1, x2, x3)

    print('symbolic dcm')
    print(symbolic_dcm)

    print('inverse of symbolic dcm')
    # Note -> does not look right, try matlab or mathmatica
    print(sp.simplify(symbolic_dcm.inv()))

    print('numerically calculating dcm')
    print(symbolic_dcm.subs({x1: 10, x2: 20, x3: 30}).evalf())

