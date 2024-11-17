from   pyccel.decorators import types
# =========================================================================
def solve_unit_sylvester_system_2d(d1: 'float[:]',
                                   d2: 'float[:]',
                                   b: 'float[:]',
                                   tau: float,
                                   x: 'float[:]'):
    """
    Solves the linear system (D1 x I2 + I1 x D2 + tau I1 x I2) x = b
    """
    n1 = len(d1)
    n2 = len(d2)

    for i1 in range(n1):
        for i2 in range(n2):
            k = i2 + i1*n2
            x[k] = b[k] / (d1[i1] + d2[i2] + tau)

# =========================================================================
def solve_unit_sylvester_system_3d(d1: 'float[:]',
                                   d2: 'float[:]',
                                   d3: 'float[:]',
                                   b: 'float[:]',
                                   tau: float,
                                   x: 'float[:]'):
    """
    Solves the linear system (D1 x I2 x I3 + I1 x D2 x I3 + I1 x I2 x D3 + tau I1 x I2 x I3) x = b
    """
    n1 = len(d1)
    n2 = len(d2)
    n3 = len(d3)

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                k = i3 + i2*n3 + i1*n2*n3
                x[k] = b[k] / (d1[i1] + d2[i2] + d3[i3] + tau)
