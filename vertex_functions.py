# Code for
# Proofs of network quantum nonlocality aided by machine learning
# arXiv:2203.16543
#
# Authors: Alejandro Pozas-Kerstjens
#
# Requires: math for square roots
# Last modified: Mar, 2022

from math import sqrt

def get_polytope_vertices(u):
    '''For a value of u, obtain the vertices of the feasible poltope.

    :param u: The value of the parameter u, inherited from the four-outcome
              family of distributions in Eq. (2) in arXiv:2203.16543.
    :type u: float (1/sqrt(2) <= u <= 1)

    :returns funs: list of vertex functions.
    '''
    assert u >= 1/sqrt(2), 'u must be larger than 1/sqrt(2)~0.70711'
    assert u <= 0.886, 'u must be lower than 0.88600'
    if u < 0.8096:
        funs = [vtx_01, vtx_02, vtx_04, vtx_05, vtx_09, vtx_10, vtx_12, vtx_13,
                vtx_21, vtx_22, vtx_24, vtx_25, vtx_29, vtx_30, vtx_32, vtx_33]
    elif u < 0.8117:
        funs = [vtx_01, vtx_02, vtx_04, vtx_05, vtx_09, vtx_10, vtx_12, vtx_14,
                vtx_16, vtx_20, vtx_21, vtx_22, vtx_24, vtx_25, vtx_29, vtx_30,
                vtx_32, vtx_33, vtx_48]
    elif u < 0.814:
        funs = [vtx_01, vtx_02, vtx_04, vtx_06, vtx_08, vtx_09, vtx_11, vtx_15,
                vtx_18, vtx_19, vtx_21, vtx_22, vtx_24, vtx_25, vtx_29, vtx_30,
                vtx_32, vtx_33, vtx_44, vtx_46, vtx_47, vtx_48]
    elif u < 0.8165:
        funs = [vtx_01, vtx_03, vtx_07, vtx_17, vtx_21, vtx_22, vtx_24, vtx_25,
                vtx_29, vtx_30, vtx_32, vtx_33, vtx_42, vtx_43, vtx_44, vtx_45,
                vtx_46, vtx_47, vtx_48]
    elif u < 0.8457:
        funs = [vtx_21, vtx_22, vtx_24, vtx_25, vtx_29, vtx_30, vtx_32, vtx_33,
                vtx_41, vtx_42, vtx_43, vtx_44, vtx_45, vtx_46, vtx_47, vtx_48]
    elif u < 0.8542:
        funs = [vtx_21, vtx_22, vtx_24, vtx_25, vtx_29, vtx_30, vtx_32, vtx_34,
                vtx_36, vtx_40, vtx_41, vtx_42, vtx_43, vtx_44, vtx_45, vtx_46,
                vtx_47]
    elif u < sqrt(3)/2:
        funs = [vtx_21, vtx_22, vtx_24, vtx_26, vtx_28, vtx_29, vtx_31, vtx_35,
                vtx_38, vtx_39, vtx_41, vtx_42, vtx_43, vtx_45]
    else:
        funs = [vtx_21, vtx_23, vtx_27, vtx_37, vtx_41]
    return funs


################################################################################
# DEFINITION OF INDIVIDUAL VERTICES
################################################################################
# Order is FAB, FAC, FBC, FABC
def vtx_01(u):
    return [2*(-u**2 + 2*u**4),
            2*(-u**2 + 2*u**4),
            2*(-u**2 + 2*u**4),
            -1 + 8*u**3*(1 - u**2)**(3/2)]

def vtx_02(u):
    return [-2*(2*u**2 - 3*u**4 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1-u**2)),
            2*(u**4 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            2*(u**4 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            -1 + 2*u**2 - 2*u**4 + 4*u**3*sqrt(1 - u**2) - 4*u**5*sqrt(1-u**2)]

def vtx_03(u):
    return [4*(-1 + u**2 + u**4),
            4*(1 - 2*u**2 + u**4),
            4*(1 - 2*u**2 + u**4),
            3 - 6*u**2 + 8*u**3*sqrt(1 - u**2) - 8*u**5*sqrt(1 - u**2)]

def vtx_04(u):
    return [2*(u**4 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            -2*(2*u**2 - 3*u**4 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1-u**2)),
            2*(u**4 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            -1 + 2*u**2 - 2*u**4 + 4*u**3*sqrt(1 - u**2) - 4*u**5*sqrt(1-u**2)]

def vtx_05(u):
    return [2*(-u**2 + 2*u**4),
            2*(-u**2 + 2*u**4),
            2*(u**2 - 4*u**3*sqrt(1 - u**2) + 4*u**5*sqrt(1 - u**2)),
            -1 + 4*u**2 - 4*u**4]

def vtx_06(u):
    return [4*(-1 + 2*u**2 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            -4*(-1 + 3*u**2 - 2*u**4 - 2*u**3*sqrt(1-u**2)+2*u**5*sqrt(1-u**2)),
            4*(1 - 2*u**2 + u**4),
            3 - 6*u**2 + 8*u**3*sqrt(1 - u**2) - 8*u**5*sqrt(1 - u**2)]

def vtx_07(u):
    return [4*(1 - 2*u**2 + u**4),
            4*(-1 + u**2 + u**4),
            4*(1 - 2*u**2 + u**4),
            3 - 6*u**2 + 8*u**3*sqrt(1 - u**2) - 8*u**5*sqrt(1 - u**2)]

def vtx_08(u):
    return [-4*(-1 + 3*u**2 - 2*u**4 - 2*u**3*sqrt(1-u**2)+2*u**5*sqrt(1-u**2)),
            4*(-1 + 2*u**2 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            4*(1 - 2*u**2 + u**4),
            3 - 6*u**2 + 8*u**3*sqrt(1 - u**2) - 8*u**5*sqrt(1 - u**2)]

def vtx_09(u):
    return [2*(u**4 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            2*(u**4 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            -2*(2*u**2 - 3*u**4 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1-u**2)),
            -1 + 2*u**2 - 2*u**4 + 4*u**3*sqrt(1 - u**2) - 4*u**5*sqrt(1-u**2)]

def vtx_10(u):
    return [2*(-u**2 + 2*u**4),
            2*(u**2 - 4*u**3*sqrt(1 - u**2) + 4*u**5*sqrt(1 - u**2)),
            2*(-u**2 + 2*u**4),
            -1 + 4*u**2 - 4*u**4]

def vtx_11(u):
    return [4*(-1 + 2*u**2 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            4*(1 - 2*u**2 + u**4),
            -4*(-1 + 3*u**2 - 2*u**4 - 2*u**3*sqrt(1-u**2)+2*u**5*sqrt(1-u**2)),
            3 - 6*u**2 + 8*u**3*sqrt(1 - u**2) - 8*u**5*sqrt(1 - u**2)]

def vtx_12(u):
    return [2*(u**2 - 4*u**3*sqrt(1 - u**2) + 4*u**5*sqrt(1 - u**2)),
            2*(-u**2 + 2*u**4),
            2*(-u**2 + 2*u**4),
            -1 + 4*u**2 - 4*u**4]

def vtx_13(u):
    return [2*(u**4 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            2*(u**4 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            2*(u**4 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            -1 + 6*u**2 - 6*u**4 - 4*u**3*sqrt(1 - u**2) + 4*u**5*sqrt(1-u**2)]

def vtx_14(u):
    return [4*(-1 + 3*u**2 - u**4 - 4*u**3*sqrt(1-u**2) + 4*u**5*sqrt(1-u**2)),
            -4*(-1 + 3*u**2 - 2*u**4 - 2*u**3*sqrt(1-u**2)+2*u**5*sqrt(1-u**2)),
            -4*(-1 + 3*u**2 - 2*u**4 - 2*u**3*sqrt(1-u**2)+2*u**5*sqrt(1-u**2)),
            3 - 6*u**2 + 8*u**3*sqrt(1 - u**2) - 8*u**5*sqrt(1 - u**2)]

def vtx_15(u):
    return [4*(1 - 2*u**2 + u**4),
            4*(-1 + 2*u**2 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            -4*(-1 + 3*u**2 - 2*u**4 - 2*u**3*sqrt(1-u**2)+2*u**5*sqrt(1-u**2)),
            3 - 6*u**2 + 8*u**3*sqrt(1 - u**2) - 8*u**5*sqrt(1 - u**2)]

def vtx_16(u):
    return [-4*(-1 + 3*u**2 - 2*u**4 - 2*u**3*sqrt(1-u**2)+2*u**5*sqrt(1-u**2)),
            4*(-1 + 3*u**2 - u**4 - 4*u**3*sqrt(1-u**2) + 4*u**5*sqrt(1-u**2)),
            -4*(-1 + 3*u**2 - 2*u**4 - 2*u**3*sqrt(1-u**2)+2*u**5*sqrt(1-u**2)),
            3 - 6*u**2 + 8*u**3*sqrt(1 - u**2) - 8*u**5*sqrt(1 - u**2)]

def vtx_17(u):
    return [4*(1 - 2*u**2 + u**4),
            4*(1 - 2*u**2 + u**4),
            4*(-1 + u**2 + u**4),
            3 - 6*u**2 + 8*u**3*sqrt(1 - u**2) - 8*u**5*sqrt(1 - u**2)]

def vtx_18(u):
    return [-4*(-1 + 3*u**2 - 2*u**4 - 2*u**3*sqrt(1-u**2)+2*u**5*sqrt(1-u**2)),
            4*(1 - 2*u**2 + u**4),
            4*(-1 + 2*u**2 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            3 - 6*u**2 + 8*u**3*sqrt(1 - u**2) - 8*u**5*sqrt(1 - u**2)]

def vtx_19(u):
    return [4*(1 - 2*u**2 + u**4),
            -4*(-1 + 3*u**2 - 2*u**4 - 2*u**3*sqrt(1-u**2)+2*u**5*sqrt(1-u**2)),
            4*(-1 + 2*u**2 - 2*u**3*sqrt(1 - u**2) + 2*u**5*sqrt(1 - u**2)),
            3 - 6*u**2 + 8*u**3*sqrt(1 - u**2) - 8*u**5*sqrt(1 - u**2)]

def vtx_20(u):
    return [-4*(-1 + 3*u**2 - 2*u**4 - 2*u**3*sqrt(1-u**2)+2*u**5*sqrt(1-u**2)),
            -4*(-1 + 3*u**2 - 2*u**4 - 2*u**3*sqrt(1-u**2)+2*u**5*sqrt(1-u**2)),
            4*(-1 + 3*u**2 - u**4 - 4*u**3*sqrt(1-u**2) + 4*u**5*sqrt(1-u**2)),
            3 - 6*u**2 + 8*u**3*sqrt(1 - u**2) - 8*u**5*sqrt(1 - u**2)]

def vtx_21(u):
    return [-2*(-1 + u**2)*(-1 + u**2 + 2*u**3*sqrt(1 - u**2)),
            -2*(-1 + u**2)*(-1 + u**2 + 2*u**3*sqrt(1 - u**2)),
            -2*(-1 + u**2)*(-1 + u**2 + 2*u**3*sqrt(1 - u**2)),
            1 - 2*u**2*(-1 + u**2)*(-3 + 2*u*sqrt(1 - u**2))]

def vtx_22(u):
    return [-2*(1 - 2*u**2 + u**4 - 2*u**3*sqrt(1-u**2) + 2*u**5*sqrt(1-u**2)),
            -2*(1 - 2*u**2 + u**4 - 2*u**3*sqrt(1-u**2) + 2*u**5*sqrt(1-u**2)),
            -2*(1 - 2*u**2 + u**4 - 2*u**3*sqrt(1-u**2) + 2*u**5*sqrt(1-u**2)),
            1 - 6*u**2 + 6*u**4 + 4*u**3*sqrt(1 - u**2) - 4*u**5*sqrt(1 - u**2)]

def vtx_23(u):
    return [-8*(-1 + u**2)*(-1 + u**2 + u**3*sqrt(1 - u**2)),
            4*(-1 + u**2)**2,
            4*(-1 + u**2)**2,
            7 - 18*u**2 + 12*u**4]

def vtx_24(u):
    return [-2 + 6*u**2 - 4*u**4,
            -2*(-1 + u**2)*(-1 + 4*u**3*sqrt(1 - u**2)),
            -2 + 6*u**2 - 4*u**4,
            (1 - 2*u**2)**2]

def vtx_25(u):
    return [-2*(-1 + u**2)*(-1 + u**2 + 2*u**3*sqrt(1 - u**2)),
            -2*(-1 + u**2)*(-1 + u**2 + 2*u**3*sqrt(1 - u**2)),
            2*(-1 + u**2)*(1 + u**2*(-3 + 2*u*sqrt(1 - u**2))),
            1 + 2*u**2*(-1 + u**2)*(1 + 2*u*sqrt(1 - u**2))]

def vtx_26(u):
    return [-4*(2 - 5*u**2 + 3*u**4),
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            4*(-1 + u**2)**2,
            7 - 18*u**2 + 12*u**4]

def vtx_27(u):
    return [4*(-1 + u**2)**2,
            -8*(-1 + u**2)*(-1 + u**2 + u**3*sqrt(1 - u**2)),
            4*(-1 + u**2)**2,
            7 - 18*u**2 + 12*u**4]

def vtx_28(u):
    return [-4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            -4*(2 - 5*u**2 + 3*u**4),
            4*(-1 + u**2)**2,
            7 - 18*u**2 + 12*u**4]

def vtx_29(u):
    return [-2 + 6*u**2 - 4*u**4,
            -2 + 6*u**2 - 4*u**4,
            -2*(-1 + u**2)*(-1 + 4*u**3*sqrt(1 - u**2)),
            (1 - 2*u**2)**2]

def vtx_30(u):
    return [-2*(-1 + u**2)*(-1 + u**2 + 2*u**3*sqrt(1 - u**2)),
            2*(-1 + u**2)*(1 + u**2*(-3 + 2*u*sqrt(1 - u**2))),
            -2*(-1 + u**2)*(-1 + u**2 + 2*u**3*sqrt(1 - u**2)),
            1 + 2*u**2*(-1 + u**2)*(1 + 2*u*sqrt(1 - u**2))]

def vtx_31(u):
    return [-4*(2 - 5*u**2 + 3*u**4),
            4*(-1 + u**2)**2,
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            7 - 18*u**2 + 12*u**4]

def vtx_32(u):
    return [2*(-1 + u**2)*(1 + u**2*(-3 + 2*u*sqrt(1 - u**2))),
            -2*(-1 + u**2)*(-1 + u**2 + 2*u**3*sqrt(1 - u**2)),
            -2*(-1 + u**2)*(-1 + u**2 + 2*u**3*sqrt(1 - u**2)),
            1 + 2*u**2*(-1 + u**2)*(1 + 2*u*sqrt(1 - u**2))]

def vtx_33(u):
    return [-2 + 6*u**2 - 4*u**4,
            -2 + 6*u**2 - 4*u**4,
            -2 + 6*u**2 - 4*u**4,
            1 - 8*u**3*(1 - u**2)**(3/2)]

def vtx_34(u):
    return [8*(-1 + u**2)*(1 + u**2*(-2 + u*sqrt(1 - u**2))),
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            7 - 18*u**2 + 12*u**4]

def vtx_35(u):
    return [4*(-1 + u**2)**2,
            -4*(2 - 5*u**2 + 3*u**4),
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            7 - 18*u**2 + 12*u**4]

def vtx_36(u):
    return [-4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            8*(-1 + u**2)*(1 + u**2*(-2 + u*sqrt(1 - u**2))),
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            7 - 18*u**2 + 12*u**4]

def vtx_37(u):
    return [4*(-1 + u**2)**2,
            4*(-1 + u**2)**2,
            -8*(-1 + u**2)*(-1 + u**2 + u**3*sqrt(1 - u**2)),
            7 - 18*u**2 + 12*u**4]

def vtx_38(u):
    return [-4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            4*(-1 + u**2)**2,
            -4*(2 - 5*u**2 + 3*u**4),
            7 - 18*u**2 + 12*u**4]

def vtx_39(u):
    return [4*(-1 + u**2)**2,
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            -4*(2 - 5*u**2 + 3*u**4),
            7 - 18*u**2 + 12*u**4]

def vtx_40(u):
    return [-4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            8*(-1 + u**2)*(1 + u**2*(-2 + u*sqrt(1 - u**2))),
            7 - 18*u**2 + 12*u**4]
def vtx_41(u):
    return [4*(-1 + u**2)**2,
            4*(-1 + u**2)**2,
            4*(-1 + u**2)**2,
            -5 + 6*u**2 + 8*u**3*(1 - u**2)**(3/2)]

def vtx_42(u):
    return [-4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            4*(-1 + u**2)**2,
            4*(-1 + u**2)**2,
            -5 + 10*u**2 - 4*u**4]

def vtx_43(u):
    return [4*(-1 + u**2)**2,
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            4*(-1 + u**2)**2,
            -5 + 10*u**2 - 4*u**4]

def vtx_44(u):
    return [-4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            4*(-1 + u**2)**2,
            -5 + 2*u**2*(7 + 4*u*(-sqrt(1 - u**2) + u*(-1 + u*sqrt(1 - u**2))))]

def vtx_45(u):
    return [4*(-1 + u**2)**2,
            4*(-1 + u**2)**2,
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            -5 + 10*u**2 - 4*u**4]

def vtx_46(u):
    return [-4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            4*(-1 + u**2)**2,
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            -5 + 2*u**2*(7 + 4*u*(-sqrt(1 - u**2) + u*(-1 + u*sqrt(1 - u**2))))]

def vtx_47(u):
    return [4*(-1 + u**2)**2,
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            -5 + 2*u**2*(7 + 4*u*(-sqrt(1 - u**2) + u*(-1 + u*sqrt(1 - u**2))))]

def vtx_48(u):
    return [-4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            -4*(-1 + u**2)*(1 + 2*u**2*(-1 + u*sqrt(1 - u**2))),
            -5 + 2*u**2*(9 - 6*u**2 - 8*u*sqrt(1 - u**2) + 8*u**3*sqrt(1-u**2))]
