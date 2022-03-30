# Code for
# Proofs of network quantum nonlocality aided by machine learning
# arXiv:2203......
#
# Authors: Alejandro Pozas-Kerstjens
#
# Requires: argparse for argument parsing
#           json for data export
#           os for filesystem operations
#           numpy for array operations
#           scipy for optimization functions
#           sympy for symbolic operations
# Last modified: Mar, 2022

import argparse
import json
import os
import numpy as np
import sympy as sp

from scipy.optimize import brentq
from utils import *
from vertex_functions import vtx_33

################################################################################
# FUNCTION DEFINITIONS
################################################################################
def evaluate_symbolic(LPA, LPb,
                      A_positions, b_positions,
                      A_values, b_values,
                      u, t):
    '''From a symbolic representation of an inflation LP A.x >= b, substitute
    the symbolic expressions by numerical ones for the two-outcome
    probability distribution described by the relevant vertex of the feasible
    polytope.
    :param LPA: The matrix of coefficients of the LP A.x >= b.
    :type LPA: numpy.array
    :param LPb: The vector of coefficients of the LP A.x >= b.
    :type LPb: numpy.array
    :param A_positions: Positions of the symbolic elements in the matrix LPA.
    :type A_positions: numpy.array
    :param b_positions: Positions of the symbolic elements in the vector LPb.
    :type b_positions: numpy.array
    :param A_values: Numerical values of the symbolic elements in the matrix LPA
    :type A_values: numpy.array
    :param b_values: Numerical values of the symbolic elements in the vector LPb
    :type b_values: numpy.array
    :param u: The value of the parameter u, inherited from the four-outcome
              family of distributions in Eq. (2) in arXiv:2203.......
    :type u: float (1/sqrt(2) <= u <= 1)
    :param t: The value of the parameter t, which describes the orientation of
              the three excitations in the local model for the coarse-grained
              four-outcome distribution.
    :type t: float (t = +1, -1)
    :param direction: The direction of the search of u'
    :type direction: str (direction = increase, decrease)

    :returns LPA: The numeric matrix of the LP.
    :returns LPb: The numeric vector of the LP.
    '''
    [FAB, FAC, FBC, FABC] = vtx_33(u)
    distribution = RGB4_two_outcomes(u, FAB, FAC, FBC, FABC, t)
    LPA[A_positions] = [element.subs(zip(symprobs.flatten(),
                                         distribution.flatten()))
                                                        for element in A_values]
    LPb[b_positions] = [element.subs(zip(symprobs.flatten(),
                                         distribution.flatten()))
                                                        for element in b_values]
    if LPA.dtype == object:
        LPA = LPA.astype(float)
    if LPb.dtype == object:
        LPb = LPb.astype(float)
    return LPA, LPb


def solve(A, b, solver):
    '''Wrapper for solving the LP A.x >= b.
    :param A: The matrix of coefficients of the LP.
    :type A: numpy.array
    :param b: The vector of values of the LP.
    :type b: numpy.array
    :param solver: The specification of solver used.
    :type solver: str (solver = mosek, gurobi)

    :returns: If the LP is feasible, a confirmation str.
              Otherwise, a certificate of infeasibility from Farkas' lemma.
    '''
    if solver == 'mosek':
        solver_fn = set_lp_mosek
    elif solver == 'gurobi':
        solver_fn = set_lp_gurobi
    [xsol, y] = solver_fn(A, b, True)
    if xsol is not None:
        return 'Finished'
    else:
        y = np.array(y)
        y[abs(y) < 1e-10] = 0
        return y


def compute_new_u(y, A, b, u, t, direction):
    '''Find the u' furthest from u such that the inequality
    max(y_u.A(u')) >= y_u.b(u') is satisfied.
    :param y: The certificate of infeasibility satisfying max(A.x) < y.b.
    :type y: np.array
    :param A: The matrix of coefficients of the LP A.x >= b.
    :type A: numpy.array
    :param b: The vector of coefficients of the LP A.x >= b.
    :type b: numpy.array
    :param u: The value of the parameter u, inherited from the four-outcome
              family of distributions in Eq. (2) in arXiv:2203.......
    :type u: float (1/sqrt(2) <= u <= 1)
    :param t: The value of the parameter t, which describes the orientation of
              the three excitations in the local model for the coarse-grained
              four-outcome distribution.
    :type t: float (t = +1, -1)
    :param direction: The direction of the search of u'
    :type direction: str (direction = increase, decrease)

    :returns new_u: The value furthest from u fir which the certificate
                    identifies infeasibility.
    '''
    def func(x, numericA, numericb):
        '''The function for which the root must be computed, namely max(y.A)-y.b
        '''
        FAB, FAC, FBC, FABC = vtx_33(x)
        numprobs = RGB4_two_outcomes(x, FAB, FAC, FBC, FABC, t)

        numericA[symbolicA_positions] = [element.subs(zip(symprobs.flatten(),
                                                          numprobs.flatten()))
                                                 for element in symbolicA_exprs]
        numericb[symbolicb_positions] = [element.subs(zip(symprobs.flatten(),
                                                          numprobs.flatten()))
                                                 for element in symbolicb_exprs]

        expression = max(y@numericA) - y@numericb
        return expression

    # Determine boundaries of the root finding
    if direction == 'increase':
        left = u
        right = 0.89
    elif direction == 'decrease':
        left = 1/np.sqrt(2)
        right = u

    new_u = brentq(func, left, right, args=(A, b))
    # Sanity checks that we are moving in the right direction
    if direction == 'increase':
        assert u <= new_u, (f'The new u, {round(new_u, 4)}, is smaller than '
                            + f'the original, {round(u, 4)}')
    elif direction == 'decrease':
        assert u >= new_u, (f'The new u, {round(new_u, 4)}, is larger than '
                            + f'the original, {round(u, 4)}')
    return new_u


def save_to_json(expression, expression_evaluation, u, expression_type):
    '''Save the expression of a witness to JSON format.

    :param expression: The witness.
    :type expression: sympy.Expression
    :param expression_evaluation: The evaluation of the expression in the
                                  distribution witnessed not to admit a
                                  triangle-local model.
    :type expression_evaluation: float
    :param u: The value of the parameter u, inherited from the four-outcome
              family of distributions in Eq. (2) in arXiv:2203.......
    :type u: float (1/sqrt(2) <= u <= 1)
    :param filename: The name of the file to be saved.
    :type filename: str
    '''
    expression_dict = {'value': expression_evaluation}
    for summand in expression.args:
        components = summand.as_coeff_Mul()
        if abs(components[0]) > 1e-10:
            expression_dict[str(components[1])] = str(components[0])
    with open(f'{certificate_folder}/{expression_type}_u_{u}.json','w') as file:
        json.dump(expression_dict, file)


################################################################################
# PARSING ARGUMENTS
################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-s', default='mosek', type=str,
                    choices=['mosek', 'gurobi'], help='Specification of solver')
parser.add_argument('-u', type=float, default=0.8101,
                    help='Value of the u parameter')
parser.add_argument('-d', default='decrease', type=str,
                    choices=['increase', 'decrease'],
                    help='Direction of the sweep')
parser.add_argument('-save', default=False, type=bool,
                    help='Save the expressions of the certificates')
args = parser.parse_args()

solver            = args.s
u                 = args.u
direction         = args.d
save_certificates = args.save

################################################################################
# BEGIN PROGRAM
################################################################################
t     = -1
n_out = 2
certificate_folder = 'certificates'
if save_certificates:
    if not os.path.exists(certificate_folder):
        os.mkdir(certificate_folder)

# We begin by creating one problem
TwoOutcomeTriangleInflation = TriangleInflationLevel2(n_out)
TwoOutcomeTriangleInflation.create_basic_inflation_LP()
FAB, FAC, FBC, FABC = vtx_33(u)
distribution = RGB4_two_outcomes(u, FAB, FAC, FBC, FABC, t)
TwoOutcomeTriangleInflation.add_hierarchy_constraints(distribution,
                                                      create_symbolicb=True)
TwoOutcomeTriangleInflation.add_higher_order_constraints(distribution)
TwoOutcomeTriangleInflation.add_lpi_constraints(distribution,
                                                create_symbolicA=True)

# Extract the symbols and their positions, so we don't need to operate as much
# with large matrices
find_expr = np.vectorize(lambda x: (type(x) != int) and (type(x) != float))

LPA      = TwoOutcomeTriangleInflation.LPA.astype(float)
LPb      = TwoOutcomeTriangleInflation.LPb.astype(float)
symprobs = TwoOutcomeTriangleInflation.symbolicp
symbolicA_positions = np.where(find_expr(TwoOutcomeTriangleInflation.symbolicA))
symbolicA_exprs     = TwoOutcomeTriangleInflation.symbolicA[symbolicA_positions]
symbolicb_positions = np.where(find_expr(TwoOutcomeTriangleInflation.symbolicb))
symbolicb_exprs     = TwoOutcomeTriangleInflation.symbolicb[symbolicb_positions]

counter = 0
while (u > 1/np.sqrt(2)) and (u < 0.89) and (counter < 10000):
    print('--------------------------------------')
    print('Getting certificate for u={}'.format(u))
    A, b = evaluate_symbolic(LPA, LPb, symbolicA_positions,
                             symbolicb_positions, symbolicA_exprs,
                             symbolicb_exprs, u, t)
    y = solve(A, b, solver)
    # Exit if the problem is not infeasible
    if type(y) == str:
        print('A feasible point was found')
        break
    new_u = compute_new_u(y, A, b, u, t, direction)
    print(f'The certificate for u={round(u, 4)} is valid '
          + f'until u={round(new_u, 4)}')
    if save_certificates:
        # Certificate as the vector y such that y.A = 0 and y.b > 0
        np.savetxt(f'{certificate_folder}/y_{u}.txt', y)

        # Certificate in full probabilities
        expression       = sp.expand(y @ TwoOutcomeTriangleInflation.symbolicb)
        expression_value = y @ b
        save_to_json(expression, expression_value, u, 'probabilities')
    u = new_u
    counter += 1
