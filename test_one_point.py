# Code for
# Proofs of network quantum nonlocality aided by machine learning
# arXiv:2203......
#
# Authors: Alejandro Pozas-Kerstjens
#
# Requires: argparse for argument parsing
#           numpy for array operations
#           sympy for symbolic operations
# Last modified: Mar, 2022

import argparse
import numpy as np
import sympy as sp

from utils import *
from vertex_functions import get_polytope_vertices, vtx_33

################################################################################
# PARSING ARGUMENTS
################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-s', default='mosek', type=str,
                    choices=['mosek', 'gurobi'], help='Specification of solver')
parser.add_argument('-u', type=float, required=True,
                    help='Value of the u parameter')
parser.add_argument('-save', default=False, type=bool,
                    help='Save the expression of the certificate')
args = parser.parse_args()

solver = args.s
u      = args.u
save   = args.save

################################################################################
# COMMON PARAMETERS
################################################################################
n_out = 2
t     = -1

################################################################################
# CREATION OF THE LP
################################################################################
# First, create a general LP for the inflation, containing only positivity and
# normalization constraints, and the symmetries coming from the permutation of
# sources in the inflation
TwoOutcomeTriangleInflation = TriangleInflationLevel2(n_out)
TwoOutcomeTriangleInflation.create_basic_inflation_LP()

# Get the vertices that define the polytope
# Now we begin adding the constraints that relate probabilities in the inflation
# to the probability distribution under test
FAB, FAC, FBC, FABC = vtx_33(u)
orig = RGB4_two_outcomes(u, FAB, FAC, FBC, FABC, t)

# Hierarchy constraints. Those used in [DOI:10.1515/jci-2018-0008]
# We switch on the create_symbolicb flag so we can extract witness expressions
TwoOutcomeTriangleInflation.add_hierarchy_constraints(orig,
                                                      create_symbolicb=True)

# Higher-order identification constraints
TwoOutcomeTriangleInflation.add_higher_order_constraints(orig)

# Linearized polynomial constraints
TwoOutcomeTriangleInflation.add_lpi_constraints(orig)

# Choose solver and solve
solver_fn = set_lp_mosek if solver == 'mosek' else set_lp_gurobi
[xsol, y] = solver_fn(TwoOutcomeTriangleInflation.LPA,
                      TwoOutcomeTriangleInflation.LPb,
                      solve=True)

# If the point is infeasible (i.e., the distribution has no model),
# extract the witness
if xsol is not None:
    print('A feasible point was found')
else:
    y = np.array(y)
    y[abs(y) < 1e-10] = 0

    print('--------------------------------------')
    print(f'The optimal vertex with u={round(u, 4)} '
          + 'does not admit a triangle-local model.')
    print('Farkas\' values:')
    print(f'y.b: {y @ TwoOutcomeTriangleInflation.LPb},'
          + f' max(y.A): {max(y @ TwoOutcomeTriangleInflation.LPA)}')
    ineq_probs = y @ TwoOutcomeTriangleInflation.symbolicb
    print('The inequality that witnesses the infeasibility is')
    print(sp.expand(ineq_probs), '<= 0')
    # Check that the whole polytope is excluded by the inequality. This is done
    # in two steps. First, check that all vertices of the polytope violate the
    # inequality.
    polytope_vertices      = get_polytope_vertices(u)
    vertices_without_model = []
    for other_vtx in polytope_vertices:
        FAB_o, FAC_o, FBC_o, FABC_o = other_vtx(u)
        other_prob = RGB4_two_outcomes(u, FAB_o, FAC_o, FBC_o, FABC_o, t)
        probs_list = evaluate_symbolic_prob(ineq_probs.free_symbols, other_prob)
        vtx_value  = ineq_probs.subs(zip(ineq_probs.free_symbols, probs_list))
        vertices_without_model.append(vtx_value > 0)
    if all(vertices_without_model):
        # Second, check that the inequality does not intersect any of the
        # faces of the polytope within the margins of the polytope
        intersections = ineq_intersects_faces(ineq_probs, u, t)
        if not intersections:
            print('The obtained inequality witnesses all the polytope as not '
                  + 'having a triangle-local model. Thus, the RGB4 distribution'
                  + f' for u={round(u, 4)} is certified not to admit a '
                  + 'triangle-local model.')
        else:
            print('The obtained inequality does not cover the whole polytope.')
    else:
        print('The obtained inequality does not cover the whole polytope.')

    if save:
        # Certificate as the vector y such that y.A = 0 and y.b > 0
        np.savetxt(f'y_{u}.txt', y)

        # Certificate in full probabilities
        expression       = sp.expand(y @ TwoOutcomeTriangleInflation.symbolicb)
        expression_value = y @ TwoOutcomeTriangleInflation.LPb
        save_to_json(expression, expression_value, f'probabilities_u_{u}.json')
