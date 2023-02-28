# Code for
# Proofs of network quantum nonlocality in continuous families of distributions
# Phys. Rev. Lett. 130, 090201 (2023)
# arXiv:2203.16543
#
# Authors: Alejandro Pozas-Kerstjens
#
# Requires: itertools for cartesian products
#           json for data export
#           mosek for LP solving
#           (optional) gurobi for LP solving
#           numpy for array operations
#           sympy for symbolic operations
#           wolframclient for interaction with Mathematica
#
# Last modified: Mar, 2022

import json
import mosek
import numpy as np
import sympy as sp

from itertools import product
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

class TriangleInflationLevel2(object):
    '''Object containing the second level of the inflation hierarchy described
    in [DOI:10.1515/jci-2018-0008] for the triangle scenario.
    '''
    def __init__(self, n_out):
        self.o = n_out
        self.positivity    = (n_out**3)**4
        self.normalization = 1
        # Only works for o=2, so far
        self.inflation     = (n_out**3)*(n_out**9-3*n_out**5 + n_out**3 - 1)
        self.hierarchy     = (n_out**3)**2
        self.higher_order  = n_out**3
        self.lpi           = 3*(n_out**3)**2
        self.n_vars        = (n_out**3)**4
        self.n_constraints = self.positivity + 2*(self.normalization
                                + self.inflation + self.hierarchy
                                + self.higher_order + self.lpi)
        self.LPA       = np.zeros((self.n_constraints, self.n_vars))
        self.LPb       = np.zeros((self.n_constraints))
        self.symbolicA = None
        self.symbolicb = None

    def create_basic_inflation_LP(self):
        '''Generates the basic inflation linear program. The coefficient matrix
        and the values vector encode the constraints of all probabilities being
        positive and summing to 1, and the symmetries derived from the
        invariance under permutation of copies of the sources. This is, it
        implements the constraints described in Eqs. (C1), (C2) and (C3) in
        arXiv:2203.16543
        '''
        # Positivity constraints
        np.fill_diagonal(self.LPA, 1)
        i = self.n_vars

        # Normalisation constraints
        self.LPA[i,:]   = 1
        self.LPb[i]     = 1
        self.LPA[i+1,:] = -1
        self.LPb[i+1]   = -1
        i += 2

        # Inflation symmetries
        used_indices = set()
        for original_idx in product(range(self.o), repeat=12):
            a11,a12,a21,a22,b11,b12,b21,b22,c11,c12,c21,c22 = original_idx
            # Original probability
            index1 = self._to_indices(original_idx)
            used_indices.add(index1)
            # Swapping the A-B sources
            index2 = self._to_indices(
                              [a12,a11,a22,a21,b21,b22,b11,b12,c11,c12,c21,c22])
            if (index1 != index2) and (index2 not in used_indices):
                used_indices.add(index2)
                self.LPA[i, index1]   = 1
                self.LPA[i, index2]   = -1
                self.LPA[i+1, index1] = -1
                self.LPA[i+1, index2] = 1
                i += 2
            # Swapping the B-C sources
            index2 = self._to_indices(
                              [a11,a12,a21,a22,b12,b11,b22,b21,c21,c22,c11,c12])
            if (index1 != index2) and (index2 not in used_indices):
                used_indices.add(index2)
                self.LPA[i, index1]   = 1
                self.LPA[i, index2]   = -1
                self.LPA[i+1, index1] = -1
                self.LPA[i+1, index2] = 1
                i += 2
            # Swapping the A-C sources
            index2 = self._to_indices(
                              [a21,a22,a11,a12,b11,b12,b21,b22,c12,c11,c22,c21])
            if (index1 != index2) and (index2 not in used_indices):
                used_indices.add(index2)
                self.LPA[i, index1]   = 1
                self.LPA[i, index2]   = -1
                self.LPA[i+1, index1] = -1
                self.LPA[i+1, index2] = 1
                i += 2

    def add_hierarchy_constraints(self, distribution, create_symbolicb=False):
        '''For an original probability distribution to assess compatibility, add
        to the coefficient matrix and the values vector the constraints that
        relate the probability distribution in the inflation to the original,
        that are at most polynomials of the same order of the inflation. This
        is, it adds the "hierarchy constraints" described by Eq. (C4) in
        arXiv:2203.16543. Optionally it generates an abstract values
        vector, which is useful for deriving nonlocality witnesses at a later
        stage.

        :param distribution: Probability distribution under test.
        :type distribution: numpy.array
        :param create_symbolicb: (Optional) generate generic values vector.
        :type create_symbolicb: bool
        '''
        i = self.positivity + 2*(self.normalization + self.inflation)
        if create_symbolicb:
            self.symbolicb = self.LPb.astype(object)
            self.symbolicp = np.zeros((self.o, self.o, self.o), dtype=object)
            for a, b, c in product(range(self.o), repeat=3):
                self.symbolicp[a,b,c] = sp.symbols(f'p[{a}_{b}_{c}]')

        for a11, a22, b11, b22, c11, c22 in product(range(self.o), repeat=6):
            indices = [self._to_indices(
                              [a11,a12,a21,a22,b11,b12,b21,b22,c11,c12,c21,c22])
                       for a12,a21,b12,b21,c12,c21 in product(range(self.o),
                                                              repeat=6)]
            self.LPA[i, indices] = 1
            self.LPb[i] = distribution[a11,b11,c11]*distribution[a22,b22,c22]
            self.LPA[i+1, indices] = -1
            self.LPb[i+1] = -distribution[a11,b11,c11]*distribution[a22,b22,c22]
            if self.symbolicb is not None:
                self.symbolicb[i]   = (self.symbolicp[a11,b11,c11]
                                       *self.symbolicp[a22,b22,c22])
                self.symbolicb[i+1] = -(self.symbolicb[i])
            i += 2


    def add_higher_order_constraints(self, distribution):
        '''For an original probability distribution to assess compatibility, add
        to the coefficient matrix and the values vector the constraints that
        relate the probability distribution in the inflation to the original,
        that involve polynomials of order higher than the inflation level.
        This is, it adds the "higher-order constraints" described in Eq. (C5) in
        arXiv:2203.16543.

        :param distribution: Probability distribution under test.
        :type distribution: numpy.array
        '''
        i = self.positivity + 2*(self.normalization + self.inflation
               + self.hierarchy)
        for a12, b12, c12 in product(range(self.o), repeat=3):
            indices = [self._to_indices(
                              [a11,a12,a21,a22,b11,b12,b21,b22,c11,c12,c21,c22])
                       for a11,a21,a22,b11,b21,b22,c11,c21,c22 in product(
                                                                  range(self.o),
                                                                  repeat=9)]
            self.LPA[i, indices] = 1
            self.LPb[i] = (sum(sum(distribution[a12,:,:]))
                           *sum(sum(distribution[:,b12,:]))
                           *sum(sum(distribution[:,:,c12])))
            self.LPA[i+1, indices] = -1
            self.LPb[i+1] = -self.LPb[i]
            if self.symbolicb is not None:
                self.symbolicb[i] = (sum(sum(self.symbolicp[a12,:,:]))
                                     *sum(sum(self.symbolicp[:,b12,:]))
                                     *sum(sum(self.symbolicp[:,:,c12])))
                self.symbolicb[i+1] = -self.symbolicb[i]
            i += 2


    def add_lpi_constraints(self, distribution, create_symbolicA=False):
        '''For an original probability distribution to assess compatibility, add
        to the coefficient matrix and the values vector the constraints that
        relate the probability distribution in the inflation to the original,
        that are at most polynomials of the same order of the inflation. This
        is, it adds the "hierarchy constraints" described in Eq. (C6) in
        arXiv:2203.16543. Optionally it generates an abstract coefficient
        matrix, which is useful for deriving nonlocality witnesses at a later
        stage.

        :param distribution: Probability distribution under test.
        :type distribution: numpy.array
        :param create_symbolicA: (Optional) generate generic coefficient matrix.
        :type create_symbolicA: bool
        '''
        i = self.positivity + 2*(self.normalization + self.inflation
               + self.hierarchy + self.higher_order)
        if create_symbolicA:
            self.symbolicA = self.LPA.astype(object)
        for a11, a22, b11, b12, c11, c21 in product(range(self.o), repeat=6):
            witha22 = []; withb22 = []; withc22 = [];
            withouta22 = []; withoutb22 = []; withoutc22 = [];
            for a12,a21,b21,b22,c12,c22 in product(range(self.o), repeat=6):
                witha22.append(self._to_indices(
                             [a11,a12,a21,a22,b11,b12,b21,b22,c11,c12,c21,c22]))
                withb22.append(self._to_indices(
                             [c11,c12,c21,c22,a11,a12,a21,a22,b11,b12,b21,b22]))
                withc22.append(self._to_indices(
                             [b11,b12,b21,b22,c11,c12,c21,c22,a11,a12,a21,a22]))
                for a22p in range(self.o):
                    withouta22.append(self._to_indices(
                            [a11,a12,a21,a22p,b11,b12,b21,b22,c11,c12,c21,c22]))
                    withoutb22.append(self._to_indices(
                            [c11,c12,c21,c22,a11,a12,a21,a22p,b11,b12,b21,b22]))
                    withoutc22.append(self._to_indices(
                            [b11,b12,b21,b22,c11,c12,c21,c22,a11,a12,a21,a22p]))

            # Now we begin filling
            self.LPA[i, witha22] = 1
            self.LPA[i, withouta22] -= sum(sum(distribution[a22,:,:]))
            self.LPA[i+1, witha22] = -1
            self.LPA[i+1, withouta22] += sum(sum(distribution[a22,:,:]))
            self.LPA[i+2, withb22] = 1
            self.LPA[i+2, withoutb22] -= sum(sum(distribution[:,a22,:]))
            self.LPA[i+3, withb22] = -1
            self.LPA[i+3, withoutb22] += sum(sum(distribution[:,a22,:]))
            self.LPA[i+4, withc22] = 1
            self.LPA[i+4, withoutc22] -= sum(sum(distribution[:,:,a22]))
            self.LPA[i+5, withc22] = -1
            self.LPA[i+5, withoutc22] += sum(sum(distribution[:,:,a22]))
            if create_symbolicA:
                self.symbolicA[i, witha22] = 1
                self.symbolicA[i, withouta22] -= sum(sum(
                                                       self.symbolicp[a22,:,:]))
                self.symbolicA[i+1, witha22] = -1
                self.symbolicA[i+1, withouta22] += sum(sum(
                                                       self.symbolicp[a22,:,:]))
                self.symbolicA[i+2, withb22] = 1
                self.symbolicA[i+2, withoutb22] -= sum(sum(
                                                       self.symbolicp[:,a22,:]))
                self.symbolicA[i+3, withb22] = -1
                self.symbolicA[i+3, withoutb22] += sum(sum(
                                                       self.symbolicp[:,a22,:]))
                self.symbolicA[i+4, withc22] = 1
                self.symbolicA[i+4, withoutc22] -= sum(sum(
                                                       self.symbolicp[:,:,a22]))
                self.symbolicA[i+5, withc22] = -1
                self.symbolicA[i+5, withoutc22] += sum(sum(
                                                       self.symbolicp[:,:,a22]))
            i += 6


    def _to_indices(self, position):
        '''
        Returns a linear index for each input-output configuration.

        :param position: List of inputs and outputs.
        :type position: list of ints

        :returns: linear index of the configuration
        '''
        return sum([coeff * self.o**pow
                               for pow, coeff in enumerate(reversed(position))])


def RGB4_two_outcomes(u, FAB, FAC, FBC, FABC, t, dtype='float64'):
    '''
    The family of binary-outcome probability distributions q_u^{t}(i,j,k)
    described in Eq. (3) of arXiv:2203.16543.

    :param u: The value of the parameter u, inherited from the four-outcome
              family of distributions in Eq. (2) in arXiv:2203.16543.
    :type u: float (1/sqrt(2) <= u <= 1)
    :param FAB: The value of the free parameter F_AB.
    :type FAB: float
    :param FAC: The value of the free parameter F_AC.
    :type FAC: float
    :param FBC: The value of the free parameter F_BC.
    :type FBC: float
    :param FABC: The value of the free parameter F_ABC.
    :type FABC: float
    :param t: The value of the parameter t, which describes the orientation of
              the three excitations in the local model for the coarse-grained
              four-outcome distribution.
    :type t: float (t = +1, -1)
    :param dtype: Type of expression.
    :type dtype: dtype

    :returns: The array q such that q[i,j,k] is the probability of the outcomes
              (i, j, k)
    '''
    oA = oB = oC = 2
    q = np.zeros((oA,oB,oC), dtype=dtype)
    v = np.sqrt(1-u**2)
    for a in range(oA):
        for b in range(oB):
            for c in range(oC):
                i = (-1)**a
                j = (-1)**b
                k = (-1)**c
                q[a,b,c] = (1 + (i*j + j*k + k*i)*(u**2-v**2)**2
                              + (i + j + k)*t*(u**2-v**2) + 8*i*j*k*u**3*v**3
                              + t*(i*j*FAB + j*k*FBC + k*i*FAC)
                              + i*j*k*t*FABC) / 8
    return q


def generate_polytope(u, FAB, FAC, FBC, FABC):
    '''Provides the inequalities describing the polytope of valid probability
    distributions in Eqs. (A1) of arXiv:2203.16543, for the given value of u.

    :param u: The value of the parameter u, inherited from the four-outcome
              family of distributions in Eq. (2) in arXiv:2203.16543.
    :type u: float (1/sqrt(2) <= u <= 1)
    :param FAB: Symbol representing the free parameter F_AB.
    :type FAB: sympy.Symbol
    :param FAC: Symbol representing the free parameter F_AC.
    :type FAC: sympy.Symbol
    :param FBC: Symbol representing the free parameter F_BC.
    :type FBC: sympy.Symbol
    :param FABC: Symbol representing the free parameter F_ABC.
    :type FABC: sympy.Symbol

    :returns: List[sympy.Expression] with the inequalities
    '''
    v = np.sqrt(1-u**2)
    return [-2-FAB-FAC-FBC+FABC+6*u**2+3*(-1+2*u**2)**2-8*(u**3*v-u**5*v) >= 0,
            4+FAB+FAC+FBC-FABC-6*u**2+3*(-1+2*u**2)**2-8*(u**3*v-u**5*v) >= 0,
            -FAB+FAC+FBC-FABC+2*u**2-(-1+2*u**2)**2+8*(u**3*v-u**5*v) >= 0,
            2+FAB-FAC-FBC+FABC-2*u**2-(-1+2*u**2)**2+8*(u**3*v-u**5*v) >= 0,
            FAB-FAC+FBC-FABC+2*u**2-(-1+2*u**2)**2+8*(u**3*v-u**5*v) >= 0,
            2-FAB+FAC-FBC+FABC-2*u**2-(-1+2*u**2)**2+8*(u**3*v-u**5*v) >= 0,
            2+FAB+FAC-FBC+FABC-2*u**2-(-1+2*u**2)**2-8*(u**3*v-u**5*v) >= 0,
            -FAB-FAC+FBC-FABC+2*u**2-(-1+2*u**2)**2-8*(u**3*v-u**5*v) >= 0,
            FAB+FAC-FBC-FABC+2*u**2-(-1+2*u**2)**2+8*(u**3*v-u**5*v) >= 0,
            2-FAB-FAC+FBC+FABC-2*u**2-(-1+2*u**2)**2+8*(u**3*v-u**5*v) >= 0,
            2+FAB-FAC+FBC+FABC-2*u**2-(-1+2*u**2)**2-8*(u**3*v-u**5*v) >= 0,
            -FAB+FAC-FBC-FABC+2*u**2-(-1+2*u**2)**2-8*(u**3*v-u**5*v) >= 0,
            2-FAB+FAC+FBC+FABC-2*u**2-(-1+2*u**2)**2-8*(u**3*v-u**5*v) >= 0,
            FAB-FAC-FBC-FABC+2*u**2-(-1+2*u**2)**2-8*(u**3*v-u**5*v) >= 0,
            4-FAB-FAC-FBC-FABC-6*u**2+3*(-1+2*u**2)**2+8*(u**3*v-u**5*v) >= 0,
            -2+FAB+FAC+FBC+FABC+6*u**2+3*(-1+2*u**2)**2+8*(u**3*v-u**5*v) >= 0]


def set_lp_mosek(A, b, solve=False):
    '''
    Sets and (optionally) solves the linear program A.x >= b in MOSEK.

    :param A: Matrix of the coefficients of each variable in every constraint.
    :type A:  2-D array
    :param b: Vector of lower bounds for each constraint.
    :type b:  1-D array
    :param solve: (Optionally) solve the linear program.
    :type solve:  Bool

    :returns: If solve=False, the MOSEK problem.
              If solve=True and the LP is feasible, an array with the optimal
              solution and None.
              If solve=True and the LP is infeasible, an array with None and a
              certificate of infeasibility.
    '''
    # Make mosek environment
    task = mosek.Env().Task(0, 0)

    n_constraints, n_vars = A.shape

    # Generate the placeholder for the constraints.
    # The constraints will initially have no bounds.
    task.appendcons(n_constraints)

    # Generate the variables in the problem.
    # The variables will initially be fixed at zero (x=0).
    task.appendvars(n_vars)

    for j in range(n_vars):
        # Set the bounds on variable j. We allow them to be free variables
        task.putvarbound(j, mosek.boundkey.fr, 0, 0)

    for i in range(n_constraints):
        # Set the constraints.
        # Select the variables that are involved and the associated coefficients
        indices = np.nonzero(A[i,:])[0]

        task.putarow(i,                  # Variable (column) index.
                     indices,            # Row index of non-zeros in column j.
                     A[i,indices])       # Non-zero Values of column j.

        # Set lower bounds
        task.putconbound(i, mosek.boundkey.lo, b[i], b[i])

    if solve:
        # Solve the problem
        task.optimize()

        # Print solution and save it to file
        # task.solutionsummary(mosek.streamtype.msg)
        # task.writedata("data.lp")

        # Get status information about the solution
        solsta = task.getsolsta(mosek.soltype.bas)

        if (solsta == mosek.solsta.optimal):
            xx = [0.] * n_vars
            task.getxx(mosek.soltype.bas, xx)    # Request the basic solution
            # print('Optimal solution: ')
            # for i in range(n_vars):
            #     print('x[' + str(i) + ']=' + str(xx[i]))
            return xx, None
        elif (solsta == mosek.solsta.dual_infeas_cer or
              solsta == mosek.solsta.prim_infeas_cer):
            y = [0.] * n_constraints
            task.gety(mosek.soltype.bas, y)    # Request the certificate
            # print('Primal or dual infeasibility certificate found')
            return None, y
        elif solsta == mosek.solsta.unknown:
            print('Unknown solution status')
        else:
            print('Other solution status')
    else:
        return task


def set_lp_gurobi(A, b, solve=False):
    '''
    Sets and (optionally) solves the linear program A.x >= b in Gurobi

    :param A: Matrix of the coefficients of each variable in every constraint.
    :type A:  2-D array
    :param b: Vector of lower bounds for each constraint.
    :type b:  1-D array
    :param solve: (Optionally) solve the linear program.
    :type solve:  Bool

    :returns: If solve=False, the Gurobi problem.
              If solve=True and the LP is feasible, an array with the optimal
              solution and None.
              If solve=True and the LP is infeasible, an array with None and a
              certificate of infeasibility.
    '''
    import gurobipy as gp

    # Make environment
    with gp.Env(empty=True) as env:
        env.start()
        with gp.Model(env=env) as task:
            task.setParam("InfUnbdInfo", 1)

            n_constraints, n_vars = A.shape

            # Generate the variables in the problem.
            # The variables will initially be fixed at zero (x=0).
            pinf = task.addMVar(n_vars, lb=-float('inf'), name='pinf')
            constrs = []
            for i in range(n_constraints):
                # Set the constraints.
                # Select the variables that are involved and the associated
                # coefficients
                indices = np.nonzero(A[i,:])[0]
                constrs.append(task.addConstr(
                                          pinf[indices] @ A[i,indices] >= b[i]))

            if solve:
                # Solve the problem
                task.optimize()
                solsta = task.status

                if (solsta == gp.GRB.status.OPTIMAL):
                    return task.getVars(), None
                elif (solsta == gp.GRB.status.INFEASIBLE):
                    y = -np.array([cons.getAttr('FarkasDual').item()
                                   for cons in constrs])
                    return None, y
                else:
                    print('Other solution status')
            else:
                return task


def save_to_json(expression, expression_evaluation, filename):
    '''Save the expression of a witness to JSON format.

    :param expression: The witness.
    :type expression: sympy.Expression
    :param expression_evaluation: The evaluation of the expression in the
                                  distribution witnessed not to admit a
                                  triangle-local model.
    :type expression_evaluation: float
    :param filename: The name of the file to be saved.
    :type filename: str
    '''
    expression_dict = {'value': expression_evaluation}
    for summand in expression.args:
        components = summand.as_coeff_Mul()
        if abs(components[0]) > 1e-10:
            expression_dict[str(components[1])] = str(components[0])
    with open(filename, 'w') as file:
        json.dump(expression_dict, file)


def evaluate_symbolic_prob(symbols, prob_dist):
    '''Substitutes the symbols describing a probability distribution by the
    values corresponding to a particular distribution.

    :param symbols: List of symbols describing the distribution
    :type symbols: List[sympy.Symbol]
    :param prob_dist: Elements of the probability distribution
    :type prob_dist: numpy.array

    :returns prob_list: The list of evaluated probabilities
    '''
    probs_list = []
    for symb in symbols:
        outcomes = list(map(int, str(symb)[2:-1].split('_')))
        probs_list.append(prob_dist[outcomes[0], outcomes[1], outcomes[2]])
    return probs_list

def ineq_intersects_faces(ineq, u, t):
    '''Wrapper for Mathematica to compute whether a certificate intersects the
    polytope of parameter values that produce valid two-outcome probability
    distributions.

    :param ineq: The certificate of infeasibility obtained from Frakas' lemma
    :type ineq: sympy.core.expr.Expr
    :param u: The value of the parameter u, inherited from the four-outcome
              family of distributions in Eq. (2) in arXiv:2203.16543.
    :type u: float (1/sqrt(2) <= u <= 1)
    :param t: The value of the parameter t, which describes the orientation of
              the three excitations in the local model for the coarse-grained
              four-outcome distribution.
    :type t: float (t = +1, -1)

    :returns intersections_exist: Whether the inequality intersects the polytope
    :type intersections_exist: bool
    '''
    sess = WolframLanguageSession()
    # First, write the inequality in terms of FAB, FAC, FBC, FABC
    FAB, FAC, FBC, FABC  = sp.symbols('FAB,FAC,FBC,FABC', commutative=True)
    symbolic_distribution = RGB4_two_outcomes(u, FAB, FAC, FBC, FABC, t,
                                              'object')
    probs_list = evaluate_symbolic_prob(ineq.free_symbols,
                                        symbolic_distribution)
    ineq_F     = sp.expand(ineq.subs(zip(ineq.free_symbols, probs_list)))

    # Define instructions to be sent to Mathematica
    turn_off_warnings = 'Off[Reduce::ratnz]; Off[Solve::svars];'
    define_q = 'q[i_, j_, k_, t_] := (1 + FAB i j t + FAC i k t + FBC j k t'   \
               + '+ FABC i j k t + t (i + j + k) (-1 + 2 u^2)'                 \
               + '+ (i j + i k + j k) (-1 + 2 u^2)^2'                          \
               + '+ 8 i j k u^3 (1-u^2)^(3/2)) / 16'
    define_polytope_faces = 'ineqList = 16 Flatten[Table[q[i, j, k, t],'       \
               + '{i, -1, 1, 2}, {j, -1, 1, 2}, {k, -1, 1, 2}, {t, -1, 1, 2}]];'
    define_intersections = \
        'subsets = Subsets[ineqList /. u->' + str(u) + ', {4}];'               \
        'allIntersections = Table[Solve['                                      \
           +'ineqs[[1]]==0 && ineqs[[2]]==0 && ineqs[[3]]==0 && ineqs[[4]]==0,'\
           +'{FAB, FAC, FBC, FABC}], {ineqs, subsets}];'
    compute_potential_vertices = \
        'nonEmpty = Complement[Range[Length[allIntersections]], '              \
                             +'Flatten[Position[allIntersections, {}]]];'      \
        'vertices = Flatten[allIntersections[[nonEmpty]], 1];'                 \
        'vertexSubsets = subsets[[nonEmpty]];'                                 \
        'nonDegenerate = Flatten[Position[vertices, _?(Length@# > 3 &), 1]];'  \
        'vertices = vertices[[nonDegenerate]];'                                \
        'vertexSubsets = vertexSubsets[[nonDegenerate]];'
    delete_duplicates = \
        'verticesAndPositions=Table[{case[[1]], First[case[[2]]]},'            \
                                 + '{case, Normal[PositionIndex[vertices]]}]'  \
                                 + f'/.u->{u};'                                \
        f'vertices = verticesAndPositions[[All, 1]];'                          \
        f'vertexSubsets=vertexSubsets[[verticesAndPositions[[All,2]]]];'
    get_actual_vertices = \
        f'isActualVertex=Table[AllTrue[(ineqList/.vtx)/.u->{u},#>-10^-15&],'   \
                              +'{vtx, vertices}];'                             \
        +'realVertexPositions = Flatten[Position[isActualVertex,True]];'       \
        +'realVertices = vertices[[realVertexPositions]];'                     \
        +'vertexSubsets = vertexSubsets[[realVertexPositions]];'               \
        +'definingFaces = DeleteDuplicates[Flatten[vertexSubsets]];'
    vertices_in_each_face =                                                    \
        'verticesInTheFacePositions = Table[Flatten[Position['                 \
            +'(face /. #) & /@ realVertices, _?(Abs[#] < 10^-15 &)]]'          \
                +', {face, definingFaces}];'                                   \
        +'facesDefiningTheIntersections = Table[DeleteCases[DeleteDuplicates[' \
            +'Flatten[vertexSubsets[[verticesInTheFacePositions[[i]]]]]],'     \
            +'definingFaces[[i]]], {i, 1, Length[definingFaces]}];'
    define_ineq = 'ineq=' + str(ineq_F)                                        \
                    .replace('**','^').replace('*',' ').replace('e','`*^') + ';'
    # print(ineq_F)
    # print(define_ineq)
    face_and_ineq_intersect = \
        'isIntersection = Table[Reduce['                                       \
            +'(ineq == 0. && definingFaces[[i]] == 0. '                        \
            +'&& Apply[And, (#>=0.)& /@ facesDefiningTheIntersections[[i]]])]' \
            +', {i, 1, Length[definingFaces]}];'
    check_any_intersection = \
        'nOfIntersections=Length[isIntersection] - Count[isIntersection,False]'
    for instruction in [turn_off_warnings, define_q, define_polytope_faces,
                        define_intersections, compute_potential_vertices,
                        delete_duplicates, get_actual_vertices,
                        vertices_in_each_face, define_ineq,
                        face_and_ineq_intersect, check_any_intersection]:
        sess.evaluate(wlexpr(instruction))
    intersections_exist = sess.evaluate(wlexpr('nOfIntersections')) != 0
    sess.terminate()
    return intersections_exist
