## Code to accompany *[Proofs of network quantum nonlocality aided by machine learning](https://www.arxiv.org/abs/2201.xxxxx)*
#### Alejandro Pozas-Kerstjens, Nicolas Gisin, and Marc-Olivier Renou

This is a repository containing the computational appendix of the article "*Proofs of network quantum nonlocality aided by machine learning*. Alejandro Pozas-Kerstjens, Nicolas Gisin, and Marc-Olivier Renou. [arXiv:2203.xxxxx](https://www.arxiv.org/abs/2203.xxxxx)." It provides the codes for setting up and solving the inflation problems that identify the distributions ![](https://latex.codecogs.com/svg.latex?q_u%5E%7Bt%3D-1%7D) as not admitting triangle-local models for the range ![](https://latex.codecogs.com/svg.latex?0.7504%5Cleq&space;u%5Cleq0.8101).

The code is written in Python and Mathematica.

Python libraries required:
- [numpy](https://www.numpy.org) for math operations
- [sympy](https://www.sympy.org) for symbolic operations
- [scipy](https://scipy.org) for root finding
- [wolframclient](https://reference.wolfram.com/language/WolframClientForPython) for interaction with Wolfram Mathematica
- [mosek](https://www.mosek.com) for solving the linear programming problems
- (optional) [gurobipy](https://www.gurobi.com) for solving the linear programming problems
- [argparse](https://docs.python.org/3/library/argparse.html), [itertools](https://docs.python.org/3/library/itertools.html), [json](https://docs.python.org/3/library/json.html), [math](https://docs.python.org/3/library/math.html), [os](https://docs.python.org/3/library/os.html)

Files:

  - [certificatesExpressionsAndValidities](https://github.com/apozas/triangle-quantum-nonlocality/blob/main/certificatesExpressionsAndValidities.tsv): Table containing the expressions of the witnesses in (symmetric) correlator form, along with the range in which they witness that the distributions ![](https://latex.codecogs.com/svg.latex?q_u%5E%7Bt%3D-1%7D) do not admit a triangle local model.

  - [ComputeWitnessesAndRanges](https://github.com/apozas/triangle-quantum-nonlocality/blob/main/ComputeWitnessesAndRanges.nb): Takes the certificates of infeasibility of the inflation linear programs created in `prove_full_range` and converts them into inequalities valid for arbitrary binary-outcome distributions. It also computes explicitly the range of validity for ![](https://latex.codecogs.com/svg.latex?u) when evaluated on the distributions ![](https://latex.codecogs.com/svg.latex?q_u%5E%7Bt%3D-1%7D).

  - [prove_full_range](https://github.com/apozas/triangle-quantum-nonlocality/blob/main/prove_full_range.py): For a specified initial value of ![](https://latex.codecogs.com/svg.latex?u), iteratively check whether ![](https://latex.codecogs.com/svg.latex?q_u%5E%7Bt%3D-1%7D) admits a triangle-local model, and if not compute until which value of ![](https://latex.codecogs.com/svg.latex?u%27) the corresponding certificate witnesses ![](https://latex.codecogs.com/svg.latex?q_%7Bu%27%7D%5E%7Bt%3D-1%7D) as not admitting a triangle-local model. Usage: `prove_full_range.py -u u_init (default=0.8101) -s solver (default mosek) -d direction_of_uprime (default decrease) -save save_expressions (default False)`.

  - [test_one_point](https://github.com/apozas/triangle-quantum-nonlocality/blob/main/test_one_point.py): For a specified value of ![](https://latex.codecogs.com/svg.latex?u) (and optionally ![](https://latex.codecogs.com/svg.latex?F_2) and ![](https://latex.codecogs.com/svg.latex?F_3)), check if the distribution ![](https://latex.codecogs.com/svg.latex?q_u%5E%7Bt%3D-1%7D) evaluated at the point ![](https://latex.codecogs.com/svg.latex?%28F_2%2CF_3%29) admits a triangle-local model. Usage: `prove_full_range.py -u u_init (required) -s solver (optional, default mosek) -f2 value_of_F2 (optional, default None) -f3 value_of_F3 (optional, default None)`.

  - [utils](https://github.com/apozas/triangle-quantum-nonlocality/blob/main/utils.py): additional functions.

  - [vertex_functions](https://github.com/apozas/triangle-quantum-nonlocality/blob/main/vertex_functions.py): definition of functions that provide the coordinates ![](https://latex.codecogs.com/svg.image?(F_%7BAB%7D,F_%7BBC%7D,F_%7BAC%7D,F_%7BABC%7D)) of all the points that are vertices of the polytope at some value of ![](https://latex.codecogs.com/svg.latex?u).

If you would like to cite this work, please use the following format:

A. Pozas-Kerstjens, N. Gisin, and M.-O. Renou, _Proofs of network quantum nonlocality aided by machine learning_, arXiv:2203.xxxxx

```
@misc{pozaskerstjens2022triangle,
author = {Pozas-Kerstjens, Alejandro and Gisin, Nicolas and Renou, Marc-Olivier},
title = {{Proofs of network quantum nonlocality aided by machine learning}},
eprint = {2203.xxxxx},
archivePrefix={arXiv}
}
```