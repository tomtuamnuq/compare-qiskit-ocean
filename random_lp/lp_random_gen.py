"""
Created on Sun Jan 31 09:58:32 2021.

@author: tom
"""

import os
from typing import Tuple, Optional, Dict, TypedDict, Union
from collections import OrderedDict
import numpy as np

from random_lp.random_qp import RandomQuadraticProgram


class RLPBoundaries(TypedDict):
    x: Tuple[int, int]
    A: Tuple[int, int]
    c: Tuple[int, int]
    d: int


class RandomLP(RandomQuadraticProgram):
    """Random linear program class.

    Defines a class with randomly constructed constraints and objective function.
    It is an optimization problem in which both the objective function and
    all constraints are linear.
    Problem:
    :math:`
        \\underset{x}{\\min}\\quad \\c^T x \\
        \\text{such that }  Ax\\le b\\
         A \\in \\mathbb{Z}^{m\\times n}\\
         x,c \\in \\mathbb{Z}^{n}\\
         b \\in \\mathbb{Z}^{m}\\
    `
    The random program is constructed as follows:
    Choose random c. Choose random x. Choose random positive number for each constraint as vector d.
    Set b = Ax + d.
    The upperbound for d alters feasible solution space.
    """

    def __init__(self, num_constr: int, num_vars: int,
                 name: str, multiple: int = 1, *,
                 penalty: Optional[float] = None,
                 boundaries: Optional[RLPBoundaries] = None):
        """
        Create an instance of RandomLP with specified boundaries.

        Args:
            num_constr (int): Number of Constraints.
            num_vars (int): Number of Variables.
            name (str): Name of Linear Program.
            multiple (int, optional): Create mutiple unconnected parts with
                num_constr and num_vars each.
            penalty (Optional[float], optional): Penalty factor see
                QuadraticProgramToQubo. Defaults to None.
            boundaries (Optional[RLPBoundaries], optional):
                Dictionary with bounds as follows. Defaults to None.
                "x" : Lower and upper bound of solution space.
                "A" : Lower and upper bound for constraint matrix A.
                "c" : Lower and upper bound for objective vector c.
                "d" : Upper bound for delta vector d.

        """
        base_boundaries = boundaries.copy()
        base_boundaries["Q"] = (0, 0)
        super().__init__(num_constr, num_vars, name, multiple,
                         penalty=penalty,
                         boundaries=base_boundaries)

    def _populate_random(self, boundaries):
        """Create random constraints and objective function."""
        lower_bound, upper_bound = boundaries["x"]
        objective = np.array([])
        _, num_vars = self._dim
        for k in range(self.multiple):
            self._add_vars(num_vars, lower_bound, upper_bound, k)
            matrix_a, vec_b, vec_c = self._create_linear_data(boundaries)
            objective = np.append(objective, vec_c)
            self._add_constrs(k, matrix_a, vec_b)
        self.minimize(linear=objective)
        self._qubo = self._conv.convert(self)

    @classmethod
    def create_random_lp(
            cls,
            name: str, num_constr: int, num_vars: int, *,
            penalty: Optional[float] = None,
            lower_bound: int, upper_bound: int,
            multiple: int = 1,
            matrix_a_lb: int = -10, matrix_a_ub: int = 10,
            c_lb: int = -1, c_ub: int = 1) -> 'RandomLP':
        """
        Create an instance of RandomLP with specified and/or default bounds.

        Args:
            name (str): Name of Linear Program.
            num_constr (int): Number of Constraints.
            num_vars (int): Number of Variables.
            penalty (Optional[float], optional): Penalty factor see
                QuadraticProgramToQubo. Defaults to None.
            lower_bound (int): Lower bound of solution space.
            upper_bound (int): Upper bound of solution space.
            multiple (int, optional): Create mutiple unconnected parts with
            num_constr and num_vars each. Defaults to 1.
            matrix_a_lb (int, optional): Lower bound for constraint matrix A.
                Defaults to -10.
            matrix_a_ub (int, optional): Upper bound for constraint matrix A.
            Defaults to 10.
            c_lb (int, optional): Lower bound for objective vector c.
                Defaults to -1.
            c_ub (int, optional): Upper bound for objective vector c.
                Defaults to 1.

        Returns:
            RandomLP: An instance of a randomly constructed linear program.

        """
        d_ub = upper_bound+1
        boundaries = {"x": (lower_bound, upper_bound),
                      "A": (matrix_a_lb, matrix_a_ub),
                      "c": (c_lb, c_ub),
                      "d": d_ub}
        return RandomLP(
            num_constr, num_vars, name, multiple,
            penalty=penalty,
            boundaries=boundaries)

    @classmethod
    def create_random_binary_prog(cls, name: str,
                                  num_constr: int, num_vars: int, *,
                                  penalty: Optional[float] = None,
                                  multiple: int = 1,
                                  matrix_a_lb: int = -3,
                                  matrix_a_ub: int = 3,
                                  c_lb: int = -2, c_ub: int = 2) -> 'RandomLP':
        """
        Create a random binary linear program by calling create_random_lp.

        Args:
            name (str): Name of binary Program..
            num_constr (int): Number of Constraints..
            num_vars (int): Number of Variables.
            penalty (Optional[float], optional): Penalty factor see
                QuadraticProgramToQubo. Defaults to None.
            multiple (int, optional): Create mutiple unconnected parts with
            num_constr and num_vars each. Defaults to 1.
            matrix_a_lb (int, optional): Lower bound for constraint matrix A.
                Defaults to -3.
            matrix_a_ub (int, optional): Upper bound for constraint matrix A.
            Defaults to 3.
            c_lb (int, optional): Lower bound for objective vector c.
                Defaults to -2.
            c_ub (int, optional): Upper bound for objective vector c.
                Defaults to 2.
        Returns:
            RandomLP: An instance of a randomly constructed binary program
            with fixed bounds.

        """
        return cls.create_random_lp(name, num_constr, num_vars,
                                    lower_bound=0, upper_bound=1,
                                    multiple=multiple,
                                    penalty=penalty,
                                    matrix_a_lb=matrix_a_lb,
                                    matrix_a_ub=matrix_a_ub,
                                    c_lb=c_lb, c_ub=c_ub)


def create_models(path: 'Union[list[str],str]', penalty:
                  Optional[float] = None) -> Dict[str, RandomQuadraticProgram]:
    """
    Create random quadratic program instances from cplex model files.

    Args:
        path (str): File link to read all models from.
        penalty (TYPE): Set individual penalty terms to be used by
            QuadraticProgramToQubo.

    Returns:
        qps_sorted (OrderedDict): A dict with RandomQuadraticProgram instances
         ordered by number of qubits.

    """
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path
    qps = {}
    for path_ in paths:
        _, _, filenames = next(os.walk(path_))

        for file in filenames:
            name, _ = os.path.splitext(file)
            qps[name] = RandomLP.create_from_lp_file(
                path_+file, penalty=penalty)

    qps_sorted = OrderedDict()

    for qp_name in sorted(qps.keys(),
                          key=lambda name: qps[name].complexity()):
        qps_sorted[qp_name] = qps[qp_name]

    return qps_sorted
