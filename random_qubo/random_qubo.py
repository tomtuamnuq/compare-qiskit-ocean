"""
Created on Mon Apr 26 10:13:34 2021.

@author: tom
"""


from typing import Tuple, Optional, Dict, TypedDict
import numpy as np
from qiskit_optimization import QuadraticProgram

from utilities.helpers import cplex_varname


class RQuboBoundaries(TypedDict):
    """Boundary dict typing class for RandomQubo instances.

    """
    Q: Tuple[int, int]
    c: Tuple[int, int]


class RandomQubo(QuadraticProgram):
    """Random qubo class.

    Defines a class with randomly constructed qubos.
    It is an optimization problem in which the objective function is quadratic and
    no constraints exist.
    Problem:
    :math:`
        \\underset{x}{\\min}\\quad \\frac{1}{2} x^T Q x + c^T x \\
         Q \\in \\mathbb{Z}^{n\\times n}\\
         x,c \\in \\mathbb{Z}^{n}\\
    `
    The random instance is constructed as follows:
    Choose random Q and c multiple times to create sparse problems with clear structure.
    """

    def __init__(self, num_vars: int,
                 name: str, multiple: int = 1, *,
                 boundaries: Optional[RQuboBoundaries] = None):
        """
        Create an instance of RandomQubo with specified boundaries.

        Args:
            num_vars (int): Number of Variables.
            name (str): Name of Quadratic Program.
            multiple (int, optional): Create mutiple unconnected parts with
                num_vars variables each.
            boundaries (Optional[dict], optional):
                Dictionary with bounds as follows. Defaults to None.
                "Q" : Lower and upper bound for objective matrix Q.
                "c" : Lower and upper bound for objective vector c.

        """
        super().__init__(name)
        self._num_vars = num_vars
        self._multiple = multiple
        if boundaries is not None:
            self._populate_random(boundaries)

    def _populate_random(self, boundaries):
        linear_objective = np.array([])
        quadratic_objective = {}
        for k in range(self._multiple):
            self._add_vars(k)
            quadratic_objective.update(
                self._create_quadratic_data(k, boundaries))

            vec_c = self._create_linear_data(boundaries)
            linear_objective = np.append(linear_objective, vec_c)

        self.minimize(linear=linear_objective, quadratic=quadratic_objective)

    def _create_quadratic_data(self, k: int,
                               boundaries: dict) -> Dict[Tuple[str, str], int]:
        """Create random quadratic data Q."""
        #  pylint: disable=invalid-name
        matrix_q_lb, matrix_q_ub = boundaries["Q"]
        n = self._num_vars
        quadratic = {}
        Q = np.random.randint(matrix_q_lb, matrix_q_ub+1, size=(n, n))
        for i in range(n):
            for j in range(i, n):
                var_i = cplex_varname(k, i)
                if i == j:
                    quadratic[var_i, var_i] = Q[i, i]
                else:
                    var_j = cplex_varname(k, j)
                    quadratic[var_i, var_j] = Q[i, j] + Q[j, i]
        return quadratic

    def _create_linear_data(self, boundaries: dict) -> Tuple[np.ndarray,
                                                             np.ndarray, np.ndarray]:
        """Create random linear data A, b and c."""
        #  pylint: disable=invalid-name
        c_lb, c_ub = boundaries["c"]
        c = np.random.randint(c_lb, c_ub+1, size=self._num_vars)
        return c

    def _add_vars(self, var_k: int):
        """Add variables to CPLEX model."""
        for j in range(self._num_vars):
            self.binary_var(cplex_varname(var_k, j))

    @classmethod
    def create_random_qubo(cls, name: str, num_vars: int, *,
                           multiple: int = 1,
                           matrix_q_lb: int = -1,
                           matrix_q_ub: int = 1,
                           c_lb: int = -1, c_ub: int = 1) -> 'RandomQubo':
        """
        Create an instance of RandomQubo with specified and/or default bounds.
        Args:
            name (str): Name of binary Program..
            num_vars (int): Number of Variables.
            multiple (int, optional): Create mutiple unconnected parts with
            num_vars each. Defaults to 1.
            matrix_q_lb (int, optional): Lower bound for objective matrix Q.
                Defaults to -1.
            matrix_q_ub (int, optional): Upper bound for objective matrix Q.
            Defaults to 1.
            c_lb (int, optional): Lower bound for objective vector c.
                Defaults to -1.
            c_ub (int, optional): Upper bound for objective vector c.
                Defaults to 1.
        Returns:
            RandomQubo: An instance of RandomQubo.

        """
        boundaries = {"Q": (matrix_q_lb, matrix_q_ub),
                      "c": (c_lb, c_ub)}
        return RandomQubo(
            num_vars, name, multiple,
            boundaries=boundaries)
