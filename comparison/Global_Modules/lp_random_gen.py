"""
Created on Sun Jan 31 09:58:32 2021.

@author: tom
"""

from typing import Tuple, Optional
import numpy as np
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.converters import QuadraticProgramToQubo
from docplex.mp.model import Model


def _varname(k, j: int) -> str:
    """Return name for linear program variable to meet CLPEX conventions."""
    return'x' + str(k) + "_" + str(j)


class RandomLP(QuadraticProgram):
    """Random linear program class."""

    _conv = QuadraticProgramToQubo()

    def __init__(self, num_constr: int, num_vars: int,
                 name: str, multiple: int = 1, *,
                 penalty: Optional[float] = None,
                 boundaries: Optional[dict] = None):
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
            boundaries (Optional[dict], optional):
                Dictionary with bounds as follows. Defaults to None.
                "x" : Lower and upper bound of solution space.
                "A" : Lower and upper bound for constraint matrix A.
                "c" : Lower and upper bound for objective vector c.
                "d" : Upper bound for delta vector d.

        """
        super().__init__(name)
        self.dim = (num_constr, num_vars)
        self.multiple = multiple
        self.penalty = penalty
        if penalty is not None:
            RandomLP._conv.penalty = penalty
        if boundaries is not None:
            lower_bound, upper_bound = boundaries["x"]
            objective = np.array([])
            for k in range(multiple):
                self._add_vars(num_vars, lower_bound, upper_bound, k)
                matrix_a, vec_b, vec_c = self._create_data(boundaries)
                objective = np.append(objective, vec_c)
                self._add_constrs(k, matrix_a, vec_b)

            self.minimize(linear=objective)
            self.qubo = RandomLP._conv.convert(self)
        else:
            self.qubo = None

    def _add_vars(self, num_vars: int, lower_bound: int, upper_bound: int,
                  var_k: int):
        for j in range(num_vars):
            if lower_bound == 0 and upper_bound == 1:
                self.binary_var(_varname(var_k, j))
            else:
                self.integer_var(lowerbound=lower_bound,
                                 upperbound=upper_bound,
                                 name=_varname(var_k, j))

    def _create_data(self, boundaries: dict) -> Tuple[np.ndarray,
                                                      np.ndarray, np.ndarray]:
        """Create random data."""
        #  pylint: disable=invalid-name
        lower_bound, upper_bound = boundaries["x"]
        matrix_a_lb, matrix_a_ub = boundaries["A"]
        c_lb, c_ub = boundaries["c"]
        d_ub = boundaries["d"]
        m, n = self.dim
        x = np.random.randint(lower_bound, upper_bound+1, size=n)
        d = np.random.randint(0, d_ub+1, size=m)
        A = np.random.randint(matrix_a_lb, matrix_a_ub+1, size=(m, n))
        b = A.dot(x) + d
        c = np.random.randint(c_lb, c_ub+1, size=n)
        return A, b, c

    def _add_constrs(self, k: int, matrix_a: np.ndarray, vec_b: np.ndarray):
        """Add linear constraints to CPLEX model."""
        m, n = self.dim  # pylint: disable=invalid-name
        for i in range(m):
            linear = {}
            for j in range(n):
                linear[_varname(k, j)] = matrix_a[i, j]
            self.linear_constraint(linear=linear, sense='<=',
                                   rhs=vec_b[i],
                                   name='A'+str(k)+"_le"+'b'+str(i))

    def complexity(self) -> int:
        """
        Measurement for complexity of problem in terms of quantum bits.

        Returns:
            int: Number of binary variables in QUBO problem.

        """
        return self.qubo.get_num_vars()

    def from_docplex(self, model: Model) -> None:
        """
        Populate an instance of RandomLP from docplex model.

        Args:
            model (Model): The model to built the LP from.

        """
        super().from_docplex(model)
        self.qubo = RandomLP._conv.convert(self)
        self.dim = (self.get_num_linear_constraints(), self.get_num_vars())
        self.multiple = 1

    @classmethod
    def create_from_docplex(cls, model: Model,
                            penalty: Optional[float] = None) -> 'RandomLP':
        """
        Create an instance of RandomLP by invoking from_docplex(model).

        Args:
            cls (TYPE): RandomLP.
            model (Model): The model to built the LP from.
            penalty (Optional[float], optional): Penalty factor see
                QuadraticProgramToQubo. Defaults to None.

        Returns:
            random_lp (RandomLP): RandomLP  model.

        """
        rlp = RandomLP(1, 1, "", penalty=penalty)  # set in from_docplex(model)
        rlp.from_docplex(model)
        return rlp

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
