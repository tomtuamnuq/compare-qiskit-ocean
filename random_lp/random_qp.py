"""
Created on Sat Mar 27 11:49:35 2021.

@author: tom
"""


from typing import Tuple, Optional, Dict, TypedDict
import numpy as np

from qiskit.optimization import QuadraticProgram
from qiskit.optimization.converters import QuadraticProgramToQubo

from docplex.mp.model import Model

from utilities.helpers import cplex_varname


class QP_Boundaries(TypedDict):
    x: Tuple[int, int]
    A: Tuple[int, int]
    Q: Tuple[int, int]
    c: Tuple[int, int]
    d: int


class RandomQP(QuadraticProgram):
    """Random quadratic program class.

    Defines a class with randomly constructed constraints and objective function.
    It is an optimization problem in which the objective function is quadratic and
    all constraints are linear.
    Problem:
    :math:`
        \\underset{x}{\\min}\\quad \\frac{1}{2} x^T Q x + c^T x \\
        \\text{such that }  Ax\\le b\\
         A \\in \\mathbb{Z}^{m\\times n}\\
         Q \\in \\mathbb{Z}^{n\\times n}\\
         x,c \\in \\mathbb{Z}^{n}\\
         b \\in \\mathbb{Z}^{m}\\
    `
    The random program is constructed as follows:
    Choose random Q and c. Choose random x.
    Choose random positive number for each constraint as vector d.
    Set b = Ax + d.
    The upperbound for d alters feasible solution space.
    """

    def __init__(self, num_constr: int, num_vars: int,
                 name: str, multiple: int = 1, *,
                 penalty: Optional[float] = None,
                 boundaries: Optional[QP_Boundaries] = None):
        """
        Create an instance of RandomQP with specified boundaries.

        Args:
            num_constr (int): Number of Constraints.
            num_vars (int): Number of Variables.
            name (str): Name of Quadratic Program.
            multiple (int, optional): Create mutiple unconnected parts with
                num_constr and num_vars each.
            penalty (Optional[float], optional): Penalty factor see
                QuadraticProgramToQubo. Defaults to None.
            boundaries (Optional[dict], optional):
                Dictionary with bounds as follows. Defaults to None.
                "x" : Lower and upper bound of solution space.
                "A" : Lower and upper bound for constraint matrix A.
                "Q" : Lower and upper bound for objective matrix Q.
                "c" : Lower and upper bound for objective vector c.
                "d" : Upper bound for delta vector d.

        """
        super().__init__(name)
        self._dim = (num_constr, num_vars)
        self.multiple = multiple
        self._conv = QuadraticProgramToQubo(penalty)
        self._qubo = None
        if boundaries is not None:
            self._populate_random(boundaries)

    def _populate_random(self, boundaries):
        lower_bound, upper_bound = boundaries["x"]
        linear_objective = np.array([])
        quadratic_objective = {}
        _, num_vars = self._dim
        for k in range(self.multiple):
            self._add_vars(num_vars, lower_bound, upper_bound, k)
            quadratic_objective.update(
                self._create_quadratic_data(k, num_vars, boundaries))

            matrix_a, vec_b, vec_c = self._create_linear_data(boundaries)
            linear_objective = np.append(linear_objective, vec_c)
            self._add_constrs(k, matrix_a, vec_b)
        self.minimize(linear=linear_objective, quadratic=quadratic_objective)
        self._qubo = self._conv.convert(self)

    def _create_quadratic_data(self, k: int, num_vars: int,
                               boundaries: dict) -> Dict[Tuple[str, str], int]:
        """Create random quadratic data Q."""
        #  pylint: disable=invalid-name
        matrix_q_lb, matrix_q_ub = boundaries["Q"]
        n = num_vars
        quadratic = {}
        Q = np.random.randint(matrix_q_lb, matrix_q_ub+1, size=(n, n))
        for i in range(num_vars):
            for j in range(i, num_vars):
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
        lower_bound, upper_bound = boundaries["x"]
        matrix_a_lb, matrix_a_ub = boundaries["A"]
        c_lb, c_ub = boundaries["c"]
        d_ub = boundaries["d"]
        m, n = self._dim
        x = np.random.randint(lower_bound, upper_bound+1, size=n)
        d = np.random.randint(0, d_ub+1, size=m)
        A = np.random.randint(matrix_a_lb, matrix_a_ub+1, size=(m, n))
        b = A.dot(x) + d
        c = np.random.randint(c_lb, c_ub+1, size=n)
        return A, b, c

    def _add_vars(self, num_vars: int, lower_bound: int, upper_bound: int,
                  var_k: int):
        for j in range(num_vars):
            if lower_bound == 0 and upper_bound == 1:
                self.binary_var(cplex_varname(var_k, j))
            else:
                self.integer_var(lowerbound=lower_bound,
                                 upperbound=upper_bound,
                                 name=cplex_varname(var_k, j))

    def _add_constrs(self, k: int, matrix_a: np.ndarray, vec_b: np.ndarray):
        """Add linear constraints to CPLEX model."""
        m, n = self._dim  # pylint: disable=invalid-name
        for i in range(m):
            linear = {}
            for j in range(n):
                linear[cplex_varname(k, j)] = matrix_a[i, j]
            self.linear_constraint(linear=linear, sense='<=',
                                   rhs=vec_b[i],
                                   name='A'+str(k)+"_le"+'b'+str(i))

    @property
    def qubo(self) -> QuadraticProgram:
        """Returns the qubo representation of this random quadratic program.

        Returns:
            This quadratic program in qubo form.
        """
        if self._qubo is None:
            self._qubo = self._conv.convert(self)
        return self._qubo

    @property
    def penalty(self) -> Optional[float]:
        """Returns the penalty factor used in conversion.

        Returns:
            The penalty factor used in conversion.
        """
        return self._conv.penalty

    @penalty.setter
    def penalty(self, penalty: Optional[float]) -> None:
        """Set a new penalty factor.

        Args:
            penalty: The new penalty factor.
                     If None is passed, penalty factor will be automatically calculated.
        """
        self._conv.penalty = penalty
        self._qubo = self._conv.convert(self)

    def complexity(self) -> int:
        """
        Measurement for complexity of problem in terms of quantum bits.

        Returns:
            int: Number of binary variables in QUBO problem.

        """
        return self.qubo.get_num_vars()

    def from_docplex(self, model: Model) -> None:
        """
        Populate an instance of RandomQP from docplex model.

        Args:
            model (Model): The model to built the QP from.

        """
        super().from_docplex(model)
        self._qubo = self._conv.convert(self)
        self._dim = (self.get_num_linear_constraints(), self.get_num_vars())
        self.multiple = 1

    @classmethod
    def create_from_docplex(cls, model: Model,
                            penalty: Optional[float] = None) -> 'RandomQP':
        """
        Create an instance of RandomQP by invoking from_docplex(model).

        Args:
            cls (TYPE): RandomQP.
            model (Model): The model to built the QP from.
            penalty (Optional[float], optional): Penalty factor see
                QuadraticProgramToQubo. Defaults to None.

        Returns:
            random_qp (RandomQP): RandomQP  model.

        """
        rqp = RandomQP(1, 1, "", penalty=penalty)  # set in from_docplex(model)
        rqp.from_docplex(model)
        return rqp

    @classmethod
    def create_random_qp(
            cls,
            name: str, num_constr: int, num_vars: int, *,
            penalty: Optional[float] = None,
            lower_bound: int, upper_bound: int,
            multiple: int = 1,
            matrix_a_lb: int = -5, matrix_a_ub: int = 5,
            matrix_q_lb: int = -2, matrix_q_ub: int = 2,
            c_lb: int = -1, c_ub: int = 1) -> 'RandomQP':
        """
        Create an instance of RandomQP with specified and/or default bounds.

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
                Defaults to -5.
            matrix_a_ub (int, optional): Upper bound for constraint matrix A.
            Defaults to 5.
            matrix_q_lb (int, optional): Lower bound for constraint matrix A.
                Defaults to -2.
            matrix_q_ub (int, optional): Upper bound for constraint matrix A.
            Defaults to 2.
            c_lb (int, optional): Lower bound for objective vector c.
                Defaults to -1.
            c_ub (int, optional): Upper bound for objective vector c.
                Defaults to 1.

        Returns:
            RandomQP: An instance of a randomly constructed quadratic program.

        """
        d_ub = upper_bound+1
        boundaries = {"x": (lower_bound, upper_bound),
                      "A": (matrix_a_lb, matrix_a_ub),
                      "Q": (matrix_q_lb, matrix_q_ub),
                      "c": (c_lb, c_ub),
                      "d": d_ub}
        return RandomQP(
            num_constr, num_vars, name, multiple,
            penalty=penalty,
            boundaries=boundaries)

    @classmethod
    def create_random_binary_prog(cls, name: str,
                                  num_constr: int, num_vars: int, *,
                                  penalty: Optional[float] = None,
                                  multiple: int = 1,
                                  matrix_a_lb: int = -1,
                                  matrix_a_ub: int = 1,
                                  matrix_q_lb: int = -1,
                                  matrix_q_ub: int = 1,
                                  c_lb: int = -1, c_ub: int = 1) -> 'RandomQP':
        """
        Create a random binary quadratic program by calling create_random_qp.

        Args:
            name (str): Name of binary Program..
            num_constr (int): Number of Constraints..
            num_vars (int): Number of Variables.
            penalty (Optional[float], optional): Penalty factor see
                QuadraticProgramToQubo. Defaults to None.
            multiple (int, optional): Create mutiple unconnected parts with
            num_constr and num_vars each. Defaults to 1.
            matrix_a_lb (int, optional): Lower bound for constraint matrix A.
                Defaults to -1.
            matrix_a_ub (int, optional): Upper bound for constraint matrix A.
            Defaults to 1.
            matrix_q_lb (int, optional): Lower bound for constraint matrix A.
                Defaults to -1.
            matrix_q_ub (int, optional): Upper bound for constraint matrix A.
            Defaults to 1.
            c_lb (int, optional): Lower bound for objective vector c.
                Defaults to -1.
            c_ub (int, optional): Upper bound for objective vector c.
                Defaults to 1.
        Returns:
            RandomQP: An instance of a randomly constructed binary program.

        """
        return cls.create_random_qp(name, num_constr, num_vars,
                                    lower_bound=0, upper_bound=1,
                                    multiple=multiple,
                                    penalty=penalty,
                                    matrix_a_lb=matrix_a_lb,
                                    matrix_a_ub=matrix_a_ub,
                                    matrix_q_lb=matrix_q_lb,
                                    matrix_q_ub=matrix_q_ub,
                                    c_lb=c_lb, c_ub=c_ub)
