
from qiskit_optimization.algorithms import SlsqpOptimizer
from qiskit_optimization import QuadraticProgram

slsqp = SlsqpOptimizer(trials=0)


problem = QuadraticProgram()
x = problem.continuous_var(name="x")
y = problem.continuous_var(name="y")
problem.maximize(linear=[2, 0], quadratic=[[-1, 2], [0, -2]])

slsqp.solve(problem)
# see discussion in issue https://github.com/Qiskit/qiskit-optimization/issues/41
