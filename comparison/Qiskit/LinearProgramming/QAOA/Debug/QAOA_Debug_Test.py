import os

from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qiskit.optimization.algorithms.optimization_algorithm import OptimizationResultStatus
from qiskit.optimization import QuadraticProgram


Q_SEED = 10598  # as used in most issues
aqua_globals.random_seed = Q_SEED
shots = 4096

# select linear program to solve
path = os.path.dirname(__file__) + "/test_8.lp"
qp = QuadraticProgram()

qp.read_from_lp_file(path)

# init classical Optimizers
optimizer = COBYLA()
cplex = CplexOptimizer()

# solve classically
print(cplex.solve(qp))

# solve qp with Minimum Eigen Optimizer QAOA
backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend,
                                   seed_simulator=Q_SEED,
                                   seed_transpiler=Q_SEED,
                                   shots=shots)

# QAOA
qaoa_mes = QAOA(quantum_instance=quantum_instance, optimizer=optimizer)
qaoa = MinimumEigenOptimizer(qaoa_mes)

res = qaoa.solve(qp)
print(res)
print(qp.is_feasible(res.x))
