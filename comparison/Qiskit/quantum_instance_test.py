from qiskit import IBMQ
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.test.mock import FakeMumbai
from qiskit.utils import QuantumInstance
from qiskit.providers.aer.noise import NoiseModel
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer


def job_callback(job_id, job_status, queue_position, job):
    # BUG ?
    print(job_id)
    print(job_status)
    print(queue_position)
    print(job)


problem = QuadraticProgram()
x = problem.continuous_var(name="x")
y = problem.continuous_var(name="y")
problem.maximize(linear=[2, 0], quadratic=[[-1, 2], [0, -2]])

# init IBM Q Experience Simulator
IBMQ.load_account()
backend = IBMQ.get_provider(hub='ibm-q').get_backend('simulator_statevector')

quantum_instance_kwargs = {"shots": 512,
                           "job_callback": job_callback}

quantum_instance = QuantumInstance(backend, **quantum_instance_kwargs)
print("only quantum instance")
print(f"{quantum_instance._job_callback=}")
qaoa_mes = QAOA(quantum_instance=quantum_instance,
                optimizer=COBYLA(maxiter=2))

qaoa_ibmq_sim = MinimumEigenOptimizer(qaoa_mes)
print("QAOA meo no noise model")
print(f"{qaoa_ibmq_sim.min_eigen_solver.quantum_instance._job_callback=}")

device = FakeMumbai()
noise_model = NoiseModel.from_backend(device)
conf = device.configuration()
quantum_instance_kwargs_noise = {"shots": 512, "noise_model": noise_model,
                                 "job_callback": job_callback,
                                 "coupling_map": conf.coupling_map,
                                 "basis_gates": conf.basis_gates}
quantum_instance_noise = QuantumInstance(backend, **quantum_instance_kwargs)
qaoa_mes_noise = QAOA(quantum_instance=quantum_instance_noise,
                      optimizer=COBYLA(maxiter=2))
qaoa_ibmq_noise = MinimumEigenOptimizer(qaoa_mes_noise)
print("QAOA meo with noise model")
print(f"{qaoa_ibmq_noise.min_eigen_solver.quantum_instance._job_callback=}")
