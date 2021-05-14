import os
from qiskit import QuantumCircuit

circuit = QuantumCircuit()
path_to_lp_file = os.path.dirname(__file__) + "/example.lp"

# 1 problem loading and conversion
import qiskit_optimization as q_opt

qp = q_opt.QuadraticProgram()
qp.read_from_lp_file(path_to_lp_file)
qp2qubo = q_opt.converters.QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)

# 2 convert to Ising operator
qubit_operator, _ = qubo.to_ising()

# 3 model with pyqubo
from pyqubo import Binary

x_0, x_1, x_2 = (Binary("x_" + str(i)) for i in range(3))
linear_obj = -x_0 - x_2
quadratic_obj = -x_0 + x_0 * x_2 + x_1 - 2 * x_1 * x_2
linear_cstr = 8 * (x_1 - x_2) ** 2
qubo_objective = linear_obj + quadratic_obj + linear_cstr
qubo_bqm = qubo_objective.compile().to_bqm()

# 4 transpile circuit

# load account and QPU
from qiskit import IBMQ, transpile, execute

IBMQ.load_account()
provider = IBMQ.get_provider(hub="ibm-q")
qpu_belem = provider.get_backend("ibmq_belem")
# bring logical into native form
circuit_transpiled = transpile(circuit, backend=qpu_belem)
# visualize
circuit_transpiled.draw(output="mpl")

# 5 find embedding
import minorminer, dwave_networkx, dwave.inspector
from dimod import StructureComposite

src = qubo_bqm.to_networkx_graph()
target = dwave_networkx.pegasus_graph(16)
embedding = minorminer.find_embedding(src, target)

# 6 job submission

# create D-Wave interface instance
from dwave.system import DWaveSampler, FixedEmbeddingComposite

struc = StructureComposite(DWaveSampler(), src.nodes, src.edges)
dwave = FixedEmbeddingComposite(struc, embedding=embedding)

# submit jobs
num_times = 1024
dwave_result = dwave.sample(qubo_bqm, num_reads=num_times)
ibmq_job = execute(circuit, qpu_belem, shots=num_times)
ibmq_result = ibmq_job.result()

# 7 call inspector
dwave.inspector.show(dwave_result)
