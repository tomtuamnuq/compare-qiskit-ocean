"""
Created on Sun Jan 31 11:07:42 2021.

@author: tom
"""

from typing import Optional

from qiskit.optimization.algorithms import MinimumEigenOptimizer
from qiskit import BasicAer
from qiskit.providers.backend import Backend
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.components.optimizers import COBYLA

from dwave.plugins.qiskit import DWaveMinimumEigensolver
from dwave.system import AutoEmbeddingComposite, DWaveSampler

from utilities.custom_args_sampler import CustomArgsSampler


def create_dwave_meo(sampler: DWaveSampler = None,
                     penalty: Optional[float] = None,
                     **sample_kwargs) -> MinimumEigenOptimizer:
    """
    Create a dwave minimum eigen optimizer with a CustomArgsSampler.

    Args:
        sampler (DWaveSampler, optional): Sampler to use.
            Defaults to None. In this case AutoEmbeddingComposite(DWaveSampler()) is used.
        penalty (Optional[int], optional): Set penalty in
           MinimumEigensolver. Defaults to None.
        **sample_kwargs:
            Optional keyword arguments for the sampling method, specified for sampler in
            D-Wave System Documentation's `solver guide
             <https://docs.dwavesys.com/docs/latest/doc_solver_ref.html>`_
            describes the parameters and properties supported on the D-Wave system.

    Returns:
        MinimumEigenOptimizer: Optimizer with DWaveMinimumEigensolver.

    """
    if sampler is None:
        sampler = AutoEmbeddingComposite(DWaveSampler())
    custom_sampler = CustomArgsSampler(sampler, sample_kwargs=sample_kwargs)
    dwave_solver = DWaveMinimumEigensolver(sampler=custom_sampler)

    return MinimumEigenOptimizer(dwave_solver, penalty=penalty)


def create_qaoa_meo(backend: Backend = None,
                    penalty: Optional[float] = None,
                    q_seed: int = 10598,
                    shots: int = 4096) -> MinimumEigenOptimizer:
    """
    Create a qaoa minimum eigen optimizer with.

    Args:
        backend (Backend, optional): Backend to use.
            Defaults to None. In this case qasm simulator is used.
        penalty (Optional[int], optional): Set penalty in
           MinimumEigensolver. Defaults to None.
        q_seed (int): Set quantum seed in aqua and quantum instance.
            Defaults to 10598(as used in most issues).
        shots (int): Set number of circuit exectutions in quantum instance.
            Defaults to 4096

    Returns:
        MinimumEigenOptimizer: Optimizer with QAOA mes.
    """
    aqua_globals.random_seed = q_seed
    optimizer = COBYLA()
    if backend is None:
        backend = BasicAer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend,
                                       seed_simulator=q_seed,
                                       seed_transpiler=q_seed,
                                       shots=shots)
    qaoa_mes = QAOA(quantum_instance=quantum_instance, optimizer=optimizer)

    return MinimumEigenOptimizer(qaoa_mes, penalty=penalty)


def cplex_varname(k, j: int) -> str:
    """Return name for quadratic program variable to meet CLPEX conventions."""
    return'x' + str(k) + "_" + str(j)
