"""
Created on Sun Jan 31 11:07:42 2021.

@author: tom
"""

import os
import warnings

from typing import Dict, Optional, Union
from collections import OrderedDict

from qiskit import BasicAer
from qiskit.providers.backend import Backend
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA

with warnings.catch_warnings():
    # Since Aqua is deprecated but dwave_qiskit_plugin is not updated
    # we ignore DeprecationWarnings.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from qiskit.optimization.algorithms import MinimumEigenOptimizer \
        as MinimumEigenOptimizer_  # deprecated
    from qiskit.optimization import QuadraticProgram \
        as QuadraticProgram_

from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


from dwave.plugins.qiskit import DWaveMinimumEigensolver
from dwave.system import AutoEmbeddingComposite, DWaveSampler

from utilities.custom_args_sampler import CustomArgsSampler


def create_dwave_meo(sampler: DWaveSampler = None,
                     penalty: Optional[float] = None,
                     **sample_kwargs) -> MinimumEigenOptimizer_:
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if sampler is None:
            sampler = AutoEmbeddingComposite(DWaveSampler())
        custom_sampler = CustomArgsSampler(
            sampler, sample_kwargs=sample_kwargs)
        dwave_solver = DWaveMinimumEigensolver(sampler=custom_sampler)

        return MinimumEigenOptimizer_(dwave_solver, penalty=penalty)


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
    algorithm_globals.random_seed = q_seed
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
    if k == 0:
        name = 'x' + "_" + str(j)
    else:
        name = 'x' + str(k) + "_" + str(j)
    return name


def create_quadratic_programs_from_paths(
        path: 'Union[list[str],str]',
        legacy: bool = False) -> dict:
    """Create quadratic program instances from cplex model files.

    Args:
        path (str): File link or links to read all models from.

    Returns:
        qps_sorted (OrderedDict): A dict with QuadraticProgram instances
         ordered by number of variables.

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
            if legacy:
                qubo = QuadraticProgram_()
            else:
                qubo = QuadraticProgram()
            qubo.read_from_lp_file(path_+file)
            qps[name] = qubo

    qps_sorted = OrderedDict()

    for qp_name in sorted(qps.keys(),
                          key=lambda name: qps[name].get_num_vars()):
        qps_sorted[qp_name] = qps[qp_name]

    return qps_sorted
