"""
Created on Sun Jan 31 11:07:42 2021.

@author: tom
"""

from typing import Optional

from qiskit.optimization.algorithms import MinimumEigenOptimizer
from dwave.plugins.qiskit import DWaveMinimumEigensolver
from dwave.system import AutoEmbeddingComposite, DWaveSampler

from utilities.custom_args_sampler import CustomArgsSampler


def create_dwave_meo(sampler: DWaveSampler = None,
                     penalty: Optional[float] = None,
                     **sample_kwargs) -> MinimumEigenOptimizer:
    """
    Create a dwave minimum eigen optimizer.

    Args:
        sampler (DWaveSampler, optional): Sampler to use.
            Defaults to None.
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


def cplex_varname(k, j: int) -> str:
    """Return name for quadratic program variable to meet CLPEX conventions."""
    return'x' + str(k) + "_" + str(j)
