"""
Created on Sun Jan 31 11:07:42 2021.

@author: tom
"""

from typing import Optional

from qiskit.optimization.algorithms import MinimumEigenOptimizer
from dwave.plugins.qiskit import DWaveMinimumEigensolver
from dwave.system import DWaveSampler


def create_dwave_meo(sampler: DWaveSampler = None,
                     num_reads: Optional[int] = None,
                     penalty: Optional[float] = None) -> MinimumEigenOptimizer:
    """
    Create a dwave minimum eigen optimizer.

    Args:
        sampler (DWaveSampler, optional): Sampler to use.
            Defaults to None.
        num_reads (Optional[int], optional): Set num_reads in
           DWaveMinimumEigensolver. Defaults to None.
        penalty (Optional[int], optional): Set penalty in
           MinimumEigensolver. Defaults to None.

    Returns:
        MinimumEigenOptimizer: Optimizer with DWaveMinimumEigensolver.

    """
    if num_reads is None:
        dwave_solver = DWaveMinimumEigensolver(sampler=sampler)
    else:
        dwave_solver = DWaveMinimumEigensolver(sampler=sampler,
                                               num_reads=num_reads)
    return MinimumEigenOptimizer(dwave_solver, penalty=penalty)
