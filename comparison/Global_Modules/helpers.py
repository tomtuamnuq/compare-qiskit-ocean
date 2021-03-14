"""
Created on Sun Jan 31 11:07:42 2021.

@author: tom
"""

from collections import OrderedDict
from typing import Optional, Dict

from docplex.mp.advmodel import AdvModel
from docplex.mp.model_reader import ModelReader

from qiskit.optimization.algorithms import MinimumEigenOptimizer
from dwave.plugins.qiskit import DWaveMinimumEigensolver
from dwave.system import DWaveSampler

import lp_random_gen as lp_gen

import os


def create_dwave_meo(sampler: DWaveSampler = None,
                     num_reads: Optional[int] = None) -> MinimumEigenOptimizer:
    """
    Create a dwave minimum eigen optimizer.

    Args:
        sampler (DWaveSampler, optional): Sampler to use.
            Defaults to None.
        num_reads (Optional[int], optional): Set num_reads in
           DWaveMinimumEigensolver. Defaults to None.

    Returns:
        MinimumEigenOptimizer: Optimizer with DWaveMinimumEigensolver.

    """
    if num_reads is None:
        dwave_solver = DWaveMinimumEigensolver(sampler=sampler)
    else:
        dwave_solver = DWaveMinimumEigensolver(sampler=sampler,
                                               num_reads=num_reads)
    return MinimumEigenOptimizer(dwave_solver)


def _create_model(filename: str, model_name: str,
                  penalty: Optional[float] = None) -> lp_gen.RandomLP:
    """Create a random linear program from cplex model in file."""
    model = ModelReader.read(filename=filename,
                             model_name=model_name, model_class=AdvModel)

    return lp_gen.RandomLP.create_from_docplex(model, penalty=penalty)


def create_models(path: str, penalty:
                  Optional[float] = None) -> Dict[str, lp_gen.RandomLP]:
    """
    Create random linear program instances from cplex model files.

    Args:
        path (str): File link to read all models from.
        penalty (TYPE): Set individual penalty terms to be used by
            QuadraticProgramToQubo.

    Returns:
        qps_sorted (OrderedDict): An ordered dict with RandomLP instances.

    """
    _, _, filenames = next(os.walk(path))
    qps = {}
    for file in filenames:
        name, _ = os.path.splitext(file)
        qps[name] = _create_model(path+file, name, penalty=penalty)

    qps_sorted = OrderedDict()

    for qp_name in sorted(qps.keys(),
                          key=lambda name: qps[name].complexity()):
        qps_sorted[qp_name] = qps[qp_name]

    return qps_sorted
