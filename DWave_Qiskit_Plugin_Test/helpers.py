#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 11:07:42 2021

@author: tom
"""

from qiskit.optimization import QuadraticProgram
from docplex.mp.advmodel import AdvModel
from docplex.mp.model_reader import ModelReader

from qiskit.optimization.algorithms import MinimumEigenOptimizer

from dwave.plugins.qiskit import DWaveMinimumEigensolver


import os

def createSolver(sampler = None, num_reads = None ):
    if sampler is None :
        if num_reads is None:
            dwave_solver = DWaveMinimumEigensolver()
        else: 
            dwave_solver = DWaveMinimumEigensolver(num_reads = num_reads)
    else:
        if num_reads is None: # use sampler default
            dwave_solver = DWaveMinimumEigensolver(sampler =  sampler)
        else:
            dwave_solver = DWaveMinimumEigensolver(sampler =  sampler, num_reads = num_reads)
    return MinimumEigenOptimizer(dwave_solver)


def createModel(filename,model_name):
    model = ModelReader.read(filename=filename,model_name = model_name, model_class=AdvModel)
    qp = QuadraticProgram()
    qp.from_docplex(model)
    qp.model = model
    return qp

def createModelsFromDir(path):
    _, _, filenames = next(os.walk(path))
    qps = {}
    for file in filenames:
        name,_ = os.path.splitext(file)
        qp = createModel(path+file,name)
        qps[name] = qp   
    return qps