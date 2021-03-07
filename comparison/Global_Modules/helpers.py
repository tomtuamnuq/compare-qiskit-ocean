#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 11:07:42 2021

@author: tom
"""

from qiskit.optimization import QuadraticProgram
from qiskit.optimization.converters import QuadraticProgramToQubo
from docplex.mp.advmodel import AdvModel
from docplex.mp.model_reader import ModelReader

from qiskit.optimization.algorithms import MinimumEigenOptimizer

from dwave.plugins.qiskit import DWaveMinimumEigensolver
from collections import OrderedDict
import re

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
    qp.qubo_model = QuadraticProgramToQubo().convert(qp)
    return qp

def createModelsFromDir(path):
    _, _, filenames = next(os.walk(path))
    qps = {}
    for file in filenames:
        name,_ = os.path.splitext(file)
        qp = createModel(path+file,name)
        
        qps[name] = qp  
    
    

    qps_sorted = OrderedDict()
        
    for qp_name in sorted(qps.keys(), key = lambda name : complexity_of_quadratic_program(qps[name])) :
        qps_sorted[qp_name] = qps[qp_name]
        
    return qps_sorted

    
def complexity_of_quadratic_program(qp):
    if hasattr(qp, "qubo_model"):
        return qp.qubo_model.get_num_vars()
    else:
        return QuadraticProgramToQubo().convert(qp).get_num_vars()