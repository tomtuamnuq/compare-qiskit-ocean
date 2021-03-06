#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 5 08:03:42 2021

@author: tom
"""

from qiskit.optimization import QuadraticProgram
from docplex.mp.advmodel import AdvModel
from docplex.mp.model_reader import ModelReader

from qiskit.optimization.algorithms import MinimumEigenOptimizer



import os


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