"""
Created on Sun Jan 31 09:58:32 2021

@author: tom
"""

import numpy as np
from qiskit.optimization import QuadraticProgram

class random_lp_int:
    def __init__(self,num_constr,num_vars,lowerbound,upperbound):
        self.m = num_constr 
        self.n = num_vars 
        self.lb = lowerbound
        self.ub = upperbound
        
    def createLP(self,name):
        self.name = name
        self._create_data(d_ub = self.ub+1)
        self._createLP()
        return self.model


    def _create_data(self, d_ub = 1 , A_lb = -10, A_ub = 10, c_lb = -1, c_ub = 1):
        self.x = np.random.randint(self.lb,self.ub+1,size = self.n)
        self.d = np.random.randint(0,d_ub+1,size = self.m)
        self.A = np.random.randint(A_lb,A_ub+1,size = (self.m,self.n))
        self.b = self.A.dot(self.x) + self.d
        self.c = np.random.randint(c_lb,c_ub+1,size = self.n)
        
    def _createLP(self):
        model = QuadraticProgram(self.name)

        # add variables
        for j in range(self.n):
            if self.lb == 0 and self.ub == 1 :
                model.binary_var('x'+str(j))
            else:
                model.integer_var(lowerbound = self.lb, upperbound = self.ub, name = 'x'+str(j))

        # add linear constraints
        for i in range(self.m):
            model.linear_constraint(linear=self.A[i], sense='<=', rhs=self.b[i], name='A'+str(i)+'le'+'b'+str(i))

        # add linear objective function
        model.minimize(linear = self.c)
        self.model = model
