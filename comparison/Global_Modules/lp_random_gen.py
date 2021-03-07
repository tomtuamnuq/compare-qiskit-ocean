"""
Created on Sun Jan 31 09:58:32 2021

@author: tom
"""

import numpy as np
from qiskit.optimization import QuadraticProgram

varname = lambda k,j : 'x'+str(k)+"_"+str(j)

class random_lp_int:
    def __init__(self,num_constr,num_vars,lowerbound,upperbound):
        self.m = num_constr 
        self.n = num_vars 
        self.lb = lowerbound
        self.ub = upperbound
        
    def createLP(self, name, multiple = 1, A_lb = -10, A_ub = 10, c_lb = -1, c_ub = 1):
        self.name = name
        self.model = QuadraticProgram(self.name)
        self.multiple = multiple
        objective = np.array([])
        for k in range(multiple):
            # add variables
            for j in range(self.n):
                if self.lb == 0 and self.ub == 1 :
                    self.model.binary_var(varname(k,j))
                else:
                    self.model.integer_var(lowerbound = self.lb, upperbound = self.ub, name = varname(k,j))
            
            self._create_data(self.ub+1, A_lb , A_ub , c_lb , c_ub )
            objective = np.append(objective,self.c)
            self._add_constrs(k)
            
        # add linear objective function
        
        self.model.minimize(linear = objective)
        return self.model

    
    def _create_data(self, d_ub , A_lb, A_ub , c_lb, c_ub ):
        self.x = np.random.randint(self.lb,self.ub+1,size = self.n)
        self.d = np.random.randint(0,d_ub+1,size = self.m)
        self.A = np.random.randint(A_lb,A_ub+1,size = (self.m,self.n))
        self.b = self.A.dot(self.x) + self.d
        self.c = np.random.randint(c_lb,c_ub+1,size = self.n)
    def _add_constrs(self,k):    

        # add linear constraints
        for i in range(self.m):
            linear = {}
            for j in range(self.n):
                linear[varname(k,j)] = self.A[i,j]
            self.model.linear_constraint(linear=linear, sense='<=', rhs=self.b[i], name='A'+str(k)+"_le"+'b'+str(i))
    

        

