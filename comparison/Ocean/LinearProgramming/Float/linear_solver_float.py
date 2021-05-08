"""
Created on Thu Dec 31 14:01:31 2020

@author: tom

The formulas are taken from the paper Floating-Point Calculations on a Quantum Annealer:
    Division and Matrix Inversion.

Many thanks to M. Rogers and R. Singleton Jr.

"""

import dimod
import numpy as np
from dwave.system import DWaveCliqueSampler


class LinearSolverFloat:
    def __init__(self, M, Y=None, R=4, num_reads=500, min_occurrence=5):
        self.M = M
        self.N = len(M)
        self.R = R
        self.num_logic_qubits = self.N * self.R
        self.num_reads = num_reads
        self.min_occurrence = min_occurrence
        if Y is not None:
            self.Y = Y
        else:
            self.bqm = None
        self.sampler = DWaveCliqueSampler()
        self.sampleset = None
        self.solutions = None

    def getY(self):
        return self._Y

    def setY(self, value):
        self._Y = value
        self.calculateIsing()

    Y = property(getY, setY)

    def calculateIsing(self):
        L = {l: self._a_l(l) for l in range(0, self.num_logic_qubits)}
        Q = {
            (l, m): self._b_l_m(l, m)
            for l in range(0, self.num_logic_qubits)
            for m in range(l, self.num_logic_qubits)
        }
        offset = np.inner(self.Y, self.Y)
        self.bqm = dimod.BinaryQuadraticModel(L, Q, offset, dimod.Vartype.BINARY)

    def _a_l(self, l):
        # eq 3.25
        i_l = int(np.floor(l / self.R))
        r_l_pot = 2 ** (-(l % self.R))
        return (
            4
            * r_l_pot
            * sum(
                self.M[k, i_l] * (r_l_pot * self.M[k, i_l] - self.Y[k] - sum(self.M[k]))
                for k in range(0, self.N)
            )
        )

    def _b_l_m(self, l, m):
        # eq 3.26
        r_l = l % self.R
        r_m = m % self.R
        i_l = int(np.floor(l / self.R))
        i_m = int(np.floor(m / self.R))
        return (
            4 * 2 ** (-(r_l + r_m)) * sum(self.M[k, i_l] * self.M[k, i_m] for k in range(0, self.N))
        )

    def _delta(self, i, l):
        return int(i == np.floor(l / self.R))

    def _get_x_i(self, i, q):
        return (
            2 * sum(2 ** (-(l % self.R)) * q[l] * self._delta(i, l) for l in range(0, len(q))) - 1
        )

    def sample(self):
        if self.Y is None:
            print(f"{self.Y =}")
            return
        self.sampleset = self.sampler.sample(self.bqm, num_reads=self.num_reads)
        self.solutions = [
            (np.array([self._get_x_i(i, q) for i in range(0, self.N)]), energy, num_occurrences)
            for q, energy, num_occurrences in self.sampleset.data(
                fields=["sample", "energy", "num_occurrences"]
            )
            if num_occurrences >= self.min_occurrence
        ]
        return self.sampleset

    def getSolutions(self, min_occurrence):
        if self.sampleset is None:
            return None
        return [
            (np.array([self._get_x_i(i, q) for i in range(0, self.N)]), energy, num_occurrences)
            for q, energy, num_occurrences in self.sampleset.data(
                fields=["sample", "energy", "num_occurrences"]
            )
            if num_occurrences >= min_occurrence
        ]
