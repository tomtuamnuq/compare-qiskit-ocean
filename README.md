# compare-qiskit-ocean
I try to find my way into quantum computing by a comparison of ocean and qiskit framework. In particular, I will test Linear and Quadratic Programs. 

Results of so far done tests are summarized below. It is obvious, that D-Wave solvers outperform current gate model QPUs.
There are multiple Jupyter Notebooks which show embedding on D-Wave systems as well as graph visuals.
For Qiskit, topologies of real IBM Q devices as well as QAOA circuits and outcomes on these devices are included.

The class RandomQubo constructs qubos of the form:
min 1/2 x Q x + cx 
x is a binary and c an integer vector of size n.
Q is an integer matrix of size n x n.

More complex problems can be constructed as follows:

In random_lp package is a class RandomQP which is used to test QAOA on IBMQ and Quantum Annealing on DWave Leap.
The class constructs quadratic programs of the form:
min 1/2 x Q x + cx 
such that Ax <= b

The class RandomLP constructs programs of the form:
min cx 
such that Ax <= b

x and c are integer vectors of size n.
b sets m integer constraints.
A is an integer matrix of size m x n.
Q is an integer matrix of size n x n.

The problems can be constructed as fully connected or sparse.

The Qiskit framework is used to derive QUBOs from the QPs.
To set DWave sampler as solver in Minimum Eigen Optimizer the dwave_qiskit_plugin is used.

In both Ocean and Qiskit folders are Jupyter Notes with tests:
comparison/Qiskit/LinearProgramming/QAOA/
comparison/Ocean/LinearProgramming/DWave_Qiskit_Plugin_Test/

Results:
For complete logs and output see RESULTS folders in the according subdirs.

Random Qubo
    Dense

        CPLEX classical optimizer needs a long time to optimize large problems with 90 or 100 vars

        DWave pure quantum with 100 qubits and clique embedding almost optimal

        QAOA noise model Mumbai simulation up to 12 qubits optimal, up to 27 qubits done
    
    Sparse

        DWave pure quantum with 525 qubits and auto embedding found almost optimal solution

        QAOA noise model Mumbai simulation up to 12 qubits successful but not optimal

Random LP
    Dense

        DWave Hybrid up to 200 qubits near optimal
        Dwave Hybrid up to 218 qubits found solution but not optimal
        DWave pure quantum with 29 qubits and clique embedding near optimal

        QAOA noise free simulation up to 22 qubits near optimal
    
    Sparse

        Dwave Hybrid up to 217 qubits with non optimal solutions
        DWave pure quantum with 100 qubits and auto embedding found solution but not optimal

        QAOA noise free simulation up to

Random QP
    Dense

        DWave Hybrid up to  102 qubits near optimal
        Dwave Hybrid up to 185 qubits found solution but not optimal
        calculation gets stuck on problem with 225 qubits

        DWave pure quantum with 25 qubits and clique embedding near optimal
        DWave pure quantum with 36 qubits and auto embedding found solution but not optimal

        QAOA noise free simulation up to 17 qubits near optimal
    
    Sparse

        Dwave Hybrid up to  qubits with non optimal solutions
        DWave pure quantum with  qubits and auto embedding found solution but not optimal

        QAOA noise free simulation up to 13 qubits optimal
        QAOA up to 17 qubits near optimal





    














# Acknowledgements:

I acknowledge the use of DWave Systems services for this work. 
I acknowledge the use of IBM Quantum services for this work. 

The views expressed are those of the author, and do not reflect the official policy or position.

I acknowledge the work of Yonghae Lee, Jaewoo Joo & Soojoon Lee under the Creative
Commons License: http://creativecommons.org/licenses/by/4.0/
I used the paper "Hybrid quantum linear equation algorithm and its experimental test on IBM Quantum Experience" 
to derive the circuits for HHL Algorithm tests.


I acknowledge the work of M. Rogers and R. Singleton Jr. in the paper "Floating-Point Calculations on a Quantum Annealer: Division and Matrix Inversion".

I have used one instance of qplib as Leap max test:
Fabio Furini, Emiliano Traversi, Pietro Belotti, Antonio Frangioni, Ambros Gleixner, Nick Gould, Leo Liberti, Andrea Lodi, Ruth Misener, Hans Mittelmann, Nikolaos Sahinidis, Stefan Vigerske, and Angelika Wiegele. QPLIB: A Library of Quadratic Programming Instances, Mathematical Programming Computation, 2018, DOI:10.1007/s12532-018-0147-4

Some linear program problems found at https://people.math.sc.edu/Burkardt/datasets/mps/mps.html (distributed under the GNU LGPL license).

