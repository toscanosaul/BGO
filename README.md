Saul Toscano-Palmerin

Ph.D. Student, Cornell University 


**SOFTWARE**

Bayesian Global Optimization (BGO) includes four algorithms: 
Stratified Bayesian Optimization(SBO), Knowledge Gradient (KG), 
Expected Improvement (EI) and Probability Improvement (PI).

**BGO** is a Bayesian Global Optimization framework written in Python,
developed by Saul Toscano-Palmerin. The library
includes four different algorithms: SBO, KG, EI and PI.

**SBO** is a new algorithm proposed by [Toscano-Palmerin and
Frazier][tf]. It's used for simulation optimization problems.
Specifically, it's used for the global optimization of the expectation
of continuos functions (respect to some metric), which depend on big
random vectors. We suppose that a small number of random variables have
a much stronger effect on the variability. In general, the functions are
time-consuming to evaluate, and the derivatives are unavailable.

[tf]: http://arxiv.org/pdf/1602.02338.pdf

In mathematical notation, we want to solve the problem:

max\_{x} E[f(x,w,z)]

where the expectation is over w and z, and w is the random vector that
have a much stronger effect on the variability of f.

We should emphasize that this class of algorithms can be used to solve a
broader class of problems. In particular, we might be only interested in
solving the problem max\_{x} G(x).

This library is still in development and soon we are going to include
examples where we need to decide how to optimally allocate
computational/experimental effort across information sources, to
optimize functions.

See CitiBike directory for one example otimized using this library. 
In this example, we consider a queuing simulation based on New York City's Bike system, in which
system users may remove an available bike from a station at one location
within the city, and ride it to a station with an available dock in some
other location within the city. The optimization problem that we
consider is the allocation of a constrained number of bikes (6000) to
available docks within the city at the start of rush hour, so as to
minimize, in simulation, the expected number of potential trips in which
the rider could not find an available bike at their preferred
origination station, or could not find an available dock at their
preferred destination station. We call such trips "negatively affected
trips".

