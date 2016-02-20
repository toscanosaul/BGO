Saul Toscano-Palmerin

Ph.D. Student, Cornell University 


**SOFTWARE**

Bayesian Global Optimization (BGO): Stratified Bayesian Optimization
(SBO), Knowledge Gradient (KG), Expected Improvement (EI) and
Probability Improvement (PI).

**DOWNLOAD!!!**Code in Python (in preparation): [BGO.](BGO.zip)

**BGO** is a Bayesian Global Optimization framework written in Python,
developed by Saul Toscano-Palmerin and Peter I. Frazier. The library
includes four different algorithms: SBO, KG, EI and PI.

**SBO** is a new algorithm proposed by [Toscano-Palmerin and
Frazier][tf]. It's used for simulation optimization problems.
Specifically, it's used for the global optimization of the expectation
of continuos functions (respect to some metric), which depend on big
random vectors. We suppose that a small number of random variables have
a much stronger effect on the variability. In general, the functions are
time-consuming to evaluate, and the derivatives are unavailable.

[tf]: http://toscanosaul.github.io/saul/SBO.pdf

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

To use any of the previous algorithms is necessary to define 6 objects:

**Objobj**: Objective object (See inter.py).

**miscObj**: Miscellaneous object (See inter.py).

**VOIobj:** Value of Information function object (See VOI.py).

**optObj:** Opt object (See inter.py).

**statObj**: Statistical object (See stat.py).

**dataObj:** Data object (See inter.py).

See the citiBikeExample files for examples. In this example, we consider
a queuing simulation based on New York City's Bike system, in which
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

![alt tag](http://toscanosaul.github.io/saul/map.png) **Bike stations in NYC.**

To run SBO on this example, we only need to write

**python citiBikeExample.py randomSeed nT nF nIterations Parallel
nRestarts**

where nT: number of training data; nF: number of samples to estimate F;
nIterations: number of iterations; Parallel: T if the code is run in
paralllel; F otherwise. On OS, the code can't be run in parallel because
there are problems with the function dot in numpy.; nRestarts: Number
that the optimization algorithms used to optimize the VOI and the answer
are restart. This parameters should be given only when Parallel is T.
(e.g. python citiBikeExample.py 3 7 15 2 T 10).

To run the KG on this example, we only need to write

**python citiBikeExampleKG.py randomSeed nT nF nIterations Parallel
nRestarts**

where the parameters are defined above.

The output of the algorithm is saved in the directory
Results15AveragingSamples7TrainingPoints/KG/randomSeedrun or
Results15AveragingSamples7TrainingPoints/SBO/randomSeedrun,
respectively. These directories contain 10 files: hyperparameters.txt
(hyperparameters of the kernel), optAngrad.txt (gradients of the mean of
the GP at the values found by the algorithm), optimalSolutions.txt
(proposed solutions by the algorithm),optimalValues.txt (values of the
proposed solutions with their variances), optVOIgrad.txt (gradients of
the VOI at the values found by the algorithm), varHist.txt (variances of
the past observations), XHist.txt (past observations), yhist.txt (past
observations).


