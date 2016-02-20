Saul Toscano-Palmerin

Ph.D. Student, Cornell University 


**Bayesian Global Optimization (BGO)**


**BGO** is a Bayesian Global Optimization framework written in Python,
developed by Saul Toscano-Palmerin. The library
includes four different algorithms: SBO, KG, EI and PI.

**SBO** is a new algorithm proposed by [Toscano-Palmerin and
Frazier][tf]. It's used for simulation optimization problems, and for global 
Bayesian optimization. In general, the objective functions are
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

For one example optimized using this library, see [CitiBike][ref].
([Intro and annotated source][annotated]) 

![citi bike simulation](https://github.com/toscanosaul/BGO/blob/master/CitiBike/animation.gif)

[annotated]: https://github.com/toscanosaul/BGO/blob/master/CitiBike/citiBike.pdf 
[ref]: https://github.com/toscanosaul/BGO/tree/master/CitiBike
