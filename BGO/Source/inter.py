#!/usr/bin/env python

"""
We define classes needed to run the SBO algorithm:

-objective: This class consits of functions and elements related to the
            objective function f(x,w,z) and distribution of w.
            
-Miscellaneous: Consists of path where the results are saved; the random
                seed used to run the program; and specifies if the program
                is run in parallel.

-opt: This class consists of functions and variables used in the gradient ascent
      method when optimizing the VOI and the expectation of the GP. This functions
      allow us to define the termination condition for gradient ascent. Moreover,
      we may want to transform the space of x to a more convenient space, e.g.
      dimension reduction.

-data: This class consists of all the history.

"""

import numpy  as np
import os
import multiprocessing as mp

class objective:
    def __init__(self, fobj,dimSeparation,noisyF,numberEstimateF,SampleFromX,
                 simulatorW,estimationObjective):
        """
        Args:
            -fobj: The simulator or objective function.
            -dimSeparation: Dimension of w.
            -noisyF: Estimator of the conditional expectation given w,
                     F(x,w)=E[f(x,w,z)|w].
                     Its arguments are:
                        -(x,w): Point where F is evaluated.
                        -N: number of samples to estimate F
                     Its output is a couple with:
                        -Estimator of F.
                        -Estimator of the variance of the output.
            -numberEstimateF: Number of samples used to estimate F.
            -sampleFromX: Chooses a point x at random.
                          Its argument is:
                            -N: Number of points chosen.
            -simulatorW: Simulates a random vector w.
                         Its argument is:
                            -N: Number of simulations taken.
            -estimationObjective: Estimates the expectation of fobj.
                                  Its arguments are:
                                     -x: Point where G is evaluated.
                                     -N: number of samples to estimate G
                                  Its output is a couple with:
                                     -Estimator of G.
                                     -Estimator of the variance of the output.
        """
        self.fobj=fobj
        self.dimSeparation=dimSeparation
        self.noisyF=noisyF
        self.numberEstimateF=numberEstimateF
        self.sampleFromX=SampleFromX
        self.simulatorW=simulatorW
        self.estimationObjective=estimationObjective

class Miscellaneous:
    def __init__(self,randomSeed,parallel,create=True,nF=0,tP=0,ALG="SBO"):
        """
        Args:
            -randomSeed: Random seed used to run the problem. Only needed for the
                         name of the files with the results.
            -parallel: True if we want to run the multistart gradient ascent algorithm
                       and train the kernel of the GP in parallel; otherwise, it's false. 
            -folder: Path where the files with the results are saved. 
            -create: True if we want to create new files for the results; it's false
                     otherwise. If we want to add more results to our previos results,
                     this variable should be false.
            -nF: Number of samples to estimate the information source
            -tP: Number of training points
            -ALG: Algorithm that is used
        """
        self.rs=randomSeed
        self.parallel=parallel
        
        nameDirectory="Results"+'%d'%nF+"AveragingSamples"+'%d'%tP+"TrainingPoints"
        folder=os.path.join(nameDirectory,ALG)
        self.folder=folder
        self.create=create

class opt:
    def __init__(self,numberParallel,dimXsteepest,transformationDomainX,
                 transformationDomainW,projectGradient,functionGradientAscentVn,
                 functionGradientAscentAn,functionConditionOpt,xtol):
        """
        Args:
        -numberParallel: Number of starting points for the multistart gradient
                         ascent algorithm.
        -dimXsteepest: Dimension of x when the VOI and a_{n} are optimized. We
                       may want to reduce the dimension of the original problem.
        -transformationDomainX: Transforms the point x given by the steepest ascent
                                method to the right domain of x.
                                Its arugment its:
                                    -x: The point to be transformed
        -transformationDomainW: Transforms the point w given by the steepest ascent
                                method to the right domain of w.
                                Its argument is:
                                  -w: The point to be tranformed.
        -projectGradient: Project a point x to the domain of the problem at each step
                          of the gradient ascent method if needed.
                          Its argument is:
                            -x: The point that is projected.
        -functionGradientAscentVn: Function used for the gradient ascent method. It
                                   evaluates the VOI, when grad and onlyGradient are
                                   False; it evaluates the VOI and computes its
                                   derivative when grad is True and onlyGradient is False,
                                   and computes only its gradient when gradient and
                                   onlyGradient are both True.
                                   Its arguments are:
                                    -x: VOI is evaluated at x.
                                    -grad: True if we want to compute the gradient;
                                           False otherwise.
                                    -i: Iteration of the SBO algorithm.
                                    -L: Cholesky decomposition of the matrix A,
                                        where A is the covariance matrix of the
                                        past obsevations (x,w).
                                    -Bfunc: Computes B(x,XW)=\int\Sigma_{0}(x,w,
                                                     XW[0:n1],XW[n1:n1+n2])dp(w).
                                    -temp2: temp2=inv(L)*B.T, where B is a matrix
                                            such that B(i,j) is
                                            \int\Sigma_{0}(x_{i},w,x_{j},w_{j})dp(w)
                                            where points x_{p} is a point of the
                                            discretization of the space of x; and
                                            (x_{j},w_{j}) is a past observation.
                                    -a: Vector of the means of the GP on G. The
                                        means are evaluated on the
                                        discretization of the space of x.
                                    -VOI: VOI object
                                    -kern: kernel
                                    -XW: Past observations
                                    -scratch: matrix where scratch[i,:] is the
                                              solution of the linear system
                                              Ly=B[j,:].transpose()
                                    -onlyGradient: True if we only want to compute
                                                    the gradient; False otherwise.
        -functionGradientAscentAn: Function used for the gradient ascent method. It evaluates
                                   a_{n}, when grad and onlyGradient are False; it evaluates
                                   a_{n} and computes its derivative when grad is True and
                                   onlyGradient is False, and computes only its gradient when
                                   gradient and onlyGradient are both True.
                                   Its arguments are:
                                    x: a_{i} is evaluated at x.
                                    grad: True if we want to compute the gradient; False
                                          otherwise.
                                    i: Iteration of the SBO algorithm.
                                    L: Cholesky decomposition of the matrix A, where A is
                                       the covariance matrix of the past obsevations (x,w).
                                    dataObj: Data object.
                                    stat: Statistical object.
                                    onlyGradient: True if we only want to compute the gradient;
                                                  False otherwise.
                                    logproductExpectations: Vector with the logarithm of the
                                                            product of the expectations of
                                                            np.exp(-alpha2[j]*((z-W[i,j])**2))
                                                            where W[i,:] is a point in the history.
                                                            Only when SK is used.
        -functionConditionOpt: Gives the stopping rule for the steepest ascent method, e.g. the
                               function could be the Euclidean norm.
                               Its arguments is:
                                -x: Point where the condition is evaluated.
        -xtol: Tolerance of x for the convergence of the steepest ascent method.
        """
        self.numberParallel=numberParallel
        self.dimXsteepest=dimXsteepest
        self.transformationDomainX=transformationDomainX
        self.transformationDomainW=transformationDomainW
        self.projectGradient=projectGradient
        self.functionGradientAscentVn=functionGradientAscentVn
        self.functionGradientAscentAn=functionGradientAscentAn
        self.functionConditionOpt=functionConditionOpt
        self.xtol=xtol

class data:
    def __init__(self,Xhist,yHist,varHist):
        """
        Arguments:
            -Xhist: Past vectors (x,w).
            -yHist: Past outputs.
            -varHist: Past variances of the outputs.
        """
        self.Xhist=Xhist
        self.yHist=yHist
        self.varHist=varHist
        
    def copyData(self):
        Xcopy=self.Xhist.copy()
        ycopy=self.yHist.copy()
        varcopy=self.varHist.copy()
        temp=data(Xcopy,ycopy,varcopy)
        return temp
    
    def getTrainingDataKG(self,trainingPoints,noisyG,numberSamplesForG,parallel):
        Xtrain=self.Xhist
        yTrain=np.zeros([0,1])
        NoiseTrain=np.zeros(0)
        if parallel:
            jobs = []
            pool = mp.Pool()
            for i in xrange(trainingPoints):
                job = pool.apply_async(noisyG,(Xtrain[i,:],numberSamplesForG))
                jobs.append(job)
            
            pool.close()  # signal that no more data coming in
            pool.join()  # wait for all the tasks to complete
            for j in range(trainingPoints):
                temp=jobs[j].get()
                yTrain=np.vstack([yTrain,temp[0]])
                NoiseTrain=np.append(NoiseTrain,temp[1])
        else:
            for i in xrange(trainingPoints):
                temp=noisyG(Xtrain[i,:],numberSamplesForG)
                yTrain=np.vstack([yTrain,temp[0]])
                NoiseTrain=np.append(NoiseTrain,temp[1])
        self.yHist=yTrain
        self.varHist=NoiseTrain
        
    def getTrainingDataSBO(self,trainingPoints,noisyF,numberSamplesForF,parallel):
        XWtrain=self.Xhist
        yTrain=np.zeros([0,1])
        NoiseTrain=np.zeros(0)
        
        if parallel:
            jobs = []
            pool = mp.Pool()
            for i in xrange(trainingPoints):
                job = pool.apply_async(noisyF,(XWtrain[i:i+1,:],numberSamplesForF))
                jobs.append(job)
            pool.close()  
            pool.join()  
            for j in range(trainingPoints):
                temp=jobs[j].get()
                yTrain=np.vstack([yTrain,temp[0]])
                NoiseTrain=np.append(NoiseTrain,temp[1])
        else:
            for i in xrange(trainingPoints):
                temp=noisyF(XWtrain[i:i+1,:],numberSamplesForF)
                yTrain=np.vstack([yTrain,temp[0]])
                NoiseTrain=np.append(NoiseTrain,temp[1])
        self.yHist=yTrain
        self.varHist=NoiseTrain