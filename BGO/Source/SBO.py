"""
Stratified Bayesian Optimization (SBO) Algorithm.

This is a new algorithm proposed by [Toscano-Palmerin and Frazier][tf].
It's used for simulation optimization problems. Specifically, it's used
for the global optimization of the expectation of continuos functions
(respect to some metric), which depend on big random vectors. We suppose
that a small number of random variables have a much stronger effect on the
variability. In general, the functions are time-consuming to evaluate, and
the derivatives are unavailable.

In mathematical notation, we want to solve the problem:

	max_{x} E[f(x,w,z)]

where the expectation is over w and z, and w is the random vector that have
a much stronger effect on the variability of f.

[tf]: http://toscanosaul.github.io/saul/SBO.pdf

This class takes six class arguments (for details, see [tf]):


(*) Data class. This class consists of all the history.

-Xhist: Past vectors (x,w).
-yHist: Past outputs.
-varHist: Past variances of the outputs.


(*) Objective class. This class consits of functions and elements related to the
objective function f(x,w,z) and distribution of w.

-fobj: The simulator or objective function.
-dimSeparation: Dimension of w.
-noisyF: Estimator of the conditional expectation given w, F(x,w)=E[f(x,w,z)|w].
-numberEstimateF: Observations used to estimate F.
-sampleFromX: Chooses a point x at random.
-simulatorW: Simulates a random vector w.
-estimationObjective: Estimates the expectation of fobj. 



(*) Statistical Class. This class consits of all the functions and elements
of the statistical model used.

-kernel: Kernel object for the GP on F.
-dimensionKernel: Dimension of the kernel.
-scaledAlpha: Parameter to scale the alpha parameters of the kernel,
	      alpha/(scaledAlpha^{2})
-B(x,XW,n1,n2,logproductExpectations=None): Computes
	\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w),
	where x can be a vector.
-numberTrainingData: Number of training data.
-computeLogProductExpectationsForAn: Computes the vector with the logarithm
		of the product of the expectations of
		np.exp(-alpha2[j]*((z-W[i,j])**2))
		where W[i,:] is a point in the history.
-gradXBforAn: Computes the gradients with respect to x of
	     B(x,i)=\int\Sigma_{0}(x,w,x_{i},w_{i})dp(w),
	     where (x_{i},w_{i}) is a point in the history.




(*) Optimization class. This class consists of functions and variables used in the gradient ascent
method when optimizing the VOI and the expectation of the GP. This functions allow us to define
the termination condition for gradient ascent. Moreover, we may want to transform the space of x
to a more convenient space, e.g. dimension reduction.

-numberParallel: Number of starting points for the multistart gradient ascent algorithm.
-dimXsteepest: Dimension of x when the VOI and a_{n} are optimized. We may want to reduce
	       the dimension of the original problem.
-transformationDomainX: Transforms the point x given by the steepest ascent method to the right domain
			of x.
-transformationDomainW: Transforms the point w given by the steepest ascent method to the right domain
			of w.
-projectGradient: Project a point x to the domain of the problem at each step of the gradient
		  ascent method if needed.
-functionGradientAscentVn: Function used for the gradient ascent method. It evaluates the VOI,
			   when grad and onlyGradient are False; it evaluates the VOI and
			   computes its derivative when grad is True and onlyGradient is False,
			   and computes only its gradient when gradient and onlyGradient are both
			   True. 
-functionGradientAscentAn: Function used for the gradient ascent method. It evaluates a_{n},
			   when grad and onlyGradient are False; it evaluates a_{n} and
			   computes its derivative when grad is True and onlyGradient is False,
			   and computes only its gradient when gradient and onlyGradient are both
			   True.
-functionConditionOpt: Gives the stopping rule for the steepest ascent method, e.g. the function
			could be the Euclidean norm. 
-xtol: Tolerance of x for the convergence of the steepest ascent method.

                 

(*) VOI class. This class consits of the functions and variables related to the VOI.

-pointsApproximation: Points of the discretization to compute the VOI.
-gradXWSigmaOfunc: Computes the gradient of Sigma_{0}, which is the covariance of the Gaussian
		   Process on F.
-gradXBfunc: Computes the gradients with respect to x_{n+1} of
	     B(x_{p},n+1)=\int\Sigma_{0}(x_{p},w,x_{n+1},w_{n+1})dp(w),
	     where x_{p} is a point in the discretization of the domain of x.
-gradWBfunc: Computes the gradients with respect to w_{n+1} of
	     B(x_{p},n+1)=\int\Sigma_{0}(x_{p},w,x_{n+1},w_{n+1})dp(w),
	     where x_{p} is a point in the discretization of the domain of x.


(*) Miscellaneous class. Consists of path where the results are saved; the random
    seed used to run the program; and specifies if the program is run in parallel.

-randomSeed: Random seed used to run the problem. Only needed for the name of the
	     files with the results.
-parallel: True if we want to run the multistart gradient ascent algorithm and
	   train the kernel of the GP in parallel; otherwise, it's false. 
-folderContainerResults: Path where the files with the results are saved. 
-createNewFiles: True if we want to create new files for the results; it's false
		 otherwise. If we want to add more results to our previos results,
		 this variable should be false.

"""
import numpy as np
from math import *
import sys
from . import optimization as op
import multiprocessing as mp
import os
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
from . import misc
from matplotlib import pyplot as plt
from numpy import multiply
from numpy.linalg import inv
from . import stat
from . import VOI
from . import files as fl
from AffineBreakPoints import *
from scipy.stats import norm
import pylab as plb
from scipy import linalg


class SBO:
    def __init__(self, Objobj,miscObj,VOIobj,optObj,statObj,dataObj):
        """
        Class to use the Stratified Bayesian Optimization on a specific
	problem specified by the Objobj object.
	
        Args:
	    Objobj: Objective object (See InterfaceSBO).
	    miscObj: Miscellaneous object (See InterfaceSBO).
	    VOIobj: Value of Information function object (See VOIGeneral).
	    optObj: Opt object (See InterfaceSBO).
	    statObj: Statistical object (See statGeneral).
	    dataObj: Data object (See InterfaceSBO).
	    plots: Ture if we want the plots of a_{n} and VOI. Only for the analytic example
        """
	self.dataObj=dataObj
	self.stat=statObj
	self._VOI=VOIobj
	self.opt=optObj
	self.miscObj=miscObj
	self.Obj=Objobj
	
	self._n1=Objobj.dimSeparation
	self._dimW=self.stat.n2
	self.numberTraining=statObj._numberTraining
	
	#Number of B(x_{p},i) computed for all x_{p} in the discretization
	self.histSaved=0 
	#Matrix with the computations B(x_{p},i)
	self.Bhist=np.zeros((VOIobj.sizeDiscretization,0))
	
	self.optRuns=[]
        self.optPointsArray=[]
	self._solutions=[]
        self._valOpt=[]
	
	self.path=os.path.join(miscObj.folder,'%d'%self.miscObj.rs+"run")
	
	self.parameters=None
	if not os.path.exists(self.path):
	    os.makedirs(self.path)

    def SBOAlg(self,m,nRepeat=10,Train=True,plots=False,**kwargs):
        """
        Run the SBO algorithm until m steps are taken.
	
        Args:
	    m: Number of stepes of the algorithm.
	    nRepeat: Number of random restarts to optimize the hyperparameters.
	    Train: True if we want to train the kernel; False otherwise.
        """
	if self.miscObj.create: #Create files for the results
	    fl.createNewFilesFunc(self.path,self.miscObj.rs) 
	fl.writeTraining(self) #Write training data
        if Train is True:
            self.trainModel(numStarts=nRepeat,**kwargs) #Train model
        points=self._VOI._points
        for i in range(m):
            print i
	    #Optimize VOI
	    if plots:
		tempN=self.numberTraining+i
		At=self.stat._k.A(self.dataObj.Xhist[0:tempN,:],noise=self.dataObj.varHist[0:tempN])
		Lt=np.linalg.cholesky(At)
		gridX=self._VOI._points
		tempX=self.dataObj.Xhist[0:tempN,1:self._dimW+1]
		logProd=self.stat.computeLogProductExpectationsForAn(tempX,
                                                         tempN,self.stat._k)

		self.stat.plotAn(i,Lt,gridX,self.path,self.dataObj,self.dataObj.Xhist[:,0:1],
				 self.dataObj.Xhist[:,1:2],self.stat._k,self.stat.B,logProd)

		Bhist=np.zeros((self._VOI.sizeDiscretization,tempN))
		for j in xrange(0,tempN):
		    temp=self.stat.B(self._VOI._points,self.dataObj.Xhist[j,:],self._n1,
				     self._dimW,self.stat._k)
		    Bhist[:,j]=temp
		muStartt=self.stat._k.mu
		yt=self.dataObj.yHist
		temp2t=linalg.solve_triangular(Lt,(Bhist).T,lower=True)
		temp1t=linalg.solve_triangular(Lt,np.array(yt)-muStartt,lower=True)
		a=muStartt+np.dot(temp2t.T,temp1t)
		m2=self._VOI._points.shape[0]
		scratch=np.zeros((m2,tempN))
		

		for j in xrange(m2):
		    scratch[j,:]=linalg.solve_triangular(Lt,Bhist[j,:].transpose(),lower=True)
		self._VOI.plotVOI(i,Lt,self.path,self.dataObj,temp2t,a,scratch,
				  self.stat._k,self.dataObj.Xhist,self.stat.B,m2,
				  self._VOI._points)
		
		
		####mu_n
	
		self.stat.plotmuN(i,Lt,temp1t,self.stat._k,self.dataObj.Xhist[:,0:1],
				  self.dataObj.Xhist[:,1:2],muStartt,
				  self._VOI._points,m2,self.path)
		
	    if self.miscObj.parallel:
		self.optVOIParal(i,self.opt.numberParallel) 
	    else:
		self.optVOInoParal(i)
            print i
	    #Otimize a_{n}
	    if self.miscObj.parallel:
		self.optAnParal(i,self.opt.numberParallel)
	    else:
		self.optAnnoParal(i)
            print i
	#Optimize a_{n}
	if self.miscObj.parallel:
	    self.optAnParal(m,self.opt.numberParallel)
	else:
	    self.optAnnoParal(i)

    def optimizeVOI(self,start, i,L,temp2,a,B,scratch):
	"""
        Optimize the value of information using gradient ascent.
	
        Args:
	    start: Starting point for the gradient ascent method.
	    i: Iteration of the algorithm.
            L: Cholesky decomposition of the matrix A, where A is the covariance
               matrix of the past obsevations (x,w).
	    B: Matrix such that B(i,j) is \int\Sigma_{0}(x_{i},w,x_{j},w_{j})dp(w)
	       where points x_{p} is a point of the discretization of
               the space of x; and (x_{j},w_{j}) is a past observation.
            temp2: temp2=inv(L)*B.T.
            a: Vector of the means of the GP on g(x)=E(f(x,w,z)). The means are evaluated on the
               discretization of the space of x.
	    scratch: matrix where scratch[i,:] is the solution of the linear system
                     Ly=B[j,:].transpose() (See above for the definition of B and L)
        """
	
        def g(x,grad,onlyGradient=False):
            return self.opt.functionGradientAscentVn(x,i=i,VOI=self._VOI,L=L,temp2=temp2,a=a,
						 scratch=scratch,onlyGradient=onlyGradient,
						 kern=self.stat._k,XW=self.dataObj.Xhist,
						 Bfunc=self.stat.B,grad=grad)

	if self.opt.MethodVn=="SLSQP":
	    opt=op.SLSP(start)
	    def g1(x):
		return -1.0*g(x,grad=False)
	    
	    def dg(x):
		return -1.0*g(x,grad=True,onlyGradient=True)
	    
	    cons=self.opt.consVn
	    opt.run(f=g1,df=dg,cons=cons)
	else:
	    opt=op.OptSteepestDescent(n1=self.opt.dimXsteepestVn,projectGradient=self.opt.projectGradient,
				      stopFunction=self.opt.functionConditionOpt,xStart=start,
				      xtol=self.opt.xtol)
	    opt.run(f=g)


        self.optRuns.append(opt)

    def getParametersOptVoi(self,i):
	"""
        Returns a dictionary with i,L,temp2,a,B, scratch. 
	This dictionary is used to run optimizeVOI.
	
        Args:
	    i: Iteration of the algorithm.
        """
	n1=self._n1
	n2=self._dimW
	tempN=self.numberTraining+i
	A=self.stat._k.A(self.dataObj.Xhist[0:tempN,:],noise=self.dataObj.varHist[0:tempN])
	L=np.linalg.cholesky(A)
	m=self._VOI._points.shape[0]
	for j in xrange(self.histSaved,tempN):
	    temp=self.stat.B(self._VOI._points,self.dataObj.Xhist[j,:],self._n1,
			     self._dimW,self.stat._k) 
	    self.Bhist=np.concatenate((self.Bhist,temp.reshape((m,1))),1)
	    self.histSaved+=1
	muStart=self.stat._k.mu
	y=self.dataObj.yHist
	temp2=linalg.solve_triangular(L,(self.Bhist).T,lower=True)
	temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True)
	a=muStart+np.dot(temp2.T,temp1)
	
	scratch=np.zeros((m,tempN))
	for j in xrange(m):
	    scratch[j,:]=linalg.solve_triangular(L,self.Bhist[j,:].transpose(),lower=True)
	args2={}
        args2['i']=i
	args2['L']=L
	args2['temp2']=temp2
	args2['a']=a
	args2['B']=self.Bhist
	args2['scratch']=scratch
	return args2

    def optVOInoParal(self,i):
	"""
	Runs the single-start gradient ascent method to optimize the VOI.
	
        Args:
	    i: Iteration of the algorithm.
        """
	n1=self._n1
	n2=self._dimW
	Xst=self.Obj.sampleFromXVn(1)
	wSt=self.Obj.simulatorW(1)
	x1=Xst[0:0+1,:]
	w1=wSt[0:0+1,:]
	tempN=self.numberTraining+i
	st=np.concatenate((x1,w1),1)
	args2=self.getParametersOptVoi(i)
    #    args2['start']=st
	self.optRuns.append(misc.VOIOptWrapper(self,st,**args2))
	fl.writeNewPointSBO(self,self.optRuns[0])

    def optVOIParal(self,i,nStart,numProcesses=None):
	"""
	Runs in parallel the multi-start gradient ascent method
	to optimize the VOI.
	It restarts the gradient ascent method nStart times.
	
        Args:
	    i: Iteration of the algorithm.
	    nStart: Number of restarts of the gradient ascent method.
        """
        try:
            n1=self._n1
            n2=self._dimW
	    tempN=self.numberTraining+i

            Xst=self.Obj.sampleFromXVn(nStart)
            wSt=self.Obj.simulatorW(nStart)
	    XWst=np.concatenate((Xst,wSt),1)
	    args3=self.getParametersOptVoi(i)
	    pool = mp.Pool(processes=numProcesses)
	    jobs = []
            for j in range(nStart):
                job = pool.apply_async(misc.VOIOptWrapper, args=(self,XWst[j:j+1,:],),
				       kwds=args3)
                jobs.append(job)
            pool.close()  # signal that no more data coming in
            pool.join()  # wait for all the tasks to complete
        except KeyboardInterrupt:
            print "Ctrl+c received, terminating and joining pool."
            pool.terminate()
            pool.join()
        for j in range(nStart):
            try:
                self.optRuns.append(jobs[j].get())
            except Exception as e:
                print "Error optimizing VOI"
        if len(self.optRuns):
            j = np.argmax([o.fOpt for o in self.optRuns])
	    fl.writeNewPointSBO(self,self.optRuns[j])
        self.optRuns=[]
        self.optPointsArray=[]

    def optimizeAn(self,start,i,L,logProduct=None):
	"""
        Optimize a_{n} using gradient ascent.
	
        Args:
	    start: Starting point for the gradient ascent method.
	    i: Iteration of the algorithm.
            L: Cholesky decomposition of the matrix A, where A is the covariance
               matrix of the past obsevations (x,w).
	    logProduct: Only used when the SEK is used.
			Vector with the logarithm of the product of the
                        expectations of np.exp(-alpha2[j]*((z-W[i,j])**2))
                        where W[i,:] is a point in the history.
        """

        tempN=i+self.numberTraining

        def g(x,grad,onlyGradient=False):
            return self.opt.functionGradientAscentAn(x,grad,self.stat,i,L,self.dataObj,
						     onlyGradient=onlyGradient,
						     logproductExpectations=logProduct)
	if self.opt.MethodAn=="SLSQP":
	    opt=op.SLSP(start)
	    def g1(x):
		return -1.0*g(x,grad=False)
	    
	    def dg(x):
		return -1.0*g(x,grad=True,onlyGradient=True)
	    
	    cons=self.opt.consAn
	    opt.run(f=g1,df=dg,cons=cons)
	else:
	    opt=op.OptSteepestDescent(n1=self.opt.dimXsteepestAn,projectGradient=self.opt.projectGradient,
				  xStart=start,xtol=self.opt.xtol,
				  stopFunction=self.opt.functionConditionOpt)
	    opt.run(f=g)

        #opt.run(f=g)
        self.optRuns.append(opt)
       # xTrans=self.opt.transformationDomainXAn(opt.xOpt[0:1,0:self.opt.dimXsteepestAn])
       # self.optPointsArray.append(xTrans)
    
    def optAnnoParal(self,i,logProd=True):
	"""
	Runs the single-start gradient ascent method to optimize a_{i}.
	
        Args:
	    i: Iteration of the algorithm.
	    logProd: True if we compute the logProduct for optimizeAn;
		     False otherwise.
        """
	n1=self._n1
	tempN=i+self.numberTraining
	
	A=self.stat._k.A(self.dataObj.Xhist[0:tempN,:],noise=self.dataObj.varHist[0:tempN])
	L=np.linalg.cholesky(A)
	
	if logProd:
	    tempX=self.dataObj.Xhist[0:tempN,n1:self._dimW+n1]
	    logProduct=self.stat.computeLogProductExpectationsForAn(tempX,
                                                         tempN,self.stat._k)
	else:
	    logProduct=None

	Xst=self.Obj.sampleFromXAn(1)
	args2={}
	args2['i']=i
	args2['L']=L
	args2['logProduct']=logProduct
	self.optRuns.append(misc.AnOptWrapper(self,start=Xst[0:1,:],**args2))
	fl.writeSolution(self,self.optRuns[0])

    def optAnParal(self,i,nStart,logProd=True,numProcesses=None):
	"""
	Runs in parallel the multi-start gradient ascent method
	to optimize a_{i}.
	It restarts the gradient ascent method nStart times.
        Args:
	    i: Iteration of the algorithm.
	    nStart: Number of restarts of the gradient ascent method.
	    logProd: True if we compute the logProduct for optimizeAn;
		     False otherwise.
        """
        try:
            n1=self._n1
	    tempN=i+self.numberTraining
	    A=self.stat._k.A(self.dataObj.Xhist[0:tempN,:],noise=self.dataObj.varHist[0:tempN])
	    L=np.linalg.cholesky(A)
	    
	    if logProd:
		tempX=self.dataObj.Xhist[0:tempN,n1:self._dimW+n1]
		logProduct=self.stat.computeLogProductExpectationsForAn(tempX,
							       tempN,self.stat._k)
	    else:
		logProduct=None

	    args3={}
	    args3['i']=i
	    args3['L']=L
	    args3['logProduct']=logProduct
	    Xst=self.Obj.sampleFromXAn(nStart)
            jobs = []
            pool = mp.Pool(processes=numProcesses)
            for j in range(nStart):
                job = pool.apply_async(misc.AnOptWrapper, args=(self,Xst[j:j+1,:],), kwds=args3)
                jobs.append(job)
            pool.close()  # signal that no more data coming in
            pool.join()  # wait for all the tasks to complete
        except KeyboardInterrupt:
            print "Ctrl+c received, terminating and joining pool."
            pool.terminate()
            pool.join()

        for j in range(nStart):
            try:
                self.optRuns.append(jobs[j].get())
            except Exception as e:
                print "Error optimizing An"

                
        if len(self.optRuns):
            j = np.argmax([o.fOpt for o in self.optRuns])
	    fl.writeSolution(self,self.optRuns[j])

        self.optRuns=[]
        self.optPointsArray=[]

    def trainModel(self,numStarts,**kwargs):
	"""
	Trains the hyperparameters of the kernel.
	
        Args:
	    numStarts: Number of random restarts to optimize
		       the hyperparameters.
        """
	if self.miscObj.parallel:
	    self.stat._k.train(scaledAlpha=self.stat.scaledAlpha,numStarts=numStarts,**kwargs)
	else:
	    self.stat._k.trainnoParallel(scaledAlpha=self.stat.scaledAlpha,**kwargs)
        f=open(os.path.join(self.path,'%d'%self.miscObj.rs+"hyperparameters.txt"),'w')
        f.write(str(self.stat._k.getParamaters()))
        f.close()

