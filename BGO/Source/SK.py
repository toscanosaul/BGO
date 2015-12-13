#!/usr/bin/env python

"""
This file defines the kernels. We can optimize the hyperparameters,
compute the log-likelihood and the matrix A from the paper [tf].
"""

import numpy as np
from scipy import linalg
from scipy.optimize import fmin_l_bfgs_b
from . matrixComputations import tripleProduct,inverseComp
from scipy.stats import multivariate_normal
import multiprocessing as mp
from . import misc
from . import optimization
from scipy import array, linalg, dot

class SEK:
    def __init__(self,n,scaleAlpha=None,nRestarts=10,X=None,y=None,
                 noise=None,optName='bfgs'):
        """
        Defines the squared exponential kernel,
            variance*exp(-0.5*sum (alpha_i/scaleAlpha)**2 *(x_i-y_i)**2))
        
        Args:
            -n: Dimension of the domain of the kernel.
            -scaleAlpha: The hyperparameters of the kernel are scaled by
                         alpha/(scaledAlpha^{2}).
            -nRestarts: Number of restarts to optimze the hyperparameters.
            -X: Training data.
            -y: Outputs of the training data.
            -noise: Noise of the outputs.
        """
        if scaleAlpha is None:
            scaleAlpha=1.0
        self.scaleAlpha=scaleAlpha
        self.dimension=n
        self.alpha=np.ones(n)
        self.variance=[1.0]
        self.mu=[0.0]
        self.optimizationMethod=optName

        self.X=X
        self.y=y
        self.noise=noise
        self.optRuns=[]
        self.optPointsArray=[]
        self.restarts=nRestarts
        
    def getParamaters(self):
        """
        Returns a dictionary with the hyperparameters and the mean
        of the GP.
        """
        dic={}
        dic['alphaPaper']=0.5*(self.alpha**2)/self.scaleAlpha
        dic['variance']=self.variance
        dic['mu']=self.mu
        return dic

    def K(self, X, X2=None,alpha=None,variance=None):
        """
        Computes the covariance matrix cov(X[i,:],X2[j,:]).
        
        Args:
            X: Matrix where each row is a point.
            X2: Matrix where each row is a point.
            alpha: It's the scaled alpha.
            Variance: Sigma hyperparameter.
            
        """
        if alpha is None:
            alpha=self.alpha
        if variance is None:
            variance=self.variance
            
        if X2 is None:
            X=X*alpha/self.scaleAlpha
            Xsq=np.sum(np.square(X), 1)
            r=-2.*np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
            r = np.clip(r, 0, np.inf)
            return variance*np.exp(-0.5*r)
        else:
            X=X*alpha/self.scaleAlpha
            X2=X2*alpha/self.scaleAlpha
            r=-2.*np.dot(X, X2.T) + (np.sum(np.square(X), 1)[:, None] + np.sum(np.square(X2), 1)[None, :])
            r = np.clip(r, 0, np.inf)
            return variance*np.exp(-0.5*r)
    
    def A(self,X,X2=None,noise=None,alpha=None,variance=None):
        """
        Computes the covariance matrix A on the points X, and adds
        the noise of each observation.
        
        Args:
            X: Matrix where each row is a point.
            X2: Matrix where each row is a point.
            noise: Noise of the observations.
            alpha: Hyperparameters of the kernel.
            Variance: Sigma hyperparameter.
        """
        if noise is None:
            K=self.K(X,X2,alpha=alpha,variance=variance)
        else:
            K=self.K(X,X2,alpha=alpha,variance=variance)+np.diag(noise)
        return K
    
    def logLikelihood(self,X,y,noise=None,alpha=None,variance=None,mu=None,gradient=False):
        """
        Computes the log-likelihood and its gradient. The gradient is respect to  log(var)
        and log(alpha**2).
        
        Args:
            -X: Matrix with the training data.
            -y: Output of the training data.
            -noise: Noise of the outputs.
            -alpha: Hyperparameters of the kernel
            -variance: Hyperparameter of the kernel.
            -mu: Mean parameter of the GP.
            -gradient: True if we want the gradient; False otherwise.
        """
        if alpha is None:
            alpha=self.alpha
        if variance is None:
            variance=self.variance
        if mu is None:
            mu=self.mu
        if noise is None:
            K=self.A(X,alpha=alpha,variance=variance)
        else:
            K=self.A(X,alpha=alpha,variance=variance,noise=noise)
        y2=y-mu
        N=X.shape[0]
        try:
            L=np.linalg.cholesky(K)
            alp=inverseComp(L,y2)
            logLike=-0.5*np.dot(y2,alp)-np.sum(np.log(np.diag(L)))-0.5*N*np.log(2.0*np.pi)
            if gradient==False:
                return logLike
            gradient=np.zeros(self.dimension+2)
            
            temp=np.dot(alp[:,None],alp[None,:])
            K2=self.A(X,alpha=alpha,variance=variance)
            for i in range(self.dimension):
                derivative=np.zeros((N,N))
                derivative=K2*(-(0.5/(self.scaleAlpha**2))*(alpha[i]**2)*((X[:,i][:,None]-X[:,i][None,:])**2))
                temp3=inverseComp(L,derivative)
                gradient[i]=0.5*np.trace(np.dot(temp,derivative)-temp3)
            
            der=self.K(X,alpha=alpha,variance=variance)
            temp3=inverseComp(L,der)
            gradient[self.dimension]=0.5*np.trace(np.dot(temp,der)-temp3)

            der=np.ones((N,N))
            temp3=inverseComp(L,der)
            gradient[self.dimension+1]=0.5*np.trace(np.dot(temp,der)-temp3)
            return logLike,gradient
        except:
            print "no"
            L=np.linalg.inv(K)
            det=np.linalg.det(K)
            logLike=-0.5*np.dot(y2,np.dot(L,y2))-0.5*N*np.log(2*np.pi)-0.5*np.log(det)
            if gradient==False:
                return logLike
            gradient=np.zeros(self.dimension+2)
            
            alp=np.dot(L,y2)
            temp=np.dot(alp[:,None],alp.T[None,:])
            K2=self.A(X,alpha=alpha,variance=variance)
            for i in range(self.dimension):
                derivative=np.zeros((N,N))
                derivative=K2*(-(0.5/(self.scaleAlpha**2))*(alpha[i]**2)*((X[:,i][:,None]-X[:,i][None,:])**2))
                temp2=np.dot(temp-L,derivative)
                gradient[i]=0.5*np.trace(temp2)
            
            temp2=np.dot(temp-L,K2)
            gradient[self.dimension]=0.5*np.trace(temp2)
            
            der=np.ones((N,N))
            temp2=np.dot(temp-L,der)
            gradient[self.dimension+1]=0.5*np.trace(temp2)
            return logLike,gradient
            
    def gradientLogLikelihood(self,X,y,noise=None,alpha=None,variance=None,mu=None):
        """
        Computes the gradient of the log-likelihood, respect to log(var)
        and log(alpha**2).
        
        Args:
            -X: Matrix with the training data.
            -y: Output of the training data.
            -noise: Noise of the outputs.
            -alpha: Hyperparameters of the kernel
            -variance: Hyperparameter of the kernel.
            -mu: Mean parameter of the GP.
            -gradient: True if we want the gradient; False otherwise.
        """
        return self.logLikelihood(X,y,noise=noise,alpha=alpha,variance=variance,mu=mu,gradient=True)[1]
    
    def minuslogLikelihoodParameters(self,t):
        """
        Computes the minus log-likelihood.
        
        Args:
            t: hyperparameters of the kernel.
        """
        alpha=t[0:self.dimension]
        variance=np.exp(t[self.dimension])
        mu=t[self.dimension+1]
        return -self.logLikelihood(self.X,self.y,self.noise,alpha=alpha,variance=variance,mu=mu)
    
    def minusGradLogLikelihoodParameters(self,t):
        """
        Computes the gradient of the minus log-likelihood.
        
        Args:
            t: hyperparameters of the kernel.
        """
        alpha=t[0:self.dimension]
        variance=np.exp(t[self.dimension])
        mu=t[self.dimension+1]
        return -self.gradientLogLikelihood(self.X,self.y,self.noise,alpha=alpha,variance=variance,mu=mu)

    def optimizeKernel(self,start=None,optimizer=None,**kwargs):
        """
        Optimize the minus log-likelihood using the optimizer method and starting in start.
        
        Args:
            start: starting point of the algorithm.
            optimizer: Name of the optimization algorithm that we want to use;
                       e.g. 'bfgs'.
            
        """
        if start is None:
            start=np.concatenate((np.log(self.alpha**2),np.log(self.variance),self.mu))
        if optimizer is None:
            optimizer=self.optimizationMethod
        
        optimizer = optimization.getOptimizationMethod(optimizer)
        opt=optimizer(start,**kwargs)
        opt.run(f=self.minuslogLikelihoodParameters,df=self.minusGradLogLikelihoodParameters)
        self.optRuns.append(opt)
        self.optPointsArray.append(opt.xOpt)
    
    
    def trainnoParallel(self,scaledAlpha,**kwargs):
        """
        Train the hyperparameters starting in only one point the algorithm.
        
        Args:
            -scaledAlpha: The definition may be found above.
        """
        dim=self.dimension
        alpha=np.random.randn(dim)
        variance=np.random.rand(1)
        st=np.concatenate((np.sqrt(np.exp(alpha)),np.exp(variance),[0.0]))
        args2={}
        args2['start']=st
        job=misc.kernOptWrapper(self,**args2)
        temp=job.xOpt
        self.alpha=np.sqrt(np.exp(np.array(temp[0:self.dimension])))
        self.variance=np.exp(np.array(temp[self.dimension]))
        self.mu=np.array(temp[self.dimension+1])

    def train(self,scaledAlpha,numStarts=None,numProcesses=None,**kwargs):
        """
        Train the hyperparameters starting in several different points.
        
        Args:
            -scaledAlpha: The definition may be found above.
            -numStarts: Number of restarting times oft he algorithm.
        """
        if numStarts is None:
            numStarts=self.restarts
        try:
            dim=self.dimension
            jobs = []
            args3=[]
            pool = mp.Pool(processes=numProcesses)
            alpha=np.random.randn(numStarts,dim)
            variance=np.random.rand(numStarts,1)
            tempZero=np.zeros((numStarts,1))
            st=np.concatenate((np.sqrt(np.exp(alpha)),np.exp(variance),tempZero),1)
            for i in range(numStarts):
               # alpha=np.random.randn(dim)
               # variance=np.random.rand(1)
               # st=np.concatenate((np.sqrt(np.exp(alpha)),np.exp(variance),[0.0]))
               # args2={}
               # args2['start']=st
               # args3.append(args2.copy())
                job = pool.apply_async(misc.kernOptWrapper, args=(self,st[i,:],))
                jobs.append(job)
            
            pool.close()  # signal that no more data coming in
            pool.join()  # wait for all the tasks to complete
        except KeyboardInterrupt:
            print "Ctrl+c received, terminating and joining pool."
            pool.terminate()
            pool.join()

        for i in range(numStarts):
            try:
                self.optRuns.append(jobs[i].get())
            except Exception as e:
                print "what"


        if len(self.optRuns):
            i = np.argmin([o.fOpt for o in self.optRuns])
            temp=self.optRuns[i].xOpt
            self.alpha=np.sqrt(np.exp(np.array(temp[0:self.dimension])))
            self.variance=np.exp(np.array(temp[self.dimension]))
            self.mu=np.array(temp[self.dimension+1])

    
    def printPar(self):
        """
        Print the hyperparameters of the kernel.
        """
        print "alpha is "+self.alpha
        print "variance is "+self.variance
        print "mean is "+ self.mu
        
        

        
