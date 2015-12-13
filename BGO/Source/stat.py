#!/usr/bin/env python
"""
This file defines the statistical model used.
It includes the statistical models for SBO, KG,
EI and PI.
"""

import numpy as np
from math import *
from scipy import linalg
from numpy import linalg as LA
from scipy.stats import norm
import os
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
from matplotlib import pyplot as plt
from . import SK
from . import gradients

class GaussianProcess:
    def __init__(self,dimKernel,numberTraining,trainingData=None,
                 kernel=None,scaledAlpha=1.0):
        """
        This class defines the statistical model used.
        
        Arguments:
            -kernel: kernel
            -dimKernel: Dimension of the kernel.
            -numberTraining: Numer of training data.
            -scaledAlpha: The hyperparameters of the kernel are scaled by
                          alpha/(scaledAlpha^{2}).
            -SEK: True if SEK is using; False otherwise.
            -trainingData: Data object.
        """
        self._k=kernel
        self._numberTraining=numberTraining ##number of points used to train the kernel
        self._n=dimKernel
        self.scaledAlpha=scaledAlpha
        if trainingData is not None:
            self.data=trainingData.copyData()
    
class SBOGP(GaussianProcess):
    def __init__(self,B,dimNoiseW,dimPoints,gradXBforAn=None, computeLogProductExpectationsForAn=None,
                 SEK=True,*args,**kargs):
        GaussianProcess.__init__(self,*args,**kargs)
        """
        Statistical model for SBO.
        
        Arguments:
            -B: Computes B(x)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
                Its arguments are
                    x: Vector of points where B is evaluated
                    XW: Point (x,w)
                    n1: Dimension of x
                    n2: Dimension of w
                    logproductExpectations: Vector with the logarithm
                                            of the product of the
                                            expectations of
                                            np.exp(-alpha2[j]*((z-W[i,j])**2))
                                            where W[i,:] is a point in the history.
                                            Only used with the SEK.
            -dimNoiseW: Dimension of w.
            -dimPoints: Dimension of x.
            -gradXBforAn: Computes the gradient of B(x,i) for i in
                          {1,...,n+nTraining} where nTraining is the
                          number of training points.
                          Its arguments are
                            x: Argument of B
                            n: Current iteration of the algorithm
                            B: Vector {B(x,i)} for i in {1,...,n}
                            kern: kernel
                            X: Past observations X[i,:] for i in {1,..,n+nTraining}
            -computeLogProductExpectationsForAn: Only used with the SEK.
                                                 Computes the logarithm of the product
                                                 of the expectations of
                                                 np.exp(-alpha2[j]*((z-W[i,j])**2))
                                                 where W[i,:] is a point in the history.
                                                 Its arguments are:
                                                   W: Matrix where each row is a past
                                                      random vector used W[i,:]
                                                   N: Number of observations
                                                   kernel: kernel
            -SEK: True if SEK is used; False otherwise. 
        """
        self.SBOGP_name="SBO"
        self.n1=dimPoints
        self.n2=dimNoiseW
        self.B=B
        self.gradXBforAn=gradXBforAn
        self.computeLogProductExpectationsForAn=computeLogProductExpectationsForAn
        if SEK:
            self._k=SK.SEK(self.n1+self.n2,X=self.data.Xhist,
                           y=self.data.yHist[:,0],
                           noise=self.data.varHist,
                           scaleAlpha=self.scaledAlpha)
            self.gradXBforAn=gradients.gradXBforAnSEK
            

        

    def aN_grad(self,x,L,n,dataObj,gradient=True,onlyGradient=False,logproductExpectations=None):
        """
        Computes a_{n} and it can compute its derivative. It evaluates a_{n},
        when grad and onlyGradient are False; it evaluates the a_{n} and computes its
        derivative when grad is True and onlyGradient is False, and computes only its
        gradient when gradient and onlyGradient are both True.
        
        Args:
            x: a_{n} is evaluated at x.
            L: Cholesky decomposition of the matrix A, where A is the covariance
               matrix of the past obsevations (x,w).
            n: Step of the algorithm.
            dataObj: Data object (it contains all the history).
            gradient: True if we want to compute the gradient; False otherwise.
            onlyGradient: True if we only want to compute the gradient; False otherwise.
            logproductExpectations: Vector with the logarithm of the product of the
                                    expectations of np.exp(-alpha2[j]*((z-W[i,j])**2))
                                    where W[i,:] is a point in the history.
                                    --Only with the SEK--
        """
        n1=self.n1
        n2=self.n2
        muStart=self._k.mu
        y2=dataObj.yHist[0:n+self._numberTraining]-self._k.mu
        B=np.zeros(n+self._numberTraining)
        
        if logproductExpectations is None:
            for i in xrange(n+self._numberTraining):
                B[i]=self.B(x,dataObj.Xhist[i,:],self.n1,self.n2)
        else:
            for i in xrange(n+self._numberTraining):
                B[i]=self.B(x,dataObj.Xhist[i,:],self.n1,self.n2,self._k,logproductExpectations[i])
        
        inv1=linalg.solve_triangular(L,y2,lower=True)

        if onlyGradient:
            gradXB=self.gradXBforAn(x,n,B,self._k,
                                    dataObj.Xhist[0:n+self._numberTraining,0:n1],
                                    n1,self._numberTraining)
            temp4=linalg.solve_triangular(L,gradXB.transpose(),lower=True)
            gradAn=np.dot(inv1.transpose(),temp4)
            return gradAn

        inv2=linalg.solve_triangular(L,B.transpose(),lower=True)
        aN=muStart+np.dot(inv2.transpose(),inv1)
        if gradient==True:
            gradXB=self.gradXBforAn(x,n,B,self._k,
                                    dataObj.Xhist[0:n+self._numberTraining,0:n1],
                                    n1,self._numberTraining)
            temp4=linalg.solve_triangular(L,gradXB.transpose(),lower=True)
            gradAn=np.dot(inv1.transpose(),temp4)
            return aN,gradAn
        else:
            return aN
        
    ####Check
    def plotAn(self,i,L,points,seed):
        m=points.shape[0]
        z=np.zeros(m)
        for j in xrange(m):
            z[j]=self.aN_grad(points[j,:],L,i,gradient=False)
        
        fig=plt.figure()
        plt.plot(points,-(points**2),label="G(x)")
        plt.plot(points,z,'--',label='$a_%d(x)$'%i)
        
        plt.xlabel('x',fontsize=26)
        plt.legend()
        plt.savefig(os.path.join('%d'%seed+"run",'%d'%i+"a_n.pdf"))
        plt.close(fig)
    

class EIGP(GaussianProcess):
    def __init__(self,dimPoints,gradXKern,SEK,*args,**kargs):
        GaussianProcess.__init__(self,*args,**kargs)
        self.SBOGP_name="GP_EI"
        self.n1=dimPoints
        self.gradXKern=gradients.gradXKern
        if SEK:
            self._k=SK.SEK(self.n1+self.n2,X=self.data.Xhist,
                           y=self.data.yHist[:,0],
                           noise=self.data.varHist,
                           scaleAlpha=self.scaledAlpha)
    
    def muN(self,x,n,L,X,temp1,grad=False,onlyGrad=False):
        x=np.array(x).reshape((1,self.n1))
        tempN=self._numberTraining+n
        B=np.zeros([1,tempN])
        
        for i in xrange(tempN):
            B[:,i]=self._k.K(x,X[i:i+1,:])
            
       # y=self._yHist[0:tempN,:]
        temp2=linalg.solve_triangular(L,B.T,lower=True)
        
        if grad:
            gradX=self.gradXKern(x,n,self)
            gradi=np.zeros(self.n1)
          #  temp3=linalg.solve_triangular(L,y-muStart,lower=True)
            
            for j in xrange(self.n1):
                temp5=linalg.solve_triangular(L,gradX[:,j].T,lower=True)
                gradi[j]=muStart+np.dot(temp5[j,:],temp1)
            
        
        if onlyGrad:
            return gradi
        
            
     #   x=np.array(x)
     #   m=1
        
     #   X=self._Xhist[0:tempN,:]
     #   A=self._k.A(self._Xhist[0:tempN,:],noise=self._noiseHist[0:tempN])
     #   L=np.linalg.cholesky(A)

       # muStart=self._k.mu
       # temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True)
        a=muStart+np.dot(temp2.T,temp1)
        if grad==False:
            return a
     #   x=np.array(x).reshape((1,self.n1))

        return a,gradi
    
    
    
class KG(GaussianProcess):
    def __init__(self,dimPoints,gradXKern=None,SEK=True,*args,**kargs):
        GaussianProcess.__init__(self,*args,**kargs)
        self.SBOGP_name="KG"
        self.n1=dimPoints
        self.gradXKern=gradXKern
        if SEK:
            self.gradXKern=gradients.gradXKernelSEK
            self._k=SK.SEK(self._n,X=self.data.Xhist,
                           y=self.data.yHist[:,0],
                           noise=self.data.varHist,
                           scaleAlpha=self.scaledAlpha)


    def muN(self,x,n,data,L,temp1,grad=True,onlyGradient=False):
        tempN=self._numberTraining+n
        x=np.array(x).reshape((1,self.n1))
        if onlyGradient:
            gradX=self.gradXKern(x,n,self._k,self._numberTraining,
                                 data.Xhist[0:tempN,:],self.n1)
            gradi=np.zeros(self.n1)
            for j in xrange(self.n1):
                temp2=linalg.solve_triangular(L,gradX[:,j].T,lower=True)
                gradi[j]=np.dot(temp2.T,temp1)
            return gradi
            
        X=data.Xhist[0:tempN,:]
        B=np.zeros([1,tempN])
        muStart=self._k.mu
        
        for i in xrange(tempN):
            B[:,i]=self._k.K(x,X[i:i+1,:])
        temp2=linalg.solve_triangular(L,B.T,lower=True)
        a=muStart+np.dot(temp2.T,temp1)
        if grad==False:
            return a
        
        gradX=self.gradXKern(x,n,self._k,self._numberTraining,
                             data.Xhist[0:tempN,:],self.n1)
        gradi=np.zeros(self.n1)
        for j in xrange(self.n1):
            temp2=linalg.solve_triangular(L,gradX[:,j].T,lower=True)
            gradi[j]=np.dot(temp2.T,temp1)
        return a,gradi
        
class PIGP(GaussianProcess):
    def __init__(self,dimPoints,gradXKern,*args,**kargs):
        GaussianProcess.__init__(self,*args,**kargs)
        self.SBOGP_name="GP_EI"
        self.n1=dimPoints
        self.gradXKern=gradXKern
    
    def muN(self,x,n,grad=False):
        x=np.array(x)
        m=1
        tempN=self._numberTraining+n
        X=self._Xhist[0:tempN,:]
        A=self._k.A(self._Xhist[0:tempN,:],noise=self._noiseHist[0:tempN])
        L=np.linalg.cholesky(A)
        x=np.array(x).reshape((1,self.n1))
        B=np.zeros([m,tempN])
        
        for i in xrange(tempN):
            B[:,i]=self._k.K(x,X[i:i+1,:])
            
        y=self._yHist[0:tempN,:]
        temp2=linalg.solve_triangular(L,B.T,lower=True)
        muStart=self._k.mu
        temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True)
        a=muStart+np.dot(temp2.T,temp1)
        if grad==False:
            return a
        x=np.array(x).reshape((1,self.n1))
       # gradX=np.zeros((n,self.n1))
        gradX=self.gradXKern(x,n,self)
        gradi=np.zeros(self.n1)
        temp3=linalg.solve_triangular(L,y-muStart,lower=True)
        
        for j in xrange(self.n1):
           # for i in xrange(n):
           #     gradX[i,j]=self._k.K(x,X[i,:].reshape((1,self._n1)))*(2.0*self._alpha1[j]*(x[0,j]-X[i,j]))
            temp2=linalg.solve_triangular(L,gradX[:,j].T,lower=True)
            gradi[j]=muStart+np.dot(temp2.T,temp3)
        return a,gradi
    
    
    def varN(self,x,n,grad=False):
        temp=self._k.K(np.array(x).reshape((1,self.n1)))
        tempN=self._numberTraining+n
        sigmaVec=np.zeros((tempN,1))
        for i in xrange(tempN):
            sigmaVec[i,0]=self._k.K(np.array(x).reshape((1,self.n1)),self._Xhist[i:i+1,:])[:,0]
        A=self._k.A(self._Xhist[0:tempN,:],noise=self._noiseHist[0:tempN])
        L=np.linalg.cholesky(A)
        temp3=linalg.solve_triangular(L,sigmaVec,lower=True)
        temp2=np.dot(temp3.T,temp3)
        temp2=temp-temp2
        if grad==False:
            return temp2
        else:
            gradi=np.zeros(self.n1)
            x=np.array(x).reshape((1,self.n1))

            gradX=self.gradXKern(x,n,self)
            #gradX=np.zeros((n,self._n1))
            for j in xrange(self.n1):
              #  for i in xrange(n):
                  #  gradX[i,j]=self._k.K(x,self._X[i,:].reshape((1,self._n1)))*(2.0*self._alpha1[j]*(x[0,j]-self._X[i,j]))
                temp5=linalg.solve_triangular(L,gradX[:,j].T,lower=True)
                gradi[j]=np.dot(temp5.T,temp3)
            gradVar=-2.0*gradi
            return temp2,gradVar
    
