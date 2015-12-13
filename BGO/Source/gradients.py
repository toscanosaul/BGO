
import numpy as np


#####SBO
def gradXBforAnSEK(x,n,B,kern,X,n1,nT):
    """Computes the gradient of B(x,i) for i in {1,...,n+nTraining}
       where nTraining is the number of training points
      
       Args:
          x: Argument of B
          n: Current iteration of the algorithm
          B: Vector {B(x,i)} for i in {1,...,n}
          kern: kernel
          X: Past observations X[i,:] for i in {1,..,n+nTraining}
          n1: Dimension of x
          nT: Number of training points
    """
    gradXB=np.zeros((n1,n+nT))
    alpha1=0.5*((kern.alpha[0:n1])**2)/(kern.scaleAlpha)**2
    for i in xrange(n+nT):
        gradXB[:,i]=B[i]*(-2.0*alpha1*(x-X[i,:]))
    return gradXB


def gradXBSEK(new,kern,BN,keep,points,n1):
    """Computes the vector of gradients with respect to x_{n+1} of
        B(x_{p},n+1)=\int\Sigma_{0}(x_{p},w,x_{n+1},w_{n+1})dp(w),
        where x_{p} is a point in the discretization of the domain of x.
        
       Args:
          new: Point (x_{n+1},w_{n+1})
          kern: Kernel
          keep: Indexes of the points keeped of the discretization of the domain of x,
                after using AffineBreakPoints
          BN: Vector B(x_{p},n+1), where x_{p} is a point in the discretization of
              the domain of x.
          points: Discretization of the domain of x
          n1: Dimension of x
    """
    alpha1=0.5*((kern.alpha[0:n1])**2)/(kern.scaleAlpha)**2
    xNew=new[0,0:n1].reshape((1,n1))
    gradXBarray=np.zeros([len(keep),n1])
    M=len(keep)
    for i in xrange(n1):
        for j in xrange(M):
            gradXBarray[j,i]=-2.0*alpha1[i]*BN[keep[j],0]*(xNew[0,i]-points[keep[j],i])
    return gradXBarray

def gradXWSigmaOfuncSEK(n,new,kern,Xtrain2,Wtrain2,n1,n2,nT):
    """Computes the vector of the gradients of Sigma_{0}(new,XW[i,:]) for
        all the past observations XW[i,]. Sigma_{0} is the covariance of
        the GP on F.
        
       Args:
          n: Number of iteration
          new: Point where Sigma_{0} is evaluated
          kern: Kernel
          Xtrain2: Past observations of X
          Wtrain2: Past observations of W
          N: Number of observations
          n1: Dimension of x
          n2: Dimension of w
          nT: Number of training points
    """
    gradXSigma0=np.zeros([n+nT+1,n1])
    tempN=n+nT
    past=np.concatenate((Xtrain2,Wtrain2),1)
    gamma=np.transpose(kern.A(new,past))
    alpha1=0.5*((kern.alpha[0:n1])**2)/(kern.scaleAlpha)**2
    gradWSigma0=np.zeros([n+nT+1,n2])

    alpha2=0.5*((kern.alpha[n1:n1+n2])**2)/(kern.scaleAlpha)**2
    xNew=new[0,0:n1]
    wNew=new[0,n1:n1+n2]
    for i in xrange(n+nT):
        gradXSigma0[i,:]=-2.0*gamma[i]*alpha1*(xNew-Xtrain2[i,:])
        gradWSigma0[i,:]=-2.0*gamma[i]*alpha2*(wNew-Wtrain2[i,:])
    return gradXSigma0,gradWSigma0

####KG
def gradXKernelSEK(x,n,kern,trainingPoints,X,n1):
    alpha=0.5*((kern.alpha)**2)/(kern.scaleAlpha)**2
    tempN=n+trainingPoints
    gradX=np.zeros((tempN,n1))
    for j in xrange(n1):
        for i in xrange(tempN):
            aux=kern.K(x,X[i,:].reshape((1,n1)))
            gradX[i,j]=aux*(-2.0*alpha[j]*(x[0,j]-X[i,j]))
    return gradX


def gradXKernel2SEK(x,Btemp,points,nD,mD,kern):
    alpha=0.5*((kern.alpha)**2)/(kern.scaleAlpha)**2
    temp=np.zeros((nD,mD))
    for i in xrange(nD):
        temp[i,:]=(-2.0*alpha[i])*(x[0,i]-points[:,i])
    return temp*Btemp[:,0]