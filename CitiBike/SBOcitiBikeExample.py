#!/usr/bin/env python

"""
We consider a queuing simulation based on New York City's Bike system,
in which system users may remove an available bike from a station at one
location within the city, and ride it to a station with an available dock
in some other location within the city. The optimization problem that we
consider is the allocation of a constrained number of bikes (6000) to available
docks within the city at the start of rush hour, so as to minimize, in
simulation, the expected number of potential trips in which the rider could not
find an available bike at their preferred origination station, or could not find
an available dock at their preferred destination station. We call such trips
"negatively affected trips".

We optimize the objective using the SBO algorithm. In this script, we create the
6 objets needed by this algorithm:

Objobj: Objective object.
miscObj: Miscellaneous object.
VOIobj: Value of Information function object.
optObj: Opt object.
statObj: Statistical object.
dataObj: Data object.

For the descrition of those objects, please refer to
https://github.com/toscanosaul/BGO/blob/master/BGO.pdf.

This script is run with 6 arguments:

1) Random seed (int).
2) Number of training points (int).
3) Number of samples to estimate the exected negatively affected trips (int).
4) Number of iterations of the algorithm (int).
5) Run the optimization algorithms at multiple starting points (bool).
6) Number of points to restart the optimization algorithms (int)
"""

import sys
sys.path.append("..")
import numpy as np
from simulationPoissonProcessNonHomogeneous import *
from math import *
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
import multiprocessing as mp
import os
from scipy.stats import poisson
import json
from BGO.Source import *
import time

"""
Save arguments given by the user.
"""

randomSeed=int(sys.argv[1])
trainingPoints=int(sys.argv[2])
numberSamplesForF=int(sys.argv[3])
numberIterations=int(sys.argv[4]) 
parallel=sys.argv[5] 

if parallel=='F':
    parallel=False
    numberRestarts=1
elif parallel=='T':
    numberRestarts=int(sys.argv[6])
    #number of restarts for the optimization methods
    parallel=True


np.random.seed(randomSeed)

##############

"""
n1: Dim(x)
n2: Dim(w)
"""

n1=4
n2=1

nDays=365
"""
We define the variables needed for the queuing simulation. 
"""

g=negativelyAffectedTrips  #Simulator


nSets=4

fil="poissonDays.txt"
fil=os.path.join("NonHomogeneousPP2",fil)
poissonParameters=np.loadtxt(fil)

###readData

poissonArray=[[] for i in xrange(nDays)]
exponentialTimes=[[] for i in xrange(nDays)]

for i in xrange(nDays):
    fil="daySparse"+"%d"%i+"ExponentialTimesNonHom.txt"
    fil2=os.path.join("SparseNonHomogeneousPP2",fil)
    poissonArray[i].append(np.loadtxt(fil2))
    
    fil="daySparse"+"%d"%i+"PoissonParametersNonHom.txt"
    fil2=os.path.join("SparseNonHomogeneousPP2",fil)
    exponentialTimes[i].append(np.loadtxt(fil2))

numberStations=329
Avertices=[[]]
for j in range(numberStations):
    for k in range(numberStations):
	Avertices[0].append((j,k))

f = open(str(4)+"-cluster.txt", 'r')
cluster=eval(f.read())
f.close()

bikeData=np.loadtxt("bikesStationsOrdinalIDnumberDocks.txt",skiprows=1)

TimeHours=4.0
numberBikes=6000

poissonParameters*=TimeHours


###upper bounds for X
upperX=np.zeros(n1)
temBikes=bikeData[:,2]
for i in xrange(n1):
    temp=cluster[i]
    indsTemp=np.array([a[0] for a in temp])
    upperX[i]=np.sum(temBikes[indsTemp])
    


"""
We define the objective object.
"""

def noisyF(XW,n,ind=None):
    """Estimate F(x,w)=E(f(x,w,z)|w)
      
       Args:
          XW: Vector (x,w)
          n: Number of samples to estimate F
    """
    simulations=np.zeros(n)
    x=XW[0,0:n1]
    w=XW[0,n1:n1+n2]


    for i in xrange(n):
        simulations[i]=g(TimeHours,w,x,nSets,
                         cluster,bikeData,poissonParameters,nDays,
			 Avertices,poissonArray,exponentialTimes,randomSeed=ind)

    
    return np.mean(simulations),float(np.var(simulations))/n

def sampleFromXAn(n):
    """Chooses n points in the domain of x at random
      
       Args:
          n: Number of points chosen
    """
    aux1=(numberBikes/float(n1))*np.ones((1,n1-1))
    if n>1:
        #s=np.random.dirichlet(np.ones(4),n-1)
        lower=100
        s=np.random.uniform(0,1,(n-1,4))
        s[:,0]=s[:,0]*upperX[0]+(1-s[:,0])*lower
        s[:,0]=np.floor(s[:,0])

        for j in range(n-1):
            s[j,1]=s[j,1]*min(upperX[1],nBikes-2*lower-s[j,0])+(1-s[j,1])*lower
            s[j,1]=np.floor(s[j,1])
	    tempMin=min(nBikes-s[j,0]-s[j,1]-lower,upperX[2])
	    tempMax=max(nBikes-s[j,0]-s[j,1]-upperX[3],lower)
            s[j,2]=s[j,2]*tempMin+(1-s[j,2])*tempMax
            s[j,2]=np.floor(s[j,2])
            s[j,3]=nBikes-np.sum(s[j,0:3])
        aux1=np.concatenate((s[:,0:n1-1],aux1),0)
    return aux1

sampleFromXVn=sampleFromXAn

def simulatorW(n,ind=False):
    """Simulate n vectors w
      
       Args:
          n: Number of vectors simulated
    """
    wPrior=np.zeros((n,n2))
    indexes=np.random.randint(0,nDays,n)
    for i in range(n):
	for j in range(n2):
	    wPrior[i,j]=np.random.poisson(poissonParameters[indexes[i]],1)
    if ind:
	return wPrior,indexes
    else:
	return wPrior
    
def g2(x,w,day,i):
    return g(TimeHours,w,x,nSets,
                         cluster,bikeData,poissonParameters,nDays,
			 Avertices,poissonArray,exponentialTimes,day,i)



def estimationObjective(x,N=10):
    """Estimate g(x)=E(f(x,w,z))
      
       Args:
          x
          N: number of samples used to estimate g(x)
    """
    estimator=N
    W,indexes=simulatorW(estimator,True)
    result=np.zeros(estimator)
    rseed=np.random.randint(1,4294967290,size=N)
    pool = mp.Pool()
    jobs = []
    for j in range(estimator):
        job = pool.apply_async(g2, args=(x,W[j,:],indexes[j],rseed[j],))
        jobs.append(job)
    pool.close()  # signal that no more data coming in
    pool.join()  # wait for all the tasks to complete
    
    for i in range(estimator):
        result[i]=jobs[i].get()
    
    return np.mean(result),float(np.var(result))/estimator

Objective=inter.objective(g,n1,noisyF,numberSamplesForF,sampleFromXVn,
                          simulatorW,estimationObjective,sampleFromXAn)




"""
We define the miscellaneous object.
"""

misc=inter.Miscellaneous(randomSeed,parallel,nF=numberSamplesForF,
			 tP=trainingPoints,prefix="2FinalNonHomogeneous011116")

"""
We define the data object.
"""

"""
Generate the training data
"""

tempX=sampleFromXVn(trainingPoints)
tempFour=numberBikes-np.sum(tempX,1)
tempFour=tempFour.reshape((trainingPoints,1))
Xtrain=np.concatenate((tempX,tempFour),1)
Wtrain=simulatorW(trainingPoints)
XWtrain=np.concatenate((Xtrain,Wtrain),1)

dataObj=inter.data(XWtrain,yHist=None,varHist=None)

dataObj.getTrainingDataSBO(trainingPoints,noisyF,numberSamplesForF,False)

"""
We define the statistical object.
"""

dimensionKernel=n1+n2
scaleAlpha=2000.0


def computeProbability(w,parLambda,nDays):
    probs=poisson.pmf(w,mu=np.array(parLambda))
    probs*=(1.0/nDays)
    return np.sum(probs)

L=650
M=8900
wTemp=np.array(range(L,M))
probsTemp=np.zeros(M-L)
for i in range(M-L):
    probsTemp[i]=computeProbability(wTemp[i],poissonParameters,nDays)


def expectation(z,alpha,parLambda,nDays,probs,L=650,M=8900):
    w=np.array(range(L,M))
    aux=np.exp(-alpha*((z-w)**2))
    return np.dot(aux,probs)

def B(x,XW,n1,n2,kernel,logproductExpectations=None):
    """Computes B(x)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
      
       Args:
          x: Vector of points where B is evaluated
          XW: Point (x,w)
          n1: Dimension of x
          n2: Dimension of w
          kernel
          logproductExpectations: Vector with the logarithm
                                  of the product of the
                                  expectations of
                                  np.exp(-alpha2[j]*((z-W[i,j])**2))
                                  where W[i,:] is a point in the history.
          
    """
    x=np.array(x).reshape((x.shape[0],n1))
    results=np.zeros(x.shape[0])
    #parameterLamb=parameterSetsPoisson
    X=XW[0:n1]
    inda=n1+n2
    W=XW[n1:inda]
    alpha2=0.5*((kernel.alpha[n1:n1+n2])**2)/scaleAlpha**2
    alpha1=0.5*((kernel.alpha[0:n1])**2)/scaleAlpha**2
    variance0=kernel.variance
    if logproductExpectations is None:
        logproductExpectations=0.0
        for j in xrange(n2):
	    temp=expectation(W[j],alpha2[j],poissonParameters,nDays,probsTemp)
            logproductExpectations+=np.log(temp)
    for i in xrange(x.shape[0]):
	compAux=np.log(variance0)-np.sum(alpha1*((x[i,:]-X)**2))
        results[i]=logproductExpectations+compAux
    return np.exp(results)

def computeLogProductExpectationsForAn(W,N,kernel):
    """Computes the logarithm of the product of the
       expectations of np.exp(-alpha2[j]*((z-W[i,j])**2))
        where W[i,:] is a point in the history.
      
       Args:
          W: Matrix where each row is a past random vector used W[i,:]
          N: Number of observations
          kernel: kernel
    """
    alpha2=0.5*((kernel.alpha[n1:n1+n2])**2)/scaleAlpha**2
    logproductExpectations=np.zeros(N)
  #  parameterLamb=parameterSetsPoisson
    for i in xrange(N):
        logproductExpectations[i]=0.0
        for j in xrange(n2):
	    temp=expectation(W[i,j],alpha2[j],poissonParameters,nDays,probsTemp)
            logproductExpectations[i]+=np.log(temp)
    return logproductExpectations

stat=stat.SBOGP(B=B,dimNoiseW=n2,dimPoints=n1,trainingData=dataObj,
                dimKernel=n1+n2, numberTraining=trainingPoints,
                computeLogProductExpectationsForAn=
                computeLogProductExpectationsForAn,scaledAlpha=scaleAlpha)


"""
We define the VOI object.
"""

pointsVOI=np.loadtxt("lowerBoundNewRandompointsPoisson1000.txt") #Discretization of the domain of X


def expectation2(z,alpha,parLambda,nDays,probs):
    w=np.array(range(L,M))
    aux=-2.0*alpha*(z-w)*np.exp(-alpha*((z-w)**2))
    return np.dot(aux,probs)

def gradWB(new,kern,BN,keep,points):
    """Computes the vector of gradients with respect to w_{n+1} of
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
    """
    alpha1=0.5*((kern.alpha[0:n1])**2)/scaleAlpha**2
    alpha2=0.5*((kern.alpha[n1:n1+n2])**2)/scaleAlpha**2
    variance0=kern.variance
    wNew=new[0,n1:n1+n2].reshape((1,n2))
    gradWBarray=np.zeros([len(keep),n2])
    M=len(keep)
    X=new[0,0:n1]
    W=new[0,n1:n1+n2]
   
    for i in xrange(n2):
        logproductExpectations=0.0
        a=range(n2)
        del a[i]
        for r in a:
	    temp=expectation(W[r],alpha2[r],poissonParameters,nDays,probsTemp)
            logproductExpectations+=np.log(temp)
	temp=expectation2(W[i],alpha2[i],poissonParameters,nDays,probsTemp)
        productExpectations=np.exp(logproductExpectations)*temp
        for j in xrange(M):
            gradWBarray[j,i]=np.log(variance0)-np.sum(alpha1*((points[keep[j],:]-X)**2))
            gradWBarray[j,i]=np.exp(gradWBarray[j,i])*productExpectations
    return gradWBarray

VOIobj=VOI.VOISBO(dimX=n1, pointsApproximation=pointsVOI,
                  gradWBfunc=gradWB,dimW=n2,
                  numberTraining=trainingPoints)


"""
We define the Opt object.

"""

dimXsteepestAn=n1-1 #Dimension of x when the VOI and a_{n} are optimized.

def functionGradientAscentVn(x,VOI,i,L,temp2,a,kern,XW,scratch,Bfunc,onlyGradient=False,grad=None):
    """ Evaluates the VOI and it can compute its derivative. It evaluates the VOI,
        when grad and onlyGradient are False; it evaluates the VOI and computes its
        derivative when grad is True and onlyGradient is False, and computes only its
        gradient when gradient and onlyGradient are both True.
    
        Args:
            x: VOI is evaluated at (x,numberBikes-sum(x)).Note that we reduce the dimension
               of the space of x.
            grad: True if we want to compute the gradient; False otherwise.
            i: Iteration of the SBO algorithm.
            L: Cholesky decomposition of the matrix A, where A is the covariance
               matrix of the past obsevations (x,w).
            Bfunc: Computes B(x,XW)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
            temp2: temp2=inv(L)*B.T, where B is a matrix such that B(i,j) is
                   \int\Sigma_{0}(x_{i},w,x_{j},w_{j})dp(w)
                   where points x_{p} is a point of the discretization of
                   the space of x; and (x_{j},w_{j}) is a past observation.
            a: Vector of the means of the GP on g(x)=E(f(x,w,z)). The means are evaluated on the
               discretization of the space of x.
            VOI: VOI object
            kern: kernel
            XW: Past observations
            scratch: matrix where scratch[i,:] is the solution of the linear system
                     Ly=B[j,:].transpose() (See above for the definition of B and L)
            onlyGradient: True if we only want to compute the gradient; False otherwise.
    """
    grad=onlyGradient
    x=np.array(x).reshape([1,n1+n2-1])
    x4=np.array(numberBikes-np.sum(x[0,0:n1-1])).reshape((1,1))
    tempX=x[0:1,0:n1-1]
    x2=np.concatenate((tempX,x4),1)
    tempW=x[0:1,n1-1:n1-1+n2]
    xFinal=np.concatenate((x2,tempW),1)
    temp=VOI.VOIfunc(i,xFinal,L=L,temp2=temp2,a=a,grad=grad,scratch=scratch,onlyGradient=onlyGradient,
                          kern=kern,XW=XW,B=Bfunc)

    

    if onlyGradient:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
        subMatrix=np.zeros((n2,n1-1))
        L=np.concatenate((L,subMatrix))
        subMatrix=np.zeros((n1,n2))
        temDiag=np.identity(n2)
        sub=np.concatenate((subMatrix,temDiag))
        L=np.concatenate((L,sub),1)
        grad2=np.dot(temp,L)

        return grad2
        

    if grad==True:
	

        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
        subMatrix=np.zeros((n2,n1-1))
        L=np.concatenate((L,subMatrix))
        subMatrix=np.zeros((n1,n2))
        temDiag=np.identity(n2)
        sub=np.concatenate((subMatrix,temDiag))
        L=np.concatenate((L,sub),1)
        grad2=np.dot(temp[1],L)
        return temp[0],grad2
    else:

        return temp
    

def functionGradientAscentAn(x,grad,stat,i,L,dataObj,onlyGradient=False,logproductExpectations=None):
    """ Evaluates a_{i} and its derivative, which is the expectation of the GP on g(x).
        It evaluates a_{i}, when grad and onlyGradient are False; it evaluates the a_{i}
        and computes its derivative when grad is True and onlyGradient is False, and
        computes only its gradient when gradient and onlyGradient are both True.
    
        Args:
            x: a_{i} is evaluated at (x,numberBikes-sum(x)).Note that we reduce the dimension
               of the space of x.
            grad: True if we want to compute the gradient; False otherwise.
            i: Iteration of the SBO algorithm.
            L: Cholesky decomposition of the matrix A, where A is the covariance
               matrix of the past obsevations (x,w).
            dataObj: Data object.
            stat: Statistical object.
            onlyGradient: True if we only want to compute the gradient; False otherwise.
            logproductExpectations: Vector with the logarithm of the product of the
                                    expectations of np.exp(-alpha2[j]*((z-W[i,j])**2))
                                    where W[i,:] is a point in the history.
    """
   
    x=np.array(x).reshape([1,n1-1])
    x4=np.array(numberBikes-np.sum(x[0,0:n1-1])).reshape((1,1))
    tempX=x[0:1,0:n1-1]
    x=np.concatenate((tempX,x4),1)

    if onlyGradient:
        temp=stat.aN_grad(x,L,i,dataObj,grad,onlyGradient,logproductExpectations)
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L2=np.concatenate((t,s))
        grad2=np.dot(temp,L2)
        return grad2

    temp=stat.aN_grad(x,L,i,dataObj,gradient=grad,logproductExpectations=logproductExpectations)
    if grad==False:
        return temp
    else:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L2=np.concatenate((t,s))
        grad2=np.dot(temp[1],L2)
        return temp[0],grad2

lower=100

def const1(x):
    return numberBikes-lower-np.sum(x[0:n1-1])

def jac1(x):
    return np.array([-1,-1,-1,0])


def const2(x):
    return x[0]-lower

def jac2(x):
    return np.array([1,0,0,0])

def const3(x):
    return x[1]-lower

def jac3(x):
    return np.array([0,1,0,0])

def const4(x):
    return x[2]-lower

def jac4(x):
    return np.array([0,0,1,0])

def const5(x):
    return x[3]-lower

def jac5(x):
    return np.array([0,0,0,1])


def const6(x):
    return upperX[0]-x[0]

def jac6(x):
    return np.array([-1,0,0,0])

def const7(x):
    return upperX[1]-x[1]

def jac7(x):
    return np.array([0,-1,0,0])

def const8(x):
    return upperX[2]-x[2]

def jac8(x):
    return np.array([0,0,-1,0])

def const9(x):
    return -numberBikes+np.sum(x[0:n1-1])+upperX[3]

def jac9(x):
    return np.array([1,1,1,0])


cons=({'type':'ineq',
        'fun': const1,
       'jac': jac1},
    {'type':'ineq',
        'fun': const2,
       'jac': jac2},
    {'type':'ineq',
        'fun': const3,
       'jac': jac3},
    {'type':'ineq',
        'fun': const4,
       'jac': jac4},
        {'type':'ineq',
        'fun': const5,
       'jac': jac5},
        {'type':'ineq',
        'fun': const6,
       'jac': jac6},
        {'type':'ineq',
        'fun': const7,
       'jac': jac7},
        {'type':'ineq',
        'fun': const8,
       'jac': jac8},
        {'type':'ineq',
        'fun': const9,
       'jac': jac9})


def transformationDomainXAn(x):
    """ Transforms the point x given by the steepest ascent method to
        the right domain of x.
        
       Args:
          x: Point to be transformed.
    """
    x4=np.array(numberBikes-np.sum(np.rint(x))).reshape((1,1))
    x=np.concatenate((np.rint(x),x4),1)
    return x

transformationDomainXVn=transformationDomainXAn

def transformationDomainW(w):
    """ Transforms the point w given by the steepest ascent method to
        the right domain of w.
        
       Args:
          w: Point to be transformed.
    """
    return np.rint(w)

def conditionOpt(x):
    """ Gives the stopping rule for the steepest ascent method, e.g.
        the function could be the Euclidean norm. 
        
       Args:
          x: Point where the condition is evaluated.
    """
    return np.max((np.floor(np.abs(x))))


def const1A(x):
    return numberBikes-lower-np.sum(x[0:n1-1])

def jac1A(x):
    return np.array([-1,-1,-1])

def const2A(x):
    return x[0]-lower

def jac2A(x):
    return np.array([1,0,0])

def const3A(x):
    return x[1]-lower

def jac3A(x):
    return np.array([0,1,0])

def const4A(x):
    return x[2]-lower

def jac4A(x):
    return np.array([0,0,1])

def const6A(x):
    return upperX[0]-x[0]

def jac6A(x):
    return np.array([-1,0,0])

def const7A(x):
    return upperX[1]-x[1]

def jac7A(x):
    return np.array([0,-1,0])

def const8A(x):
    return upperX[2]-x[2]

def jac8A(x):
    return np.array([0,0,-1])

def const9A(x):
    return -numberBikes+np.sum(x[0:n1-1])+upperX[3]

def jac9A(x):
    return np.array([1,1,1])



consA=({'type':'ineq',
        'fun': const1A,
       'jac': jac1A},
    {'type':'ineq',
        'fun': const2A,
       'jac': jac2A},
    {'type':'ineq',
        'fun': const3A,
       'jac': jac3A},
    {'type':'ineq',
        'fun': const4A,
       'jac': jac4A},
        {'type':'ineq',
        'fun': const6A,
       'jac': jac6A},
        {'type':'ineq',
        'fun': const7A,
       'jac': jac7A},
        {'type':'ineq',
        'fun': const8A,
       'jac': jac8A},
        {'type':'ineq',
        'fun': const9A,
       'jac': jac9A})

opt=inter.opt(numberRestarts,n1-1,n1-1,transformationDomainXVn,
	      transformationDomainXAn,transformationDomainW,
	      None,functionGradientAscentVn,
              functionGradientAscentAn,conditionOpt,1.0,cons,consA,
	      "SLSQP","SLSQP")


"""
We define the SBO object.
"""
l={}
l['VOIobj']=VOIobj
l['Objobj']=Objective
l['miscObj']=misc
l['optObj']=opt
l['statObj']=stat
l['dataObj']=dataObj


sboObj=SBO.SBO(**l)


"""
We run the SBO algorithm.
"""

sboObj.SBOAlg(numberIterations,nRepeat=1,Train=True)


