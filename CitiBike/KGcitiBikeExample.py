#!/usr/bin/env python

"""
We consider a queuing simulation based on New York City's Bike system,
in which system users may remove an available bike from a station at one
location within the city, and ride it to a station with an available dock
in some other location within the city. The optimization problem that we
consider is the allocation of a constrained number of bikes (6000) to available
docks within the city at the start of rush hour, so as to minimize, in simulation,
the expected number of potential trips in which the rider could not find an
available bike at their preferred origination station, or could not find an
available dock at their preferred destination station. We call such trips
"negatively affected trips".

To use the KG algorithm, we need to create 6 objets:

Objobj: Objective object (See InterfaceSBO).
miscObj: Miscellaneous object (See InterfaceSBO).
VOIobj: Value of Information function object (See VOIGeneral).
optObj: Opt object (See InterfaceSBO).
statObj: Statistical object (See statGeneral).
dataObj: Data object (See InterfaceSBO).

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
from BGO.Source import *

nTemp=int(sys.argv[1])
nTemp2=int(sys.argv[2])
nTemp3=int(sys.argv[3])
nTemp4=int(sys.argv[4]) #number of iterations
nTemp5=sys.argv[5] #True if code is run in parallel; False otherwise.

if nTemp5=='F':
    nTemp5=False
    nTemp6=1
elif nTemp5=='T':
    nTemp6=int(sys.argv[6]) #number of restarts for the optimization method
    nTemp5=True

randomSeed=nTemp
np.random.seed(randomSeed)

print "random seed is"
print nTemp

######

n1=4
n2=1
numberSamplesForG=nTemp3

nDays=153
######

"""
We define the variables needed for the queuing simulation. 
"""

g=unhappyPeople

nSets=4

fil="poissonDays.txt"
fil=os.path.join("NonHomegeneousPP",fil)
poissonParameters=np.loadtxt(fil)

###readData

poissonArray=[[] for i in xrange(nDays)]
exponentialTimes=[[] for i in xrange(nDays)]

for i in xrange(nDays):
    fil="daySparse"+"%d"%i+"ExponentialTimesNonHom.txt"
    fil2=os.path.join("NonHomogeneousPP2",fil)
    poissonArray[i].append(np.loadtxt(fil2))
    
    fil="daySparse"+"%d"%i+"PoissonParametersNonHom.txt"
    fil2=os.path.join("NonHomogeneousPP2",fil)
    exponentialTimes[i].append(np.loadtxt(fil2))

numberStations=329
Avertices=[[]]
for j in range(numberStations):
    for k in range(numberStations):
	Avertices[0].append((j,k))

#A,lamb=generateSets(nSets,fil)

#parameterSetsPoisson=np.zeros(n2)
#for j in xrange(n2):
#    parameterSetsPoisson[j]=np.sum(lamb[j])


#exponentialTimes=np.loadtxt("2014-05"+"ExponentialTimes.txt")
with open ('json.json') as data_file:
    data=json.load(data_file)

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

def sampleFromXAn(n):
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
            s[j,2]=s[j,2]*min(nBikes-s[j,0]-s[j,1]-lower,upperX[2])+(1-s[j,2])*max(nBikes-s[j,0]-s[j,1]-upperX[3],lower)
            s[j,2]=np.floor(s[j,2])
            s[j,3]=nBikes-np.sum(s[j,0:3])
        aux1=np.concatenate((s[:,0:n1-1],aux1),0)
    return aux1

sampleFromXVn=sampleFromXAn

def noisyG(X,n,randSeed=None):
    if len(X.shape)==2:
       X=X[0,:]
    estimator=n
    W,indexes=simulatorW(estimator,True)
    result=np.zeros(estimator)
    for i in range(estimator):
        result[i] = g(TimeHours,W[i,:],X,nSets,
                         data,cluster,bikeData,poissonParameters,nDays,
			 Avertices,poissonArray,exponentialTimes,indexes[i],randSeed)
    return np.mean(result),float(np.var(result))/estimator

def g2(x,w,day,i):
    return g(TimeHours,w,x,nSets,
                         data,cluster,bikeData,poissonParameters,nDays,
			 Avertices,poissonArray,exponentialTimes,day,i)

def estimationObjective(x,N=1000):
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

Objective=inter.objective(g,n1,noisyG,numberSamplesForG,sampleFromXVn,
                          simulatorW,estimationObjective,sampleFromXAn)

"""
We define the miscellaneous object.
"""
parallel=nTemp5

trainingPoints=nTemp2

#nameDirectory="Results"+'%d'%numberSamplesForG+"AveragingSamples"+'%d'%trainingPoints+"TrainingPoints"
#folder=os.path.join(nameDirectory,"KG")

misc=inter.Miscellaneous(randomSeed,parallel,nF=numberSamplesForG,tP=trainingPoints,ALG="KG",prefix="FinalNonHomogeneous112715")

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

dataObj=inter.data(Xtrain,yHist=None,varHist=None)
dataObj.getTrainingDataKG(trainingPoints,noisyG,numberSamplesForG,parallel)

"""
We define the statistical object.
"""

dimensionKernel=n1

scaleAlpha=2000.0

stat=stat.KG(dimKernel=dimensionKernel,numberTraining=trainingPoints,
                scaledAlpha=scaleAlpha, dimPoints=n1,trainingData=dataObj)

"""
We define the VOI object.
"""

pointsVOI=np.loadtxt("lowerBoundNewRandompointsPoisson1000.txt")

voiObj=VOI.KG(numberTraining=trainingPoints,
           pointsApproximation=pointsVOI,dimX=n1)

"""
We define the Opt object.
"""

dimXsteepest=n1-1 #Dimension of x when the VOI and a_{n} are optimized.

def projectGradientDescent(x,direction,xo):
    minx=np.min(x)
    alph=[]
    if (minx < 0):
 	ind=np.where(direction<0)[0]
	quotient=xo[ind].astype(float)/direction[ind]
	alp=-1.0*np.max(quotient)
	alph.append(alp)
    if (np.sum(x[0:n1])>numberBikes):
	if (np.sum(direction[0:n1])>0):
	    alph2=(float(numberBikes)-np.sum(xo[0:n1]))/(np.sum(direction[0:n1]).astype(float))	        
    	    alph.append(alph2)
    if (len(alph)==0):
	return x
    return xo+direction*min(alph)

##EI object
def functionGradientAscentVn(x,grad,VOI,i,L,data,kern,temp1,temp2,a,onlyGrad):
    grad=onlyGrad
    x=np.array(x).reshape([1,n1-1])
    x4=np.array(numberBikes-np.sum(x[0,0:n1-1])).reshape((1,1))
    tempX=x[0:1,0:n1-1]
    x2=np.concatenate((tempX,x4),1)
    temp=VOI.VOIfunc(i,x2,L,data,kern,temp1,temp2,grad,a,onlyGrad)
    if onlyGrad:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
        grad2=np.dot(temp,L)
        return grad2
        
    
    if grad==True:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
        grad2=np.dot(temp[1],L)
        return temp[0],grad2
    else:
        return temp
    
def functionGradientAscentMuN(x,grad,data,stat,i,L,temp1,onlyGrad):
    x=np.array(x).reshape([1,n1-1])
    x4=np.array(numberBikes-np.sum(x[0,0:n1-1])).reshape((1,1))
    x=np.concatenate((x,x4),1)
    temp=stat.muN(x,i,data,L,temp1,grad,onlyGrad)
    if onlyGrad:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
        grad2=np.dot(temp,L)
        return grad2
    if grad:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
        grad2=np.dot(temp[1],L)
        return temp[0],grad2
    else:
        return temp

dimXsteepest=n1-1

def transformationDomainXAn(x):
    x4=np.array(numberBikes-np.sum(np.rint(x))).reshape((1,1))
    x=np.concatenate((np.rint(x),x4),1)
    return x

transformationDomainXVn=transformationDomainXAn


lower=100

def conditionOpt(x):
    return np.max((np.floor(np.abs(x))))

def const1(x):
    return numberBikes-lower-np.sum(x[0:n1-1])

def jac1(x):
    return np.array([-1,-1,-1])


def const2(x):
    return x[0]-lower

def jac2(x):
    return np.array([1,0,0])

def const3(x):
    return x[1]-lower

def jac3(x):
    return np.array([0,1,0])

def const4(x):
    return x[2]-lower

def jac4(x):
    return np.array([0,0,1])


def const6(x):
    return upperX[0]-x[0]

def jac6(x):
    return np.array([-1,0,0])

def const7(x):
    return upperX[1]-x[1]

def jac7(x):
    return np.array([0,-1,0])

def const8(x):
    return upperX[2]-x[2]

def jac8(x):
    return np.array([0,0,-1])

def const9(x):
    return -numberBikes+np.sum(x[0:n1-1])+upperX[3]

def jac9(x):
    return np.array([1,1,1])





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
###returns the value and the variance


opt=inter.opt(nTemp6,dimXsteepest,n1-1,transformationDomainXVn,transformationDomainXAn,None,
              projectGradientDescent,functionGradientAscentVn,
              functionGradientAscentMuN,conditionOpt,1.0,cons,cons,"SLSQP","SLSQP")


#nameDirectory="Results"+'%d'%numberSamplesForG+"AveragingSamples"+'%d'%trainingPoints+"TrainingPoints"

l={}
l['VOIobj']=voiObj
l['Objobj']=Objective
l['miscObj']=misc
l['optObj']=opt
l['statObj']=stat
l['dataObj']=dataObj

kgObj=KG.KG(**l)

kgObj.KGAlg(nTemp4,nRepeat=10,Train=True)

