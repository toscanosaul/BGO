#!/usr/bin/env python

"""
Queuing simulation based on New York City's Bike system,
in which system users may remove an available bike from
a station at one location within the city, and ride it
to a station with an available dock in some other location
within the city.
"""

import numpy as np
from math import *
from geopy.distance import vincenty
import json
from numpy import linalg as LA
from scipy.stats import poisson
from scipy.stats import rv_discrete
import os
from scipy.sparse import csr_matrix as csr

nBikes=6000
nStations=329
distancesBikeStations=np.loadtxt("distanceBikeStations.txt")

def PoissonProcess(T,lamb,A,N):
    """
    Simulate the poisson process N(T,(i,j)) where (i,j) is in A, and
    N(T,(i,j)) is the poisson process related to the station (i,j);
    and A is a subset of stations. Return the number of times generated.
    
    Output:
        TIME: List with the arrival times for each set A: (A[i],Times)
        nArrivals: Number of arrivals
    
    Args:
        T: Final time of the simulation.
        A: Subsets A(i,j) of pair of bike stations. 
        N: N(T,A)=sum(i,j in A) N(T,(i,j))
        lamb: Vector with the parameters of the poisson processes N(T,(i,j))
                (lamb[i] is related to the ith entry of A)
    """
    n=len(A) ##cardinality of A
    prob=np.zeros(n)
    lambSum=(np.sum(lamb[0][0,:]))
    nElements=len(lamb[0][0,:])
    for j in xrange(nElements):
        prob[lamb[0][2,j]+(lamb[0][1,j])*nStations]=(float(lamb[0][0,j])/(lambSum))
    X=np.random.multinomial(N,prob,size=1)[0]
    nArrivals=np.sum(X)
    TIME=[]
    for i in xrange(n):
        if (X[i]>0):
            unif=np.random.uniform(0,1,X[i])
            temp=np.sort(float(T)*unif)
            TIME.append([A[i],temp])

    ##Time is a list where each entry represets a station A[i]
    return TIME,nArrivals



####No use
def generateSets(n,fil):
    """
    Generates subsets A(i,j) based on the flow between the bike station,
    with their parameters lambda.
    
    Args:
        n: Number of sets generated.
        fil: Name of the file with the parameters of the Poisson processes.
    """
    poiss=np.loadtxt(fil)
    n1=poiss.shape[0]
    A=[[] for i in range(n)]
    lamb=[[] for i in range(n)]
    for i in xrange(n1):
        for j in range(i,n1):
            ind=((i-1)*n1+j)%n
            if (poiss[i,j]>0):
                A[ind].append((i,j))
                lamb[ind].append(poiss[i,j])
            if (i!=j and poiss[j,i]>0):
                A[ind].append((j,i))
                lamb[ind].append(poiss[j,i])
    return A,lamb

def generateParametersPoisson(fil):
    poiss=np.loadtxt(fil)
    n1=poiss.shape[1]
    lamb=[]
    nSets=1
    A=[[] for i in range(nSets)]
    for j in range(n1):
        for k in range(n1):
            lamb.append(poiss[j,k])
            A[0].append((j,k))
    return lamb,A
        

def SimulateNt (A,lamb,T):
    """
    Simulate the random variable N(T,A)=sum(i,j in A) N(T,(i,j)).
    
    Args:
        -A: Subsets A(i,j) based on the flow between the bike stations.
        -lamb: Vector with the parameters of the elements in A
        -T: Time of the variable N(T,(i,j))
    """
#    print lamb
 #   print np.sum(lamb)
    la=np.sum(lamb)
    la=la*T
    res=np.random.poisson(la)
    return res

def startInitialConfiguration (X,m,data,cluster,bikeData,files=False):
    """
    Starts the Initial configuration of the citibike problem. Returns a
    matrix with the number of docks and bikes available.
    
    Args:
        -X: Vector with the initial configuration of bikes.
        -m: Number of groups according to K-means algorithm.
        -data: Contain the latitudes,longitudes,addreses,indexes of the
                bike stationsload; this data is loaded from a json file.
        -cluster: Array read from a txt file. It contains the clusters of
                 the bike stations.
        -bikeData: Matrix with the ID, numberDocks, Latitute,longitude.
        -files: True if we want to save the initial configuration;
                False otherwise.
    """


    A=np.zeros((nStations,2))
    A[:,0]=bikeData[:,2]
   # print A[:,0]
   # for i in range(nStations):
    #    A[:,0]=
    if files:
      f= open(str(m)+"-initialConfiguration.txt", 'w')
      f.write("docks"+","+"bickes"+","+"ID"+","+"total"+","+
              "bikes/total"+","+"latitude"+","+"longitude"+
              ","+"streets")
    for i in range(m):
        temp=cluster[i]
       # setBikes=X[i]/len(temp)
        
        resT=X[i]
        inds=np.array([a[0] for a in temp])
        indx=np.where(A[inds,0]>0)[0]
        nElm=len(indx)
        indx2=inds[indx]
        while (resT>0):
            setBikes=int(resT/nElm)
            if setBikes==0:
               # index2=np.where(A[inds,0]>0)[0]
                A[indx2[0:resT],1]+=1
                A[indx2[0:resT],0]-=1
                break
            index2=np.where((A[indx2,0]-setBikes)<0)[0] #indices where no all bikes can be placed
            index3=set(range(0,nElm))-set(index2) #indices where all bikes can be placed
            index3=np.array(list(index3))
            tempA=A[indx2[index2],0]
            A[indx2[index2],1]+=tempA
            A[indx2[index2],0]-=tempA #docks
            A[indx2[index3],1]+=setBikes
            A[indx2[index3],0]-=setBikes
            
            res2=np.sum(-A[indx2[index2],1]+setBikes)
            resT=(resT%nElm)+res2
            indx=np.where(A[inds,0]>0)[0]
            indx2=inds[indx]
            nElm=len(indx)
     #   print A
    return A

def findBikeStation(state,currentBikeStation):
    """
    Find the closest bike station to currentBikeStation with available docks.
    
    Args:
        state: Matrix with states of all the bike stations.
        currentBikeStation: index of the current bike station.
    """
    dist=distancesBikeStations[currentBikeStation,:]
    sort=[i[0] for i in sorted(enumerate(dist), key=lambda x:x[1])]
    k=1
    while True:
        ind=sort[k]
        if state[ind,0]>0:
            return ind
        else:
            k+=1
    return 0

def unhappyPeople (T,N,X,m,data,cluster,bikeData,parLambda,nDays,A,poissonArray,timesArray,ind,randomSeed=None):
    """
    Counts the number of people who doesn't find a bike or an available dock.
    We divide the bike stations in m groups according to K-means algorithm.
    The bikes are distributed uniformly in each group.
    
    Args:
        T: Time of the simulation.
        N: Vector N(T,A_{i}).
        X: Initial configuration of the bikes.
        m: Number of groups formed with the bike stations.
        lamb: List with vectors of the parameters of the
              poisson processes N(T,(i,j)).
        A: List with all the sets considered.
        date: String: yyyy-mm. We are using the data of that
              date.
        exponentialTimes: Parameters of the exponential distributions.
        data:  Contain the latitudes,longitudes,addreses,indexes of the
               bike stationsload; this data is loaded from a json file.
        cluster: Array read from a txt file. It contains the clusters of
                 the bike stations.
        bikeData: Matrix with the ID, numberDocks, Latitute,longitude.
        ind: Day
    """
    if randomSeed is not None:
        np.random.seed(randomSeed)
   # parLambda=parLambda.astype(int)
  #  probs=poisson.pmf(int(N[0]),mu=np.array(parLambda))
   # probs=probs/np.sum(probs)

 #   ind=np.random.choice(range(nDays),size=1,p=probs)
    exponentialTimes=timesArray[ind][0]
    exponentialTimes2=np.zeros((nStations,nStations))
    nExp=len(exponentialTimes[0,:])
    for i in range(nExp):
        exponentialTimes2[exponentialTimes[1,i],exponentialTimes[2,i]]=exponentialTimes[0,i]
    poissonParam=poissonArray[ind]

    unHappy=0
    state=startInitialConfiguration(X,m,data,cluster,bikeData,nDays)

    nSets=1
    times=[]
    nTimes=0
    for i in range(nSets):
        temp=PoissonProcess(T,poissonParam,A[i],N[i])

        nTimes+=temp[1]
        times.extend(temp[0])

    Times=np.zeros((nTimes,3))
    k=0
    for i in range(len(times)):
        for j in range(len(times[i][1])):
            Times[k,0]=times[i][1][j] #arrival times
            Times[k,1]=times[i][0][0] #station i
            Times[k,2]=times[i][0][1] #station j
            k+=1
    Times=Times[Times[:,0].argsort()]
    currentTime=0
    dropTimes=[]
    for i in xrange(nTimes):
        currentTime=Times[i,0]
        while (dropTimes and currentTime>dropTimes[0][0]):
            if state[dropTimes[0][1],0]>0:
                state[dropTimes[0][1],0]=state[dropTimes[0][1],0]-1
                state[dropTimes[0][1],1]+=1
                dropTimes.pop(0)
            else:
                unHappy+=1
                j=findBikeStation(state,dropTimes[0][1])
                state[j,0]=state[j,0]-1
                state[j,1]=state[j,1]+1
                dropTimes.pop(0)
        bikePickUp=Times[i,1]
        bikeDrop=Times[i,2]

        if state[bikePickUp,1]==0:
            unHappy+=1
            continue
        indi=exponentialTimes[1,]
        timeUsed=np.random.exponential(exponentialTimes2[bikePickUp,bikeDrop])
        dropTimes.append((currentTime+timeUsed,bikeDrop))
        dropTimes=sorted(dropTimes, key=lambda x:x[0])
        
        state[bikePickUp,1]=state[bikePickUp,1]-1
        state[bikePickUp,0]=state[bikePickUp,0]+1
    return -unHappy

def generatePoissonParameters(nDays,nStations):
    parametersLambda=np.zeros(nDays)
    for i in xrange(nDays):
        lamb=[]
        fil="day"+"%d"%i+"PoissonParametersNonHom.txt"
        fil=os.path.join("NonHomegeneousPP",fil)
        poiss=np.loadtxt(fil)
        for j in range(nStations):
            for k in range(nStations):
                lamb.append(poiss[j,k])
        parametersLambda[i]=np.sum(np.array(lamb))
    f=open(os.path.join("NonHomegeneousPP","poissonDays.txt"),'w')
    np.savetxt(f,parametersLambda)
    f.close()

##probability of w
def computeProbability(w,parLambda,nDays):
    probs=poisson.pmf(w,mu=np.array(parLambda))
    probs*=(1.0/nDays)
    return np.sum(probs)

##cum distribution
def cumProba(T,parLambda,nDays,L=650,M=8900):
    w=range(L,M)
    probs=np.zeros(M-L)
    for i in range(M-L):
        probs[i]=computeProbability(w[i],parLambda,nDays)
    return np.sum(probs)

def expectation(z,alpha,parLambda,nDays,L=650,M=8900):
    w=np.array(range(L,M))
    probs=np.zeros(M-L)
    aux=np.exp(-alpha*((z-w)**2))
    for i in range(M-L):
        probs[i]=computeProbability(w[i],parLambda,nDays)
    return np.dot(aux,probs)

def writeProbabilities(poissonParameters,nDays):
    L=650
    M=8900
    wTemp=np.array(range(L,M))
    probsTemp=np.zeros(M-L)
    for i in range(M-L):
        probsTemp[i]=computeProbability(wTemp[i],poissonParameters,nDays)
    f=open(os.path.join("NonHomegeneousPP","probabilitiesExpectations.txt"),'w')
    np.savetxt(f,probsTemp)
    f.close()
    
def writeSparseMatrix(nDays):
    for i in range(nDays):
        fil="day"+"%d"%i+"PoissonParametersNonHom.txt"
        A=np.loadtxt(os.path.join("NonHomegeneousPP",fil))
        fil2="daySparse"+"%d"%i+"PoissonParametersNonHom.txt"
        f=open(os.path.join("NonHomogeneousPP2",fil2),'w')
        A=csr(A)
        temp=A.nonzero()
        temp2=np.array([A.data,temp[0],temp[1]])
        np.savetxt(f,temp2)
        f.close()

        fil="day"+"%d"%i+"ExponentialTimesNonHom.txt"
        A=np.loadtxt(os.path.join("NonHomegeneousPP",fil))
        fil2="daySparse"+"%d"%i+"ExponentialTimesNonHom.txt"
        f=open(os.path.join("NonHomogeneousPP2",fil2),'w')
        A=csr(A)
        temp=A.nonzero()
        temp2=np.array([A.data,temp[0],temp[1]])
        
        np.savetxt(f,temp2)
        f.close()


if __name__ == '__main__':
    np.random.seed(9)
    nSets=1
    nStations=329
    m=4
    nDays=153
    fil="2014modPoissonParametersNonHom.txt"
    date="2014-05"
  #  fil="2014-05PoissonParameters.txt"
    T=4.0 ##the rate of the PP is per hour, so this is from 7:00 to 11:00
    
    N=np.zeros(nSets)
    
    A=[[] for i in range(nSets)]
    for j in range(nStations):
        for k in range(nStations):
            A[0].append((j,k))
    
  #  generatePoissonParameters(nDays,nStations)
    fil="poissonDays.txt"
    fil=os.path.join("NonHomegeneousPP",fil)
    poissonParameters=np.loadtxt(fil)
    
    with open ('json.json') as data_file:
        data=json.load(data_file)
        
    f = open(str(m)+"-cluster.txt", 'r')
    cluster=eval(f.read())
    f.close()
    
    bikeData=np.loadtxt("bikesStationsOrdinalIDnumberDocks.txt",skiprows=1)
    poissonParameters*=T

    for i in range(nSets):
        index=np.random.randint(0,nDays)
        fil="day"+"%d"%index+"PoissonParametersNonHom.txt"
        fil=os.path.join("NonHomegeneousPP",fil)
        fil2="day"+"%d"%index+"ExponentialTimesNonHom.txt"
        exponentialTimes=np.loadtxt(os.path.join("NonHomegeneousPP",fil2))
        lamb,A=generateParametersPoisson(fil)
        N[i]=SimulateNt(A[i],lamb,T)
    #print N[0]
    writeSparseMatrix(nDays)
    #print computeProbability(N[0],T,poissonParameters,nDays)
   # print cumProba(T,poissonParameters,nDays)
   # print expectation(8000,1.0/8000,poissonParameters,nDays)
   # print unhappyPeople (T,N,np.array([1500,1500,1500,1500]),m,
    #                     data,cluster,bikeData,poissonParameters,nDays)
  #  writeProbabilities(poissonParameters,nDays)
    #X=PoissonProcess(T,lamb[0],A[0],N[0])
