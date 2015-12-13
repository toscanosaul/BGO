#!/usr/bin/env python
import numpy as np
import os
import json
from scipy.stats import poisson

n1=4
n2=2


nDays=153
"""
We define the variables needed for the queuing simulation. 
"""


#fil="2014-05PoissonParameters.txt"
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
    
    
prob=0
U=np.zeros(nDays)
L=np.zeros(nDays)
for i in xrange(nDays):
    temp=poisson.ppf([.95],poissonParameters[i])[0]
    temp2=poisson.ppf([.001],poissonParameters[i])[0]
    U[i]=temp
    L[i]=temp2

print L
print U

f=open("percentilesDays.txt",'w')
np.savetxt(f,L)
np.savetxt(f,U)
f.close()

prob=0
for i in xrange(nDays):
    for j in range(int(L[i]),int(U[i])+1):
        prob+=poisson.pmf(j,mu=poissonParameters[i])

print prob/nDays

