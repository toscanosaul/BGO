#!/usr/bin/env python

import json
import numpy as np
from geopy.distance import vincenty
from numpy import linalg as LA

###There are 600 bikes

###We are using ONLY 329 bike stations because the Json file doesn't contain all of them.
###They might are not working at the moment
nStations=329



###Write a matrix in IDtoOrdinal with columns: Ordinal-ID-NumberDocks-Latitude-Longitude; 
###The oridinals are according to the order of the Json file
###Write the matrix B with the distances between the bike Stations
def writeTransformationIDtoOrdinal ():
    with open ('json.json') as data_file:
        data=json.load(data_file)
  #  f1= open("IDtoOrdinal"+".txt","w")
    numberStations=len(data["stationBeanList"])
    A=np.zeros((numberStations,5))
    B=np.zeros((numberStations,numberStations))
    for i in xrange(numberStations):
        A[i,0]=i
        A[i,1]=int(data["stationBeanList"][i]["id"])
        A[i,2]=int(data["stationBeanList"][i]["totalDocks"])
        A[i,3]=float(data["stationBeanList"][i]["latitude"])
        A[i,4]=float(data["stationBeanList"][i]["longitude"])
        for j in range(i,numberStations):
            latit=float(data["stationBeanList"][j]["latitude"])
            longi=float(data["stationBeanList"][j]["longitude"])
            B[i,j]=float(str(vincenty((A[i,3],A[i,4]),(latit,longi))).split()[0])
            B[j,i]=B[i,j]
    A.astype(int)
    np.savetxt("bikesStationsOrdinalIDnumberDocks.txt",A,fmt='%i',header="Number,ID,NumberDocks,Latitude,Longitude")
    np.savetxt("distanceBikeStations.txt",B)

##Write poisson parameters in Poisson.txt
##Write mean of Times in exponentialTimes.txt
##nDaysMonth is the number of days in the month considered
##nHours is the number of hours considered in the day
def writeParameters(fil,nDaysMonth,nHours):
    poisson=np.zeros((nStations,nStations)) ###parameters of poisson processes
    times=np.zeros((nStations,nStations)) ##exponential times
    f = open(fil, 'r')
    f.next()
    f.next()
    A=np.loadtxt("bikesStationsOrdinalIDnumberDocks.txt",skiprows=1)
    cont=0
    cont2=0
    for line in f:
        line=line.split(",")
        i1=np.argwhere(A[:,1]==int(line[1]))
        i2=np.argwhere(A[:,1]==int(line[3]))
        if i1 and i2:
            cont2+=1
            i1=i1[0][0]
            i2=i2[0][0]
            poisson[i1,i2]+=1
            temp=times[i1,i2]
            temp1=poisson[i1,i2]
            if temp1==0:
                times[i1,i2]=line[0]
            else:
                temp=(float(temp1)/(temp1+1))*(temp+float(line[0])/temp1)
                times[i1,i2]=temp
    poisson=(1/float(nDaysMonth))*(poisson)*(1/float(nHours))
    times=times/(float(60*60))
    np.savetxt(fil[0:7]+"PoissonParameters.txt",poisson)
    np.savetxt(fil[0:7]+"ExponentialTimes.txt",times)
    
##K-means algorithm for the bikeStations
##n is the number of clusters
def Kmeans(n,nStations):
    with open ('json.json') as data_file:
        data=json.load(data_file)
    temp=np.arange(nStations)
    np.random.shuffle(temp)
    x=temp[0:n]
    seeds=np.zeros((n,2))
    past=np.zeros((n,2))
    for i in xrange(n):
        seeds[i,0]=float(data["stationBeanList"][x[i]]["latitude"])
        seeds[i,1]=float(data["stationBeanList"][x[i]]["longitude"])
    cluster=[[] for i in range(n)]
    dist=np.zeros(n)
    
    iter=0
    while (True):
        for i in range(nStations):
            a1=float(data["stationBeanList"][i]["latitude"])
            a2=float(data["stationBeanList"][i]["longitude"])
            for j in range(n):
                dist[j]=float(str(vincenty((a1,a2),(seeds[j,0],seeds[j,1]))).split()[0])
            j1=np.argmin(dist)
            cluster[j1].append([i,a1,a2])
        for j in range(n):
            meanlat=0
            meanlon=0
            for k in range(len(cluster[j])):
                meanlat=meanlat+cluster[j][k][1]
                meanlon=meanlon+cluster[j][k][2]
            past[j,0]=seeds[j,0]
            past[j,1]=seeds[j,1]
            seeds[j,0]=float(meanlat)/len(cluster[j])
            seeds[j,1]=float(meanlon)/len(cluster[j])
        if (LA.norm(seeds-past,1)<0.00000001):
            break
        cluster=[[] for i in range(n)]
        iter+=1
    
    f = open(str(n)+"-"+"cluster.txt", 'w')
    f.write(repr(cluster))
    f.close()
    
    
if __name__ == '__main__':
    writeTransformationIDtoOrdinal()
    ##Number of days in the month considered
    nDaysMonth=31
    nHours=4 
    writeParameters("2014-05 modifiedCitiBikeTripData.txt",nDaysMonth,nHours)
    Kmeans(4,nStations)