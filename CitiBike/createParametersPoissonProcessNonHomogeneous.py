#!/usr/bin/env python

import json
import numpy as np
from geopy.distance import vincenty
from numpy import linalg as LA
import os
from urllib2 import urlopen



###There are 6000 bikes

###We are using ONLY 329 bike stations because the Json file doesn't contain all of them.
###They might not be working at that moment
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
def writeParameters(fil,nHours,nDays,dayDates):
    poisson=np.zeros((nDays,nStations,nStations)) ###parameters of poisson processes
    times=np.zeros((nDays,nStations,nStations)) ##exponential times
    f = open(fil, 'r')
    f.next()
    f.next()
    A=np.loadtxt("bikesStationsOrdinalIDnumberDocks.txt",skiprows=1)
    cont=0
    cont2=0
    for line in f:
        line=line.split(",")
        line[0]=line[0].replace('/','-')
        temp=line[0].split("-")
        if (len(temp[2])==4):
            tempc=list(temp)
            temp[0]=tempc[2]
            temp[1]=tempc[0]
            if len(tempc[0])==1:
                temp[1]="0"+temp[1]
            temp[2]=tempc[1]
            if len(tempc[1])==1:
                temp[2]="0"+temp[2]
            line[0]='-'.join(temp)
        ind=dayDates.index(line[0])
        i1=np.argwhere(A[:,1]==int(line[2]))
        i2=np.argwhere(A[:,1]==int(line[4]))
        if i1 and i2:
            cont2+=1
            i1=i1[0][0]
            i2=i2[0][0]
            poisson[ind,i1,i2]+=1
            temp=times[ind,i1,i2]
            temp1=poisson[ind,i1,i2]-1
            if temp1==0:
                times[ind,i1,i2]=line[1]
            else:
                temp=(float(temp1)/(temp1+1))*(temp+float(line[1])/temp1)
                times[ind,i1,i2]=temp
    poisson=(poisson)*(1/float(nHours))
    times=times/(float(60*60)) #in hours
    os.makedirs("NonHomegeneousPP")
    for i in xrange(nDays):
        path=os.path.join("NonHomegeneousPP","day"+"%d"%i+"PoissonParametersNonHom.txt")
        np.savetxt(path,poisson[i,:,:])
        path=os.path.join("NonHomegeneousPP","day"+"%d"%i+"ExponentialTimesNonHom.txt")
        np.savetxt(path,times[i,:,:])
    
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

def addMonthDic(dayD,day,month,off,year):
    lis31=['07','08','10']
    if month in lis31:
        for i in range(1,32):
            dayD.append(year+"-"+month+"-"+"%02d"%i)
            day[i+off-1]=i+off-1
    else:
        for i in range(1,31):
            dayD.append(year+"-"+month+"-"+"%02d"%i)
            day[i+off-1]=i+off-1

def createDicDays(nDays):
    dayDate=[]
    day=np.zeros(nDays)
    off=[0,31,31,30,31]
    off=np.cumsum(np.array(off))
    for i in range(7,12):
        addMonthDic(dayDate,day,"%02d"%i,off[i-7],"2014")
    return dayDate,day
    
    

if __name__ == '__main__':
   # writeTransformationIDtoOrdinal()
    ##Number of days in the month considered
   
   # nDaysMonth=31
   nHours=4 
   # writeParameters("2014-05 modifiedCitiBikeTripData.txt",nDaysMonth,nHours)
   nDays=153
   dayD,day=createDicDays(nDays)
   
   fil="2014modifiedCitiBikeTripData.txt"
   writeParameters(fil,nHours,nDays,dayD)
   # Kmeans(4,nStations)