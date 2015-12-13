#!/usr/bin/env python

import csv
import math
from geopy.distance import vincenty
import datetime
import os
from urllib2 import urlopen
import zipfile
import StringIO

###True if start<=time1<=end1<=end
###Precondition:start<=end,time1<=end1. Format of times: hh:mm:ss and 0<=hh<=23, 0<=mm<=59
###             0<=ss<=59
###If time1>end1 (this can occur if the trip ends on the next day), the function is false
def timeWithinInterval (start,end,time1,end1):
    [hours0, minutes0, seconds0] = [int(x) for x in start.split(':')]
    [hours1, minutes1, seconds1] = [int(x) for x in time1.split(':')]
    x0=datetime.timedelta(hours=hours0, minutes=minutes0, seconds=seconds0)
    x1=datetime.timedelta(hours=hours1, minutes=minutes1, seconds=seconds1)
    
    if x1<x0:
        return False
    
    [hours0, minutes0, seconds0] = [int(x) for x in end.split(':')]
    [hours1, minutes1, seconds1] = [int(x) for x in end1.split(':')]
    x0=datetime.timedelta(hours=hours0, minutes=minutes0, seconds=seconds0)
    x2=datetime.timedelta(hours=hours1, minutes=minutes1, seconds=seconds1)
    
    if x2<x1:
        return False
    
    if x2<=x0:
        return True
    

def writeline(line,start,end,f1):
    temp=line[1].split(" ") #time of starting the trip
    temp2=line[2].split(" ") #time of ending the trip
    if timeWithinInterval(start,end,temp[1],temp2[1]):
        f1.write(",\n")
        f1.write(temp[0]+",")
        f1.write(line[0]+",")
        f1.write(line[3]+",")
        dist=vincenty((float(line[5]),float(line[6])),(float(line[9]),float(line[10])))
        f1.write(str(dist).split()[0]+",")
        f1.write(line[7])
    

###Creates modifiedCitiBikeTripData.txt with the tripduration, start station id, distance, end station id.
###The data is within the range of start and end which are hours
###fil is the name of the file
###Precondition: start, end are in the format hh:mm:ss and 0<=hh<=23, 0<=mm<=59,0<=ss<=59
###             fil is a string
def readFile(start,end,fil,writeFil):
  #  f = open(fil, 'rU')
    zfile=zipfile.ZipFile(fil)
    data = StringIO.StringIO(zfile.read(zfile.namelist()[0]))
    #f1= open("modifiedCitiBikeTripData"+".txt","w")
  #  f1.write("from "+start+" to "+end+"\n")
    readd=csv.reader(data)
    header=readd.next()
   # fl.write("day"+",")
   # f1.write(header[0]+",")
   # f1.write(header[3]+",")
   # f1.write("Distance (km)"+",")
   # f1.write(header[7])
    for line in readd:
        writeline(line,start,end,writeFil)
   # f.close()
   # f1.close()
   
def readAllFiles(start,end,listfil):
    f1= open("modifiedCitiBikeTripData"+".txt","w")
    f1.write("from "+start+" to "+end+"\n")
    f1.write("day"+",")
    f1.write("tripduration"+",")
    f1.write("start station id"+",")
    f1.write("Distance (km)"+",")
    f1.write("end station id,")
    for f in listfil:
        print f
        readFile(start,end,f,f1)
        print "done"
    f1.close()
    
def downloadFile(url):
    print url
    f=urlopen(url)
    print "downloading " + url
    
    with open(os.path.basename(url),"w") as local_file:
        local_file.write(f.read())
    
def downloadFiles():
    prefix="https://s3.amazonaws.com/tripdata/"
    post="-citibike-tripdata.zip"
    
    years=["2013","2014","2015"]
    for j in range(0,3):
        for i in range(7,12):
            if i<10:
                downloadFile(prefix+years[j]+"0"+str(i)+post)
            else:
                downloadFile(prefix+years[j]+str(i)+post)
                

def getNames():
    post="-citibike-tripdata.zip"
    years=["2013","2014","2015"]
    listFil=[]
    for j in range(0,2):
        for i in range(7,12):
            if i<10:
                temp=years[j]+"0"+str(i)+post
            else:
                temp=years[j]+str(i)+post
            listFil.append(temp)
    j=2
    for i in range(7,11):
        if i<10:
            temp=years[j]+"0"+str(i)+post
        else:
            temp=years[j]+str(i)+post
        listFil.append(temp)
    return listFil

def subset(year,file1,start,end):
    f2=open(file1,'r')
    f1=open(str(year)+file1,'w')
    l=f2.next()
    f1.write("from "+start+" to "+end+"\n")
    f1.write("day"+",")
    f1.write("tripduration"+",")
    f1.write("start station id"+",")
    f1.write("Distance (km)"+",")
    f1.write("end station id,")
    f2.next()
    
    for line in f2:
        temp=line.split(",")
        temp[0]=temp[0].replace('/','-')
        temp=temp[0].split("-")
        if int(temp[0])==year or int(temp[2])==year:
            f1.write(line)
    
    f1.close()
    f2.close()

if __name__ == '__main__':
   # names=getNames()
    start="07:00:00"
    end="11:00:00"
   # readAllFiles(start,end,names)
    subset(2014,"modifiedCitiBikeTripData.txt",start,end)
    
   # read("07:00:00","11:00:00","2014-05 - Citi Bike trip data.csv")