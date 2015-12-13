import numpy as np

###m is the length of the vectors for the sum
###all the numbers are saved in result


if __name__ == "__main__":
    n1=4
    f = open(str(4)+"-cluster.txt", 'r')
    cluster=eval(f.read())
    f.close()
    ####
    bikeData=np.loadtxt("bikesStationsOrdinalIDnumberDocks.txt",skiprows=1)
    upperX=np.zeros(n1)
    temBikes=bikeData[:,2]

    for i in xrange(n1):
        temp=cluster[i]
        indsTemp=np.array([a[0] for a in temp])
        upperX[i]=np.sum(temBikes[indsTemp])
        
    result=[]
    np.random.seed(1)
    nBikes=6000
    nPoints=10000
  #  s=np.random.dirichlet(np.ones(4),nPoints)
    s=np.random.uniform(0,1,(nPoints,4))
    lower=100
    s[:,0]=s[:,0]*upperX[0]+(1-s[:,0])*lower
    s[:,0]=np.floor(s[:,0])

    
    for j in range(nPoints):
        s[j,1]=s[j,1]*min(upperX[1],4000-s[j,0])+(1-s[j,1])*lower
        s[j,1]=np.floor(s[j,1])
        s[j,2]=s[j,2]*min(nBikes-s[j,0]-s[j,1]-lower,upperX[2])+(1-s[j,2])*max(nBikes-s[j,0]-s[j,1]-upperX[3],lower)
        s[j,2]=np.floor(s[j,2])
        s[j,3]=nBikes-np.sum(s[j,0:3])

    
    f=open("lowerBoundNewRandompointsPoisson1000.txt","w")
    np.savetxt(f,s)
    f.close()
