import numpy as np

###m is the length of the vectors for the sum
###all the numbers are saved in result


if __name__ == "__main__":
    result=[]
    nBikes=6000
    
    n1=4
    f = open(str(4)+"-cluster.txt", 'r')
    cluster=eval(f.read())
    f.close()
    ####
    bikeData=np.loadtxt("bikesStationsOrdinalIDnumberDocks.txt",skiprows=1)
    upperX=np.zeros(n1)
    temBikes=bikeData[:,2]
    #######
    for i in xrange(n1):
        temp=cluster[i]
        indsTemp=np.array([a[0] for a in temp])
        upperX[i]=np.sum(temBikes[indsTemp])
    #####
    nI=int(max(upperX))
    upperX=upperX.astype(int)
    configurations=[]
    f=open("pointsPoissonNew.txt","w")
    f.close()
    lower=1000
    for i in xrange(lower,upperX[0]+1):
        for j in xrange(lower,upperX[1]+1):
            for k in xrange(max(nBikes-i-j-upperX[3],lower),min(nBikes-i-j-lower,upperX[2])+1):
                l=nBikes-i-j-k
                with open("pointsPoissonNew.txt","a") as f:
                    f.write("%d " %i)
                    f.write("%d " %j)
                    f.write("%d " %k)
                    f.write("%d\n" %l)
                    f.close()
   # l=np.linspace(1,max(upperX),max(upperX))
   # a,b,c= np.meshgrid(l,l,l)
   # sum1=a+b+c
   # ind=np.where(sum1<=nBikes)
   # a1=a[ind]
   # a2=b[ind]
   # a3=c[ind]
   # n1=len(a1)
   # print n1
   # a1=a1.reshape((n1,1))
   # a2=a2.reshape((n1,1))
   # a3=a3.reshape((n1,1))
   # a4=nBikes-a1-a2-a3
   # t=np.concatenate((a1,a2,a3,a4),1)
    
    
    
