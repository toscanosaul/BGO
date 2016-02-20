
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



def computeProbability(w,parLambda,nDays):
    probs=poisson.pmf(w,mu=np.array(parLambda))
    probs*=(1.0/nDays)
    return np.sum(probs)



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
