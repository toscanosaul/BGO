import numpy as np
from math import *
import sys
from . import optimization as op
import multiprocessing as mp
import os
from . import misc
from matplotlib import pyplot as plt
from numpy import multiply
from numpy.linalg import inv
from . import stat
from . import VOI
from . import files as fl
from AffineBreakPoints import *
from scipy.stats import norm
import pylab as plb
from scipy import linalg

class KG:
    def __init__(self, Objobj,miscObj,VOIobj,optObj,statObj,dataObj):
	self.dataObj=dataObj
	self.stat=statObj
	self._VOI=VOIobj
	self.opt=optObj
	self.miscObj=miscObj
	self.Obj=Objobj
	
	self._n1=Objobj.dimSeparation
	self.numberTraining=statObj._numberTraining
	
	self.path=os.path.join(miscObj.folder,'%d'%miscObj.rs+"run")
        if not os.path.exists(self.path):
            os.makedirs(self.path)
    
	self._solutions=[]
        self._valOpt=[]
        self.optRuns=[]
        self.optPointsArray=[]


    def KGAlg(self,m,nRepeat=1,Train=True,**kwargs):
	if self.miscObj.create:
	    fl.createNewFilesFunc(self.path,self.miscObj.rs)
	fl.writeTraining(self)
        if Train is True:
	    self.trainModel(numStarts=nRepeat,**kwargs)
        for i in range(m):
            print i
	    if self.miscObj.parallel:
		self.optVOIParal(i,self.opt.numberParallel)
	    else:
		 self.optVOInoParal(i)

            print i
	    if self.miscObj.parallel:
		self.optAnParal(i,self.opt.numberParallel)
	    else:
		self.optAnnoParal(i)
            print i
	if self.miscObj.parallel:
	    self.optAnParal(i,self.opt.numberParallel)
	else:
	    self.optAnnoParal(i)
        
    ###start is a matrix of one row
    ###
    def optimizeVOI(self,start, i,L,temp1,temp2,a):
        opt=op.OptSteepestDescent(n1=self.opt.dimXsteepest,projectGradient=self.opt.projectGradient,
				  xStart=start,xtol=self.opt.xtol,stopFunction=self.opt.functionConditionOpt)

        def g(x,grad,onlyGradient=False):
            return self.opt.functionGradientAscentVn(x,grad,self._VOI,i,L,self.dataObj,self.stat._k,
						     temp1,temp2,a,
						     onlyGradient)
        opt.run(f=g)
        self.optRuns.append(opt)
        xTrans=self.opt.transformationDomainX(opt.xOpt[0:1,0:self.opt.dimXsteepest])
        self.optPointsArray.append(xTrans)
        
    def getParametersOptVoi(self,i):
	tempN=self.numberTraining+i
	args={}
	args['i']=i
	A=self.stat._k.A(self.dataObj.Xhist[0:tempN,:],noise=self.dataObj.varHist[0:tempN])
        L=np.linalg.cholesky(A)
	args['L']=L
	
	muStart=self.stat._k.mu
	y=self.dataObj.yHist
	temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True)
	args['temp1']=temp1
	
	m=self._VOI._points.shape[0]
	temp2=np.zeros((m,tempN))
	
	X=self.dataObj.Xhist
	B=np.zeros((m,tempN))
	for i in xrange(tempN):
	    B[:,i]=self.stat._k.K(self._VOI._points,X[i:i+1,:])[:,0]
	
	a=np.zeros(m)
	for i in xrange(m):
	    temp2[i,:]=linalg.solve_triangular(L,B[i,:].T,lower=True)
	    a[i]=muStart+np.dot(temp2[i,:],temp1)

	args['temp2']=temp2

	args['a']=a
	return args

    def optVOInoParal(self,i):
	n1=self._n1
	args3=self.getParametersOptVoi(i)
	Xst=self.Obj.sampleFromX(1)
	st=Xst[0:1,:]
	self.optRuns.append(misc.VOIOptWrapper(self,st,**args3))
	fl.writeNewPointKG(self,self.optRuns[0])
	
        self.optRuns=[]
        self.optPointsArray=[]

    def optVOIParal(self,i,nStart,numProcesses=None):
        try:
            n1=self._n1
          #  n2=self._dimW
         #   dim=self.dimension
	    args3=self.getParametersOptVoi(i)
	    Xst=self.Obj.sampleFromX(nStart)
            jobs = []
            pool = mp.Pool(processes=numProcesses)
            for j in range(nStart):
                job = pool.apply_async(misc.VOIOptWrapper, args=(self,Xst[j:j+1,:],), kwds=args3)
                jobs.append(job)
            
            pool.close()  # signal that no more data coming in
            pool.join()  # wait for all the tasks to complete
        except KeyboardInterrupt:
            print "Ctrl+c received, terminating and joining pool."
            pool.terminate()
            pool.join()

        numStarts=nStart
     #   print jobs[0].get()
        for j in range(numStarts):
            try:
                self.optRuns.append(jobs[j].get())
            except Exception as e:
                print "Error optimizing VOI"
                
        if len(self.optRuns):
            j = np.argmax([o.fOpt for o in self.optRuns])
	    fl.writeNewPointKG(self,self.optRuns[j])
        self.optRuns=[]
        self.optPointsArray=[]
        
        
    def optimizeAn(self,start,i,L,temp1):
        opt=op.OptSteepestDescent(n1=self.opt.dimXsteepest,projectGradient=self.opt.projectGradient,
				  xStart=start,xtol=self.opt.xtol,stopFunction=self.opt.functionConditionOpt)
        tempN=i+self.numberTraining
        def g(x,grad,onlyGradient=False):
            return self.opt.functionGradientAscentAn(x,grad,self.dataObj,
						       self.stat,i,L,temp1,onlyGradient)
        opt.run(f=g)
        self.optRuns.append(opt)
        xTrans=self.opt.transformationDomainX(opt.xOpt[0:1,0:self.opt.dimXsteepest])
        self.optPointsArray.append(xTrans)
    
    def optAnnoParal(self,i):
	tempN=self.numberTraining+i
	n1=self._n1
	args3={}
	args3['i']=i

	A=self.stat._k.A(self.dataObj.Xhist[0:tempN,:],noise=self.dataObj.varHist[0:tempN])
	L=np.linalg.cholesky(A)
	
	args3['L']=L
	
	muStart=self.stat._k.mu
	y=self.dataObj.yHist[0:tempN,:]
	temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True)
	args3['temp1']=temp1
	Xst=self.Obj.sampleFromX(1)
	self.optRuns.append(misc.AnOptWrapper(self,Xst[0:1,:],**args3))
	fl.writeSolution(self,self.optRuns[0])
    
    def optAnParal(self,i,nStart,numProcesses=None):
        try:
	    tempN=self.numberTraining+i
            n1=self._n1
	    args3={}
	    args3['i']=i

	    A=self.stat._k.A(self.dataObj.Xhist[0:tempN,:],noise=self.dataObj.varHist[0:tempN])
	    L=np.linalg.cholesky(A)
	    
	    args3['L']=L
	    
	    muStart=self.stat._k.mu
	    y=self.dataObj.yHist[0:tempN,:]
	    temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True)
	    args3['temp1']=temp1
	    Xst=self.Obj.sampleFromX(nStart)
            jobs = []
            pool = mp.Pool(processes=numProcesses)
            
            for j in range(nStart):
                job = pool.apply_async(misc.AnOptWrapper, args=(self,Xst[j:j+1,:],), kwds=args3)
                jobs.append(job)
            
            pool.close()  # signal that no more data coming in
            pool.join()  # wait for all the tasks to complete
        except KeyboardInterrupt:
            print "Ctrl+c received, terminating and joining pool."
            pool.terminate()
            pool.join()
     
        numStarts=nStart
        
        for j in range(numStarts):
            try:
                self.optRuns.append(jobs[j].get())
            except Exception as e:
                print "Error optimizing An"

                
        if len(self.optRuns):
            j = np.argmax([o.fOpt for o in self.optRuns])
	    fl.writeSolution(self,self.optRuns[j])
            
        self.optRuns=[]
        self.optPointsArray=[]
    
    def trainModel(self,numStarts,**kwargs):
	if self.miscObj.parallel:
	    self.stat._k.train(scaledAlpha=self.stat.scaledAlpha,
			       numStarts=numStarts,**kwargs)
	else:
	    self.stat._k.trainnoParallel(scaledAlpha=self.stat.scaledAlpha,**kwargs)
        
        f=open(os.path.join(self.path,'%d'%self.miscObj.rs+"hyperparameters.txt"),'w')
        f.write(str(self.stat._k.getParamaters()))
        f.close()
        
