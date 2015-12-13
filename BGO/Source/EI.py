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

class EI:
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


    def EIAlg(self,m,nRepeat=1,Train=True,**kwargs):
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
    def optimizeVOI(self,start, i,L,temp1,maxObs):
      #  opt=op.OptSteepestDescent(n1=self.opt.dimXsteepest,projectGradient=self.opt.projectGradient,
	#			  xStart=start,xtol=self.opt.xtol,stopFunction=self.opt.functionConditionOpt)

        def g(x,grad,onlyGradient=False):
            return self.opt.functionGradientAscentVn(i,x,grad,maxObs,self.stat._k,
						     self.dataObj.Xhist, L,temp1,
						     onlyGradient)
        if self.opt.MethodVn=="SLSQP":
	    opt=op.SLSP(start)
	    def g1(x):
		return -1.0*g(x,grad=False)
	    def dg(x):
		return -1.0*g(x,grad=True,onlyGradient=True)
	    cons=self.opt.consVn
	    opt.run(f=g1,df=dg,cons=cons)

	self.optRuns.append(opt)

        
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
	
        vec=np.zeros(tempN)
        X=self.dataObj.Xhist
        for i in xrange(tempN):
            vec[i]=self.stat.muN(X[i,:],n,L,X,temp1)
        maxObs=np.max(vec)
	args['maxObs']=maxObs

	return args

    def optVOInoParal(self,i):
	n1=self._n1
	args3=self.getParametersOptVoi(i)
	Xst=self.Obj.sampleFromXVn(1)
	st=Xst[0:1,:]
	self.optRuns.append(misc.VOIOptWrapper(self,st,**args3))
	fl.writeNewPointKG(self,self.optRuns[0])
	


    def optVOIParal(self,i,nStart,numProcesses=None):
        try:
            n1=self._n1
          #  n2=self._dimW
         #   dim=self.dimension
	    args3=self.getParametersOptVoi(i)
	    Xst=self.Obj.sampleFromXVn(nStart)
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
       # opt=op.OptSteepestDescent(n1=self.opt.dimXsteepest,projectGradient=self.opt.projectGradient,
#				  xStart=start,xtol=self.opt.xtol,stopFunction=self.opt.functionConditionOpt)
        tempN=i+self.numberTraining
        def g(x,grad,onlyGradient=False):
            return self.opt.functionGradientAscentAn(x,i,L,self.dataObj.Xhist,temp1,grad,
						       onlyGradient)

	if self.opt.MethodAn=="SLSQP":
	    opt=op.SLSP(start)
	    def g1(x):
		return -1.0*g(x,grad=False)
	    def dg(x):
		return -1.0*g(x,grad=True,onlyGradient=True)
	    cons=self.opt.consAn
	    opt.run(f=g1,df=dg,cons=cons)
	    
        self.optRuns.append(opt)

    
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
	Xst=self.Obj.sampleFromXAn(1)
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
	    Xst=self.Obj.sampleFromXAn(nStart)
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
        
