#!/usr/bin/env python
"""
This file includes the optimization methods used such as
gradient ascent (maximizes) and BFGS (minimizes).
"""


import numpy as np
from scipy.optimize import *
from warnings import warn
from math import *
from scipy import linalg
from numpy import linalg as LA


class Optimization:
    def __init__(self,xStart):
	"""
	General class for any optimization method used.
	
	Args:
	    -xStart: Starting point of the algorithms.
	"""
        self.optMethod=None
        self.xStart=xStart
	
        self.xOpt=None #Optimal point
        self.fOpt=None #Optimal value
        self.status=None 
        self.gradOpt=None #Derivative at the optimum
        self.nIterations=None

    def run(self, **kwargs):
        self.opt(**kwargs)
    
        
    def opt(self,f=None,fp=None,cons=None):
	"""
	Optimizes f.
	
	Args:
	    f: Objective function.
	    fp: Derivative of the function.
	"""
        raise NotImplementedError, "optimize needs to be implemented"
    

class SLSP(Optimization):
    def __init__(self,xStart):
	Optimization.__init__(self,xStart)
	self.Name="SLSP"
	
    def opt(self,f=None,df=None,cons=None):
        statuses = ['Converged', 'Maximum number of f evaluations reached', 'Error']

	optResult=minimize(f,self.xStart,jac=df,constraints=cons,method='SLSQP')

        self.xOpt=np.array(optResult.x).reshape([1,len(optResult.x)])


        self.fOpt = -1.0*optResult.fun
        self.gradOpt=optResult.jac
        self.nIterations=optResult.nit
       # self.status=statuses[optResult[2]['warnflag']]
	return 0


class OptBFGS(Optimization):
    def __init__(self, xStart,maxfun=1e4,gtol=None,bfgsFactor=None):
        Optimization.__init__(self,xStart)
	"""
	This is the class of the bfgs algorithm.
	
	Args:
	    -maxfun: Maximum number of evaluations of the objective.
	    -gtol: Tolerance of the change of the gradient.
	    -bfgsFactor= Parameter related to the bfgs method.
	    
	"""
        self.Name="bfgs"
	self.gtol=gtol
	self.bfgsFactor=bfgsFactor
	self.maxFun=maxfun
    
    def opt(self,f=None,df=None):
	"""
	Minimizes f.
	
	Args:
	    -f: Objective function.
	    -df: Derivative of the function.
	"""
        assert df!=None, "Derivative is necessary"
        
        statuses = ['Converged', 'Maximum number of f evaluations reached', 'Error']
    
        dictOpt={}
        if self.gtol is not None:
            dictOpt['pgtol']=self.gtol
        if self.bfgsFactor is not None:
            dictOpt['factr']=self.bfgsFactor
            
        optResult=fmin_l_bfgs_b(f,self.xStart,df,maxfun=self.maxFun,**dictOpt)
  
        self.xOpt=optResult[0]
        self.fOpt = optResult[1]
        self.status=statuses[optResult[2]['warnflag']]
        
        
class OptSteepestDescent(Optimization):
    def __init__(self,stopFunction,n1,maxIters=1e3,xtol=None,
		 projectGradient=None, *args, **kwargs):
        Optimization.__init__(self,*args,**kwargs)
	"""
	Gradient Ascent algorithm.
	
	Args:
	    stopFunction: Gives the stopping rule, e.g. the
			  function could be the Euclidean norm.
			  Its arguments is:
			    -x: Point where the condition is evaluated.
	    n1: Dimension of the domain of the objective function.
	    maxIters: Maximum number of iterations.
	    xtol: Tolerance in the change of points of two consecutive
		  iterations.
	    projectGradient: Project a point x to the domain of the problem
			     at each step of the gradient ascent method if
			     needed. Its argument is:
				    x: The point that is projected.
	"""
        self.Name="steepest"
        self.maxtry=25
        self.n1=n1
	self.maxIters=maxIters
	self.xtol=xtol
	if stopFunction is None:
	    stopFunction=LA.norm
	self.stopFunction=stopFunction
        self.projectGradient=projectGradient
        
        #Golden Section
    # Here q=(x,w)=(xNew,wNew) are vectors of inputs into Vn, (ql,qr)=xl,xr,wl,wr are upper and lower
    # values for x,w resp., also given as vectors. However, the function only works
    # on the dimension dim
    #fn is the function to optimize
    def goldenSection(self,fn,q,ql,qr,dim,tol=1e-8,maxit=100):
        gr=(1+sqrt(5))/2
        ql=np.array(ql)
        ql=ql.reshape(ql.size)
        qr=np.array(qr).reshape(qr.size)
        q=np.array(q).reshape(q.size)
        pl=q
        pl[dim]=ql[dim]
        pr=q
        pr[dim]=qr[dim]
        pm=q
        pm[dim]=pl[dim]+(pr[dim]-pl[dim])/(1+gr)
        FL=fn(pl)
        FR=fn(pr)
        FM=fn(pm)
        tolMet=False
        iter=0
        while tolMet==False:
            iter=iter+1
            if pr[dim]-pm[dim] > pm[dim]-pl[dim]:
                z=pm+(pr-pm)/(1+gr)
                FY=fn(z)
                if FY>FM:
                    pl=pm
                    FL=FM
                    pm=z
                    FM=FY
                else:
                    pr=z
                    FR=FY
            else:
                z=pm-(pm-pl)/(1+gr)
                FY=fn(z)
                if FY>FM:
                    pr=pm
                    FR=FM
                    pm=z
                    FM=FY
                else:
                    pl=z
                    FL=FY
            if pr[dim]-pm[dim]< tol or iter>maxit:
                tolMet=True
        return pm
    
    #we are optimizing in X+alpha*g2
    ##guarantee that the point is in the compact set ?
    def goldenSectionLineSearch (self,fns,tol,maxtry,X,g2):
         # Compute the limits for the Golden Section search
        ar=np.array([0,2*tol,4*tol])
        fval=np.array([fns(0),fns(2*tol),fns(4*tol)])
        tr=2
        while fval[tr]>fval[tr-1] and tr<maxtry:
            ar=np.append(ar,2*ar[tr])
            tr=tr+1
            fval=np.append(fval,fns(ar[tr]))
        if tr==maxtry:
            al=ar[tr-1]
            ar=ar[tr]
        else:
            al=ar[tr-2]
            ar=ar[tr]
        ##Now call goldensection for line search
        if fval[tr]==-float('inf'):
            return 0
        else:
            return self.goldenSection(fns,al,al,ar,0,tol=tol)
    
    def steepestAscent(self,f):
	"""
	Steepest Ascent algorithm.
	
	Args:
	    -f: objective function and its gradient.
		Its arguments are:
		    x: Point where the function is evaluated.
		    grad: True if we want the gradient; False otherwise.
		    onlyGradient: True if we only want the gradient; False otherwise.
	"""
        xStart=self.xStart
        tol=self.xtol
        maxit=self.maxIters
        maxtry=self.maxtry
        tolMet=False
        iter=0
        X=xStart
        g1=-100
        n1=self.n1
        while tolMet==False:
            iter=iter+1
            oldEval=g1
            oldPoint=X
            g1,g2=f(X,grad=True)
            if (tolMet==True):
                break
            def fns(alpha,X_=oldPoint,g2=g2):
                tmp=X_+alpha*g2
                return f(tmp,grad=False)
	    def fLine(x):
		x=x.reshape((1,len(x)))
		z=-1.0*f(x,grad=False)
		return z
            def gradfLine(x):
		x=x.reshape((1,len(x)))
   		df=f(x,grad=True,onlyGradient=True)
                df=df.reshape((1,x.shape[1]))
		z=-1.0*df[0,:]
		return z
  	    g2=g2.reshape((1,len(oldPoint[0,:])))

	    lineSearch2=line_search(fLine,gradfLine,oldPoint[0,:],g2[0,:])
            step=lineSearch2[0]
            if step is None:
	       print "step is none"
	       tolMet=True
               g1,g2=f(X,grad=True)
	       return X,g1,g2,iter
            X=X+lineSearch2[0]*g2
            X[0,:]=self.projectGradient(X[0,:],g2[0,:],oldPoint[0,:])
            if self.stopFunction(X[0,:]-oldPoint[0,:])<tol or iter > maxit:
                tolMet=True
                g1,g2=f(X,grad=True)
                return X,g1,g2,iter
                
    def opt(self,f=None,df=None):
	"""
	Runs the steepest ascent method.
	
	Args:
	    -f: objective function and its gradient.
		Its arguments are:
		    x: Point where the function is evaluated.
		    grad: True if we want the gradient; False otherwise.
		    onlyGradient: True if we only want the gradient; False otherwise.
	
	"""
        x,g,g1,it=self.steepestAscent(f)
        self.xOpt=x
        self.fOpt =g
        self.gradOpt=g1
        self.nIterations=it
        

def getOptimizationMethod(x):
    """
    Get the optimization method.
    
    Args:
	-x: String with the name of the method, e.g.
	    'bfgs','steepest'.
    """
    optimizers={'bfgs': OptBFGS,'steepest':OptSteepestDescent}
    
    for optMethod in optimizers.keys():
        if optMethod.lower().find(x.lower()) != -1:
            return optimizers[optMethod]
        
    raise KeyError('No optimizer was found matching the name: %s' % x)
