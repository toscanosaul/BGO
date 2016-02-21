#!/usr/bin/env python

import os
import numpy as np

def createNewFilesFunc(path,rs):
    f=open(os.path.join(path,'%d'%rs+"hyperparameters.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"XHist.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"yhist.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"varHist.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"optimalSolutions.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"optimalValues.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"optVOIgrad.txt"),'w')
    f.close()
    f=open(os.path.join(path,'%d'%rs+"optAngrad.txt"),'w')
    f.close()
    
def writeTraining(ALGObj):
    tempDir=os.path.join(ALGObj.path,'%d'%ALGObj.miscObj.rs+"XHist.txt")
    with open(tempDir, "a") as f:
        np.savetxt(f,ALGObj.dataObj.Xhist)
    tempDir=os.path.join(ALGObj.path,'%d'%ALGObj.miscObj.rs+"yhist.txt")
    with open(tempDir, "a") as f:
        np.savetxt(f,ALGObj.dataObj.yHist)
    tempDir=os.path.join(ALGObj.path,'%d'%ALGObj.miscObj.rs+"varHist.txt")
    with open(tempDir, "a") as f:
        np.savetxt(f,ALGObj.dataObj.varHist)
        
        
def writeNewPointSBO(ALGObj,optim):
    temp=optim.xOpt
    gradOpt=optim.gradOpt
    numberIterations=optim.nIterations
    gradOpt=np.sqrt(np.sum(gradOpt**2))
    gradOpt=np.array([gradOpt,numberIterations])
    
    tempArg=optim.xOpt[0:1,0:ALGObj.opt.dimXsteepestVn]
    xTrans=ALGObj.opt.transformationDomainXVn(tempArg)
    
    tempArg=optim.xOpt[0:1,ALGObj.opt.dimXsteepestVn:ALGObj.opt.dimXsteepestVn
                                                       +ALGObj._dimW]
    wTrans=ALGObj.opt.transformationDomainW(tempArg)
    temp=np.concatenate((xTrans,wTrans),1)
    ALGObj.dataObj.Xhist=np.vstack([ALGObj.dataObj.Xhist,temp])
    y,var=ALGObj.Obj.noisyF(temp,ALGObj.Obj.numberEstimateF)
    ALGObj.dataObj.yHist=np.vstack([ALGObj.dataObj.yHist,y])
    ALGObj.dataObj.varHist=np.append(ALGObj.dataObj.varHist,var)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.miscObj.rs+
                           "varHist.txt"), "a") as f:
        var=np.array(var).reshape(1)
        np.savetxt(f,var)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.miscObj.rs+
                           "yhist.txt"), "a") as f:
        y=np.array(y).reshape(1)
        np.savetxt(f,y)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.miscObj.rs+
                           "XHist.txt"), "a") as f:
        np.savetxt(f,temp)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.miscObj.rs+
                           "optVOIgrad.txt"), "a") as f:
        np.savetxt(f,gradOpt)
    ALGObj.optRuns=[]
    ALGObj.optPointsArray=[]
    
def writeNewPointKG(ALG,optim):
    temp=optim.xOpt
    gradOpt=optim.gradOpt
    numberIterations=optim.nIterations
    gradOpt=np.sqrt(np.sum(gradOpt**2))
    gradOpt=np.array([gradOpt,numberIterations])
    xTrans=ALG.opt.transformationDomainXVn(
        optim.xOpt[0:1,0:ALG.opt.dimXsteepestVn])

    temp=xTrans

    ALG.dataObj.Xhist=np.vstack([ALG.dataObj.Xhist,temp])

    y,var=ALG.Obj.noisyF(temp,ALG.Obj.numberEstimateF)
    ALG.dataObj.yHist=np.vstack([ALG.dataObj.yHist,y])

    ALG.dataObj.varHist=np.append(ALG.dataObj.varHist,var)

    with open(os.path.join(ALG.path,'%d'%ALG.miscObj.rs+
                           "varHist.txt"), "a") as f:
        var=np.array(var).reshape(1)
        np.savetxt(f,var)
    with open(os.path.join(ALG.path,'%d'%ALG.miscObj.rs+
                           "yhist.txt"), "a") as f:
        y=np.array(y).reshape(1)
        np.savetxt(f,y)
    with open(os.path.join(ALG.path,'%d'%ALG.miscObj.rs+
                           "XHist.txt"), "a") as f:
        np.savetxt(f,temp)
    with open(os.path.join(ALG.path,'%d'%ALG.miscObj.rs+
                           "optVOIgrad.txt"), "a") as f:
        np.savetxt(f,gradOpt)
    ALG.optRuns=[]
    ALG.optPointsArray=[]
    
def writeSolution(ALGObj,optim):
    temp=optim.xOpt
    tempGrad=optim.gradOpt
    tempGrad=np.sqrt(np.sum(tempGrad**2))
    tempGrad=np.array([tempGrad,optim.nIterations])
    xTrans=ALGObj.opt.transformationDomainXAn(
        optim.xOpt[0:1,0:ALGObj.opt.dimXsteepestAn])
    ALGObj._solutions.append(xTrans)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.miscObj.rs+
                           "optimalSolutions.txt"), "a") as f:
        np.savetxt(f,xTrans)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.miscObj.rs+
                           "optimalValues.txt"), "a") as f:
        result,var=ALGObj.Obj.estimationObjective(xTrans[0,:])
        res=np.append(result,var)
        np.savetxt(f,res)
    with open(os.path.join(ALGObj.path,'%d'%ALGObj.miscObj.rs+
                           "optAngrad.txt"), "a") as f:
        np.savetxt(f,tempGrad)
    ALGObj.optRuns=[]
    ALGObj.optPointsArray=[]