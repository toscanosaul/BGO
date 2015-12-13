#!/usr/bin/env python

import numpy as np

def logSumExp(x):
    """
    Computes log*sum(exp(x)) for a vector x, but in numerically careful way
    """
    maxAbs=np.max(np.abs(x))
    if maxAbs>max(x):
        c=np.min(x)
    else:
        c=np.max(x)
    y=c+np.log(np.sum(np.exp(x-c)))
    return y

def kernOptWrapper(m,start,**kwargs):
    """
    This function just wraps the optimization procedure of a kernel
    object so that optimize() pickleable (necessary for multiprocessing).
    
    Args:
        m: kernel object
    """
    m.optimizeKernel(start=start,**kwargs)
    return m.optRuns[-1]

def VOIOptWrapper(m,start,**kwargs):
    """
    This function just wraps the optimization procedure of a kernel
    object so that optimize() pickleable (necessary for multiprocessing).
    
    Args:
        m: global optimization object.
    """
    m.optimizeVOI(start=start,**kwargs)
    return m.optRuns[-1]

def AnOptWrapper(m,start,**kwargs):
    """
    This function just wraps the optimization procedure of a kernel
    object so that optimize() pickleable (necessary for multiprocessing).
    
    Args:
        m: global optimization object.
    """
    m.optimizeAn(start=start,**kwargs)
    return m.optRuns[-1]
