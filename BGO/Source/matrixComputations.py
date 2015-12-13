#!/usr/bin/env python

import numpy as np
from scipy import linalg

##Computes the product: v^T.((L.L^T)^-1).w
##w and v can be matrices.
##L is lower triangular
##If w=None, computes ((L.L^T)^-1).v
def tripleProduct(v,L,w):
    temp=linalg.solve_triangular(L,w,lower=True)
    alp=linalg.solve_triangular(L.T,v.T,lower=False)
    res=np.dot(alp,temp)
    return res

##Computes:(L.L^T)^-1).v
def inverseComp(L,v):
    temp=linalg.solve_triangular(L,v,lower=True)
    alp=linalg.solve_triangular(L.T,temp,lower=False)
    return alp


