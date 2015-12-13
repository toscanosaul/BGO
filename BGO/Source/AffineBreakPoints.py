#!/usr/bin/env python

import numpy as np
import random

# Prepares vectors for passing to AffineEmaxBreakpoints, changing their
# order and removing elements with duplicate slope.
def AffineBreakPointsPrep(a,b):
    #a,b are numpy vectors
    
    #PF: Experimental preprocessing step, which I hope will remove
    # a large number of the entries.
    b1, i1=min(b), np.argmin(b) # [a1,b1] is best at z=-infinity
    a1=a[i1]
    a2,i2=max(a), np.argmax(a) # [a2,b2] is best at z=0
    b2=b[i2]
    b3,i3=max(b), np.argmax(b) #[a3,b3] is best at z=+infinity
    a3=a[i3]
    
    cleft = (a.astype(np.float)-a1)/(b1-b) #instersection with leftmost line
    cright = (a.astype(np.float)-a3)/(b3-b) #instersection with rightmost line
    c2left = (a2-a1)/(b1-b2) #instersection with leftmost line
    c2right = (a2-a3)/(b3-b2) #instersection with rightmost line
    keep = [i for i in xrange(len(b)) if b[i]==b1 or b[i]==b3 or cleft[i] <= c2left or cright[i] >=c2right]
    
    a = a[keep]
    b = b[keep]
    keep1=np.array(keep)
    del cleft,cright,keep
    
    # Form a matrix for which ba(x,1) is the slope b(x) and ba(x,2) is the
    # y-intercept a(x).  Sort this matrix in ascending order of slope, 
    # breaking ties in slope with the y-intercept.  
    ba = np.reshape(np.concatenate((b,a)),(len(a),2),order='F')
    ind=np.lexsort((ba[:,1],ba[:,0]))
    ba = ba[ind,:]
    keep1=keep1[ind]
    a = ba[:,1];
    b = ba[:,0];
    
    # Then, from each pair of indices with the b component equal, remove
    # the one with smaller a component.  This code works because the sort
    # above enforced the condition: if b(i) == b(i+1), then a(i) <= a(i+1).
    ind = []
    tmp = np.diff(b)
    keep = [i for i in xrange(len(tmp)) if tmp[i]!=0]
    keep.append(len(tmp))
    a = a[keep]
    b = b[keep]
    keep2 =keep
    keep1=keep1[keep2]
    return a,b,keep1

# Inputs are two M-vectors, a and b.
# Requires that the b vector is sorted in increasing order.
# Also requires that the elements of b all be unique.

# The output is an (M+1)-vector c and a vector A ("A" is for accept).  Think of
# A as a set which is a subset of {1,...,M}.  This output has the property
# that, for any i in {1,...,M} and any real number z,
#   i \in argmax_j a_j + b_j z
# iff
#  i \in A and z \in [c(j+1),c(i+1)],
#   where j = sup {0,1,...,i-1} \cap A.

# A note about indexing:
#For the vectors a and b,we need to reference a_1 or b_1 as a_0 or b_0, so we reference
# a_i and b_i by a[i-1] and b[i-1] respectively.
# A[0] is not important: it was defined by convenience

def AffineBreakPoints(a,b):
    # Preallocate for speed.  Instead of resizing the array A whenever we add
    # to it or delete from it, we keep it the maximal size, and keep a length
    # indicator Alen telling us how many of its entries are good.  When the
    # function ends, we remove the unused elements from A before passing
    # it.
    
    M=len(a)
    c=np.zeros(M+1)
    A=np.zeros(M+1)
    
    # Step 0
    i=0;
    c[i] = -float('Inf');
    c[i+1] = float('Inf');
    A[1] = 1;
    Alen = 1;
    
    for i in xrange(1,M):
        c[i+1]=float('Inf')
        while True :
            j=A[Alen] #jindex=Alen
            c[j]=(a[j-1]-a[i])/(b[i]-b[j-1])
            if len (A) >1 and c[j]<=c[A[Alen-1]]:
                Alen=Alen-1 #Remove last element j
                #continue in while(True) loop
            else:
                break #quite while(True) loop
        
        A[Alen+1]=i+1
        Alen=Alen+1
    
    Alen=Alen+1
    A=A[1:Alen]-1
    return A,c

if __name__ == "__main__":
    np.random.seed(10)
    a=np.random.uniform(1,100,5)
    b=np.random.uniform(1,100,5)
    print "\n"
    print a
    print b
    print "\n"
    l=AffineBreakPointsPrep(a,b)
    l2=AffineBreakPoints(l[0],l[1])
    print "\n"
    print l
    print "\n"
    print l2
    print "\n"


