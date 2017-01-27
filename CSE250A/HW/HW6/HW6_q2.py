# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:32:29 2016

@author: gopal
"""

import numpy as np

OX = np.loadtxt('spectX.txt', dtype=int)
OY = np.loadtxt('spectY.txt', dtype=int)

PZ_X = np.zeros(OX.shape[1], dtype=int)
PZ_X = PZ_X + (1/len(PZ_X))

#P(Y|X)
def getPY_X(PZ_X, X, Y):
    result = 1.0
    for i in range(len(PZ_X)):
        result = result * (np.power((1-PZ_X[i]), X[i]))
    if(Y==1):
        result = 1-result
    return result

def LLhood(OX, OY, PZ_X):
    llhood = 0.0
    for i in range (len(OX)):
        llhood = llhood + np.log(getPY_X(PZ_X, OX[i], OY[i]))
    llhood = llhood/len(OX)
    return llhood

def countMistakes(PY_X):
    return np.count_nonzero(PY_X<0.5)

def run(OX,OY,PZ_X):
    #PZX_XY E-step
    for loop in range(257):
        PY_X = np.array([getPY_X(PZ_X, OX[i],OY[i]) for i in range(len(OX))])
        a = (OX.T/PY_X).T * PZ_X
        PnewZ_X = np.inner(OY, a.T)    
        PnewZ_X = np.array([PnewZ_X[i]/np.count_nonzero(OX.T[i]) for i in range(len(PnewZ_X))])
        if (loop & loop-1) == 0:
            print (loop, countMistakes(PY_X), LLhood(OX, OY, PZ_X))
        PZ_X = PnewZ_X #M-step