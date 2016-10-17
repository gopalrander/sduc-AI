# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 19:48:25 2016

@author: gopal
"""

import numpy as np
import matplotlib.pyplot as plt

def fB(B):
    n = len(B)
    b=0;
    for i in range(n):
        if(B[i] == 1):
            b = b + np.power(2,i)
    return b
    
#Z = constant, B = Matrix, alpha = noise
def PZgivenB(Z, B, alpha):
    ret = [(np.power(alpha, np.abs(Z-fB(B[i])))) * (1-alpha)/(1+alpha) for i in range(len(B))];
    return ret
    
    
def calc(B, Z, i_val, alpha):
    PZ_B = PZgivenB(Z, B, alpha)
    n = len(PZ_B)
    numerator = sum(PZ_B[i] for i in range(n) if B[i][i_val]==1) 
    denominator = sum(PZ_B[i] for i in range(n))
    return numerator/denominator
    
def plotDict(d):
    lists = sorted(d.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    for i in range(len(x)):
        print (x[i], '\t', y[i])
    
def run(rows, Z_val, i_val, bits_val, alpha):
    sample_B = np.random.randint(2, size=(rows, bits_val))
    curRows =1
    PB_Z = dict()
    while (curRows <= rows):
        subSample_B = sample_B[:curRows, :]
        PB_Z[curRows] = calc(subSample_B, Z_val, i_val, alpha)
        curRows = curRows*2
    plotDict(PB_Z)

#all indexes are zero reference. So from the Assignment question, if i_val is 2, put 1 here..    
run(rows=2048,Z_val=128, i_val=1,bits_val=10,alpha=0.2)
    