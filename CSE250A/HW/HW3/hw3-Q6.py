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
    
#calculate the estimate for B i_val given sample size and B i's. 
def calc(B, PZ_B, i_val):
    rows = len(B)
    sumPZ_B = np.zeros(shape=(rows, 1))
    sumPZ_Bq = np.zeros(shape=(rows, 1))
    sumPZ_B[0] = PZ_B[0];
    sumPZ_Bq[0] = B[0][i_val]*PZ_B[0]
    
    for i in range (1, rows):
        sumPZ_B[i] = PZ_B[i] + sumPZ_B[i-1]
        sumPZ_Bq[i] = (B[i][i_val]*PZ_B[i]) + sumPZ_Bq[i-1]

    result= np.array([sumPZ_Bq[i]/sumPZ_B[i] for i in range(rows)]).flatten()
    return result
    
def plotDict(d, Z, title):
    lists = sorted(d.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.plot(x,y)
    plt.ylabel('P(B[i]=1 | Z=' + str(Z) +')')
    plt.xlabel('Sample Size')
    plt.title(title);
    plt.show()
    
#generate big random observation data and iteratively add them to sample data.
def run(rows, Z_val, i_val, bits_val, alpha, begin_rows, increment, epsilon):
    sample_B = np.random.randint(2, size=(rows, bits_val))
    PZ_B = np.array(PZgivenB(Z_val, sample_B, alpha))
    result = calc(sample_B, PZ_B, i_val)    
    
    curRows = begin_rows    
    PB_Z = dict()
    delta=1000

    while (curRows < rows and delta > epsilon):
        PB_Z[curRows] = result[curRows]
        curRows = curRows + increment
    plotDict(PB_Z, Z_val ,title='Estimate for i='+ str(i_val))

#all indexes are zero reference. So from the Assignment question, if i_val is 2, put 1 here.    
run(rows=2000000,Z_val=128, i_val=9,bits_val=10,alpha=0.2, begin_rows=0, increment = 500, epsilon = 0.000)