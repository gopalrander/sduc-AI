# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 21:33:02 2016

@author: gopal
"""
import numpy as np
nasdaq2k = np.loadtxt('nasdaq00.txt') #249
nasdaq2k1 = np.loadtxt('nasdaq01.txt') #248

d = 4
# xn = linear combination of xn-1, xn-2, xn-3, xn-4
A = np.zeros(shape=(d,d), dtype='float')
B = np.zeros(d, dtype='float')
for i in range(d,len(nasdaq2k)):
    X = np.array(nasdaq2k[i-d:i])
    A+=np.outer(X,X)
    B+= nasdaq2k[i] * X

a = np.linalg.solve(A,B)
print('a=', a)

def Prob(y, X, W):
    Py_x = 1/(np.sqrt(2 * np.pi)) * np.exp( - (y-W*X)**2 / 2)
    return Py_x
    
mse2k =0.0
for i in range (d, len(nasdaq2k)):
    X = np.array(nasdaq2k[i-d:i])
    Y = nasdaq2k[i]
    mse2k = mse2k + ((Y-np.dot(a,X))**2)
    
print('Mean Squared error on data from year 2000:', mse2k)

mse2k = mse2k / (len(nasdaq2k)-d)

mse2k1 =0.0
for i in range (d, len(nasdaq2k1)):
    X = np.array(nasdaq2k[i-d:i])
    Y = nasdaq2k1[i]
    mse2k1 = mse2k1 + ((Y-np.dot(a,X))**2)

mse2k = mse2k1 / (len(nasdaq2k1)-d)
print('Mean Squared error on data from year 2001:', mse2k1)
