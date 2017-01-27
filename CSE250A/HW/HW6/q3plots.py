# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 20:15:00 2016
@author: gopal
"""
import numpy as np
import matplotlib.pyplot as plt 
def g(x):
    result = 0.0
    for k in range(10):
        result += np.log(np.math.cosh(x+ (1/(k+1))))
    return result/10;

def gd(x):
    result=0.0
    for k in range(10):
        result += np.tanh(x + (1/(k+1)))
    return result/10;
    
def f(x):
    return np.log(np.cosh(x));

def fd(x):
    return np.tanh(x);

def fdd(x):
    return 1/(np.cosh(x) * np.cosh(x))

def Q(x,y):
    return f(y)+fd(y)*(x-y) + np.power((x-y),2)/2    

def ConvergerAux(x0, loops, func, funcd):
    x = x0;
    it = np.zeros(loops, dtype=int)
    xVal = np.zeros(loops)
    for i in range(loops):
        xVal[i]= x
        it[i] = i
        print (it[i], x)
        x = x - funcd(x)
    plt.plot(it, xVal)
    plt.xlabel("iterations(n)")
    plt.ylabel("x_n")
    print (x, func(x))

def ConvergerNewton(x0, loops):
    x = x0;
    it = np.zeros(loops, dtype=int)
    xVal = np.zeros(loops)
    for i in range(loops):
        xVal[i]= x
        it[i] = i
        print (it[i], x)
        x = x - (fd(x)/fdd(x))
    plt.plot(it, xVal)
    plt.xlabel("iterations(n)")
    plt.ylabel("x_n")
    print (x, f(x))
    
def plot(l,r,step,func, y0=0):
    xRange = np.arange(l,r,step)
    if func==Q:
        y = np.array([func(x,y0) for x in xRange])
    else:
        y = np.array([func(x) for x in xRange])
    plt.plot(xRange,y)
    plt.xlabel("x")
    ylabel = func.__name__ + '(x'
    if y0!=0: ylabel+= (', '+str(y0))
    ylabel += ')'                               
    plt.ylabel(ylabel)
    
def plotAll(l,r,step):
    plot(l,r,step,f, y0=0)
    plot(l,r,step,Q, y0=1)
    plot(l,r,step,Q, y0=-2)
        