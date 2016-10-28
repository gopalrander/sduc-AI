# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 23:56:30 2016

@author: gopal
"""
import numpy as np
import matplotlib.pyplot as plt
def prepData():
    train3 = np.loadtxt('train3.txt', dtype=int)
    train5 = np.loadtxt('train5.txt', dtype=int)
    test3 = np.loadtxt('test3.txt', dtype=int)
    test5 = np.loadtxt('test5.txt', dtype=int)
    trainlabel3 = np.zeros(len(train3), dtype=int)
    trainlabel5 = np.ones(len(train5), dtype=int)
    testlabel3 = np.zeros(len(test3), dtype=int)
    testlabel5 = np.ones(len(test5), dtype=int)    
    trainSet = np.append(train3, train5, axis=0)
    trainLabel = np.append(trainlabel3, trainlabel5)
    testSet = np.append(test3, test5, axis=0 )
    testLabel = np.append(testlabel3, testlabel5)
    return trainSet, trainLabel, testSet, testLabel
    
trainData, trainLabel, testData, testLabel = prepData()

def sigmoid(z):
    return 1/(1+np.exp(-z))

def derivatives(y,W,X):
    sigma = sigmoid(np.dot(W,X))
    gradient = (y-sigma)*X 
    hesian = -(sigma*(1-sigma))*np.outer(X,X) 
    return gradient, hesian

def logLikelihood(data, label, W):
    llhood = 0.0
    for i in range(len(data)):
        llhood += ((label[i]*np.log(sigmoid(np.dot(W, data[i])))) + ((1-label[i])*np.log(sigmoid(-np.dot(W, data[i])))))
    return llhood

def Classify(data, W):
    res = np.dot(data, W)
    result =np.zeros(len(res))
    for i in range(len(res)):
        if res[i]>=0 :
            result[i] = 1
    return result
    
W = np.zeros(trainData.shape[1])
iterations = 20
loops = np.array([])
llhoods = np.array([])
errorRates = np.array([])

for loop in range(iterations):
    G = np.zeros(trainData.shape[1])
    H = np.zeros(shape=(trainData.shape[1], trainData.shape[1]))    
    for i in range(len(trainData)):
        gradient, hesian = derivatives(trainLabel[i], W, trainData[i])
        G = G + gradient
        H = H + hesian
    W = W - np.linalg.solve(H, G)
    llhood = logLikelihood(trainData, trainLabel, W)
    classify = Classify(trainData, W)
    errorRate = 100*np.count_nonzero(classify-trainLabel)/len(trainLabel)
    errorRates = np.append(errorRates, [errorRate], axis=0)
    loops = np.append(loops, [loop], axis=0)
    llhoods = np.append(llhoods, [llhood], axis=0)
    

plt.figure()
f, axes = plt.subplots(1,1)
#axes.plot(loops, llhoods)
#axes.set_xlabel("Iterations")
#axes.set_ylabel("Log Likelihood")

axes.plot(loops, errorRates)
axes.set_xlabel("Iterations")
axes.set_ylabel("Error rate (%)")

W_mesh = np.reshape(W, (8,8))

testClassify = Classify(testData, W)
testErrorRate = 100*np.count_nonzero(testClassify-testLabel)/len(testLabel)
