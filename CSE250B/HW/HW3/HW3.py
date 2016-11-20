# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:16:51 2016

@author: gopal
"""
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lrs
from sklearn import preprocessing

X = np.loadtxt('wine.data', delimiter=',')
X = X[np.random.permutation(np.shape(X)[0])]
Y = X[:,0]
X = np.column_stack((X[:,1:np.shape(X)[1]], np.ones(np.shape(X)[0])))
X_Train = X[:128, :]
Y_Train = Y[:128]
X_Test = X[128:, :]
Y_Test = Y[128:]
X_Train = preprocessing.scale(X_Train, axis=0)
X_Test = preprocessing.scale(X_Test, axis=0)
clf = lrs(solver='lbfgs', max_iter=1000, random_state=0, multi_class='multinomial').fit(X_Train,Y_Train)
print("training score : %.3f" % (clf.score(X_Train, Y_Train)))
print("test score : %.3f" % (clf.score(X_Test, Y_Test)))
W_base = (clf.coef_)


RicherX = np.zeros(3*np.shape(X_Train)[1])
RicherW = np.zeros(3*np.shape(X_Train)[1])
RicherY = [0]
for i in range(len(X_Train)):
    temp = np.zeros(np.shape(X_Train)[1])
    y = np.array(Y_Train[i] == [1,2,3], dtype=int)
    
    RicherX = np.row_stack((RicherX, np.append(X[i], np.append(temp, temp))))   
    #elif Y_Train[i] == 2:
    RicherX = np.row_stack((RicherX, np.append(temp, np.append(X[i], temp))))
    #elif Y_Train[i] == 3:
    RicherX = np.row_stack((RicherX, np.append(temp, np.append(temp,X[i]))))         
    RicherY = np.concatenate((RicherY, y));
RicherX = RicherX[1:,:]
RicherY = RicherY[1:]

clf = lrs(solver='lbfgs', max_iter=1000, random_state=0).fit(RicherX,RicherY);
RicherW = (clf.coef_)

def PY_X(W, x, y, k):
    num = np.exp(np.inner(W, Phi(x,y,k)))
    den = np.sum(np.array([np.exp(np.inner(W , Phi(x,i,k))) for i in range(1, k+1, 1)]))
    return (num/den)

def Y_it(yt, i):
    if yt==i:
        return 1
    else:
        return 0
        
def Classify (W, X, Y, k):
    classified = np.zeros(len(X))
    for i in range(len(X)):
        classified[i] = np.argmax(np.array([PY_X(W, X[i],j, k) for j in range(1,k+1,1)]))+1
    return np.count_nonzero(classified-Y)
    
def Phi(x, y, maxClasses):
    return np.concatenate((np.concatenate((np.zeros(len(x)*(y-1)),x)), np.zeros(len(x)*(maxClasses-y)))) 

def derivative(W,X,Y,k):
    der = np.zeros(len(W))
    for j in range(len(X)):
        der += Phi(X[j],Y[j],k);
        der -= np.sum(np.array([PY_X(W,X[j],i,k) *  Phi(X[j],i,k) for i in range(1, k+1, 1)]), axis=0)
    return der;
    
W = np.zeros((np.shape(X_Train)[1]* 3))

T=1000
X_axis=[]
Y_axis=[]
Y_axis2=[]
eta = np.ones((np.shape(X_Train)[1]* 3))
for t in range(T):
    #for i in range(len(W)):
    der = derivative(W, X_Train, Y_Train, 3)#[len(W)-1-i]
    W + der
    index = np.argmax(np.abs(der)* eta)
    
    #    W[len(W)-1-i] += der_i
    X_axis+=[t]
    Y_axis+=[Classify(W,X_Train, Y_Train,3)*100/len(Y_Train)]
    Y_axis2 += [Classify(W,X_Test, Y_Test,3)*100/len(Y_Test)]
    W_last = W
    W[index] += eta[index]* der[index]
    eta[index] /= (1.2)
    #print(index)
    T-=1
plt.plot(X_axis, Y_axis)
plt.plot(X_axis, Y_axis2)
'''
while T>0 :
    for i in range(len(W)-1):
        dl = np.zeros(3)
        for t in range(len(X_Train)):
            yt = Y_Train[t]-1
            for j in range(3):        
                dl[j] += (Y_it(yt, j) - PY_X(W, X_Train[t], yt))* X_Train[t][i]
            #if (np.abs(dl[j]) >= 1): print (" no  yay!!")
            for j in range(3):
                W[i][j] += (0.001)*(dl[j]/len(X_Train))
    T-=1
'''    