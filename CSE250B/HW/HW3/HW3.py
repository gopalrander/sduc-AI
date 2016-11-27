# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:16:51 2016

@author: gopal
"""
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lrs
from sklearn import preprocessing
def OneTimePreprocessAndRandomize():
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
    np.savetxt('X_Train.txt', X_Train)
    np.savetxt('Y_Train.txt', Y_Train)
    np.savetxt('X_Test.txt', X_Test)
    np.savetxt('Y_Test.txt', Y_Test)

X_Train = np.loadtxt('X_Train.txt')
X_Test = np.loadtxt('X_Test.txt')
Y_Train = np.loadtxt('Y_Train.txt')
Y_Test = np.loadtxt('Y_Test.txt')

def PY_X_star(W, x, y, k):
    num= np.exp(np.inner(W[y-1], x))
    den = np.sum(np.array([np.exp(np.inner(W[i], x)) for i in range(k)]))
    return num/den

def PY_X(W, x, y, k):
    num = np.exp(np.inner(W, Phi(x,y,k)))
    den = np.sum(np.array([np.exp(np.inner(W , Phi(x,i,k))) for i in range(1, k+1, 1)]))
    return num/den
        
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
    der-= W
    return der;
    
def LLhood_star(W,X,Y, k):
    llhood = 0
    for j in range(len(X)):
        llhood+= PY_X_star(W, X[j], Y[j],k)
    return llhood;
    
def LLhood(W,X,Y,k):
    llhood= 0;
    for j in range(len(X)):
        llhood += PY_X(W,X[j], Y[j], k)
    return llhood;
    
    
def run(T, eta, isRandomIndex=False):
    #----Logistic regression multiclass----#
    clf = lrs(solver='lbfgs', max_iter=T, random_state=0, multi_class='multinomial').fit(X_Train,Y_Train)
    print("training score : %.3f" % (clf.score(X_Train, Y_Train)))
    print("test score : %.3f" % (clf.score(X_Test, Y_Test)))
    W_base = (clf.coef_)

    llhood_star = LLhood_star(W_base, X_Train, Y_Train, 3)
    print ("Logistic Loss :", -llhood_star)

    #----Coordinate Descent Method---#
    W = np.zeros((np.shape(X_Train)[1]* 3))
    X_axis=[]
    Y_axisTrain=[]
    Y_axisTest=[]
    Y_axis_LLhood = []
    
    for t in range(T):
        #for i in range(len(W)):
        der = derivative(W, X_Train, Y_Train, 3)#[len(W)-1-i]
        if(isRandomIndex):
            index = np.random.randint(len(W))
            label = "random-feature coordinate descent"
        else:
            index = np.argmax(np.abs(der))
            label = "coordinate descent"
        
        X_axis+=[t]
        Y_axisTrain+=[Classify(W,X_Train, Y_Train,3)*100/len(Y_Train)]
        Y_axisTest += [Classify(W,X_Test, Y_Test,3)*100/len(Y_Test)]   
        Y_axis_LLhood += [-LLhood(W, X_Train, Y_Train, 3)]
        
        W[index] += eta* der[index]
        
        #eta[index] /= 1.1
        #print(index)
        T-=1
    print ("Loss in Coordinate Descent :", Y_axis_LLhood[T-1])
    
    #plt.plot(X_axis, Y_axisTrain, label = label)
    plt.plot(X_axis, Y_axisTest, label = label)
    #plt.plot(X_axis, Y_axis_LLhood, label = label)
    plt.xlabel("Iterations")
    plt.ylabel("Test Error")
    return Y_axisTest, Y_axisTrain

run(500, 0.05)
#run(200, 0.05, True) 
plt.legend(["coordinate descent", "random-feature coordinate descent"])