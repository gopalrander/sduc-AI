import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt('data1.txt')
#data1 = np.loadtxt('data2.txt')
m = np.shape(data1)[1]; #number of features (included the extra constant)
n = np.shape(data1)[0]; #number of data points

X = np.column_stack((np.array(data1[:,:m-1]), np.ones(len(data1))))
Y = np.array(data1[:,m-1])
XX = np.inner(X,X)
alpha = np.zeros(n)
#XXRBF = np.array()

def PhiStore(i, j, d):
    return np.power(XX[i][j] + 1, d)
def Phi(X, Y, d):
    return np.power((np.inner(X,Y) + 1), d)
def RBFKernal(X,Y,sigma):
    return np.exp(-(np.power(np.linalg.norm((X-Y), 2),2) / (2*sigma*sigma)))

sigma = 10
l =1
c = np.ones(1)
w = np.zeros(m)
W = np.zeros(m)
Wavg = np.zeros(m)
index = np.arange(n)
T = 10
while T>=0:
    randomOrder = np.random.permutation(index)
    for i in randomOrder:
        if (np.inner(X[i],w))*Y[i] <= 0 :
            W = np.row_stack((W,w))
            c = np.append(c,l)
            Wavg = Wavg + w*l
            w = w + Y[i]*X[i]
            l=1
        else:
            l=l+1
        ksum = 0    
        for j in range(n):
            #ksum = ksum + (alpha[j]*Y[j]*PhiStore(i,j,sigma))
            ksum = ksum + (alpha[j]*Y[j]*RBFKernal(X[i],X[j],sigma))
        if ksum*Y[i] <= 0:
            alpha[i]+=1
    T-=1 
    #print ("mistakes =", mistakes)            

def countMiss(W, X, Y):
    return np.array([np.count_nonzero(np.multiply(np.inner(X,W).T, Y)[i] <= 0) for i in range(len(W))])
    
def VotedPerceptron(W,c,x):
    return np.sign(np.sum(c*np.sign(np.inner(x, W))))

def AveragePerceptron(Wavg,c, x):
    return np.sign(np.inner(x,Wavg))

def KernelPerceptron(alpha, X, x):
    return np.sign(np.sum(np.array([alpha[j]*Y[j]*Phi(X[j],x,sigma) for j in range(len(alpha))])))

def KernelRBF(alpha, X, x):
    return np.sign(np.sum(np.array([alpha[j]*Y[j]*RBFKernal(X[j],x,sigma) for j in range(len(alpha))])))
    
def PlotF(W, c, X, l,r,step, func):
    x1Range = np.arange(l,r,step)
    x2Range = np.arange(l,r,step)
    for x1 in x1Range:
        for x2 in x2Range:
            if func(W, c, [x1, x2, 1]) > 0:
                plt.plot(x1, x2, 'b.', alpha=0.1, fillstyle='full', markeredgecolor='blue', markeredgewidth=0.0)
            else:
                plt.plot(x1, x2, 'r.', alpha=0.1, fillstyle='full', markeredgecolor='red', markeredgewidth=0.0)
    PlotData(X)
        
def PlotK(alpha, X, l ,r, step, func):
    x1Range = np.arange(l,r,step)
    x2Range = np.arange(l,r,step)
    for x1 in x1Range:
        for x2 in x2Range:
            if func(alpha, X, [x1, x2, 1]) > 0:
                plt.plot(x1, x2, 'b.', alpha=0.1, fillstyle='full', markeredgecolor='blue', markeredgewidth=0.0)
            else:
                plt.plot(x1, x2, 'r.', alpha=0.1, fillstyle='full', markeredgecolor='red', markeredgewidth=0.0)
    PlotData(X)

def PlotData(X):
    for i in range(len(X)):
        if (Y[i] > 0):
            plt.plot(X[i][0], X[i][1], 'b+')
        else:
            plt.plot(X[i][0], X[i][1], 'r+')
   