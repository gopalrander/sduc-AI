"""
Created on Sun Oct 23 00:32:51 2016
@author: gopal
"""

import numpy as np;
import math;
import matplotlib.pyplot as plt;
 
vocab = np.loadtxt('vocab250A.txt', dtype=str, unpack=True)
vocabCount = np.loadtxt('unigram.txt', dtype=float, unpack=True)
preWordId, nextWordId, followCount = np.loadtxt('bigram.txt', dtype=int, unpack=True)

#returns index of a word in vocab. fixes the byte encoding b' match.
def WhatIndex(s):
    s = "b'" + s + "'"
    result = np.where(vocab==s)[0]
    if len(result) == 0 :
        return -1
    else:
        return result[0]

lenVocab = np.size(vocab);
totalWords = np.sum(vocabCount)
unigramP = np.divide(vocabCount, totalWords)

bigramP = np.zeros(shape=(lenVocab, lenVocab), dtype=float)

for i in range(len(preWordId)):
    bigramP[preWordId[i]-1][nextWordId[i]-1] = followCount[i]
bigramP = bigramP/bigramP.sum(axis=1, keepdims=True)

def UnigramStartsWith(s):
    s = s.upper()
    result1Index = np.array([i for i in range(lenVocab) if vocab[i][2]==s])
    result = np.column_stack((vocab[result1Index], unigramP[result1Index]))
    return result

def BigramNext(prevWord, howMany):
    prevWord = prevWord.upper()
    prevIndex = WhatIndex(prevWord)
    if prevIndex != -1 :
        nextWord = bigramP[WhatIndex(prevWord)]
        result2Index = np.argsort(nextWord)[::-1]
        result = np.column_stack((vocab[result2Index][:howMany], nextWord[result2Index][:howMany]))
        return result
    else:
        return "Word not found!"

def UnigramSentence(s):
    s =s.upper();
    logLikelihood = 0.0
    for w in s.split():
        indexS = WhatIndex(w)
        if indexS != -1:
            logLikelihood += math.log(unigramP[indexS])
        else:
            logLikelihood -= math.inf
            print (w)
    return logLikelihood;

def BigramSentence(s):
    s =s.upper();
    logLikelihood = 0.0
    s = '<s> ' + s
    splitS = s.split();
    indices = np.array([WhatIndex(w) for w in splitS])
    for i in range(1,len(splitS)):
        if indices[i]!=-1 and indices[i-1]!= -1:
            if bigramP[indices[i-1]][indices[i]] > 0.0:
                logLikelihood += math.log(bigramP[indices[i-1]][indices[i]])
            else:
                logLikelihood -=math.inf
                print (vocab[indices[i-1]], vocab[indices[i]])
        else :
            logLikelihood -=math.inf
            print (vocab[indices[i-1]], vocab[indices[i]])
    return logLikelihood
              
def MixedSentence(s, lmda):
    s = s.upper()
    s = '<s> ' + s
    logLikelihood = 0.0
    splitS = s.split();
    indices = np.array([WhatIndex(w) for w in splitS])
    for i in range(1,len(splitS)):
        if indices[i]!=-1 and indices[i-1]!= -1:
            Pb = bigramP[indices[i-1]][indices[i]]
            Pu = unigramP[indices[i]]
            mixedP = (1-lmda)*Pu + lmda*Pb
            if mixedP > 0.0:
                logLikelihood += math.log(mixedP)
            else:
                logLikelihood -= math.inf
        else:
            logLikelihood -=math.inf
            print (vocab[indices[i-1]], vocab[indices[i]])
    return logLikelihood

def lmbdFunc(s, step):
    xRange = np.arange(0,1,step)
    y = np.array([MixedSentence(s, x) for x in xRange])
    print ('Optimum lambda:', xRange[np.argmax(y)])
    print ('Maximum Likelihood: ' , np.max(y))
    plt.plot(xRange, y)
    plt.xlabel('lambda')
    plt.ylabel('LogLikelihood in Mixed model')