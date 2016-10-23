# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 21:57:30 2016

@author: gopal
"""
import numpy as np

vocab = np.loadtxt('vocabulary.txt', dtype= str, unpack=True)
stopVocab = np.loadtxt('english.txt', dtype=str, unpack=True)

trainDocId, trainWordId, trainCount= np.loadtxt('train.data', dtype = int, unpack=True)
trainLabel = np.loadtxt('train.label', dtype=int, unpack=True)

testDocId, testWordId, testCount = np.loadtxt('test.data', dtype=int, unpack=True)
testLabel = np.loadtxt('test.label', dtype=int, unpack=True)

#print (trainDocId[1], wordId[1], trainCount[1])
#print (trainLabel[1])
def DensityPie(trainLabel, trainLabelLen, totalClasses):
    return np.array([np.count_nonzero(trainLabel == i+1)/(trainLabelLen) for i in range(totalClasses)])
#DensityPie = [np.count_nonzero(trainLabel == i+1)/(trainLabelLen) for i in range(totalClasses)]
def DocClassRepresent(trainDocId, trainWordId, trainCount, trainLabel, vocab, vocabLen, totalClasses, stopVocab=[]):
    indexFix = np.zeros(vocabLen, dtype=int)
    intersection = np.count_nonzero(np.in1d(stopVocab, vocab, assume_unique = True))
    DocClassRepresent = np.ones(shape=(totalClasses, vocabLen-intersection), dtype=float)
    for i in range(len(trainDocId)):
        if vocab[trainWordId[i]-1] in stopVocab or vocab[trainWordId[i]-1] == '-1 stopword':
            if vocab[trainWordId[i]-1] != '-1 stopword':
                fixZeros = np.zeros((trainWordId[i]), dtype=int)
                fixZeros = np.insert(fixZeros, trainWordId[i], np.ones(vocabLen-trainWordId[i])).flatten()
                indexFix = np.add(indexFix, fixZeros)
                vocab[trainWordId[i]-1] = '-1 stopword' 
        else:
            DocClassRepresent[trainLabel[trainDocId[i]-1]-1][trainWordId[i]-1-indexFix[trainWordId[i]-1]]+= trainCount[i]
    DocClassRepresent = np.log(DocClassRepresent/DocClassRepresent.sum(axis=1, keepdims=True))
    CanReject = np.flatnonzero((np.amax(DocClassRepresent, axis=0) - np.amin(DocClassRepresent, axis=0)) < 0.5)
    print(vocab[CanReject])
    return DocClassRepresent

def testDocRepresent(testDocId, testWordId, testCount, vocab, vocabLen, testLabelLength):
    testDocs = np.zeros(shape=(testLabelLength, vocabLen), dtype=int)
    for i in range(len(testDocId)):
        testDocs[testDocId[i]-1][testWordId[i]-1]+= testCount[i]
    return testDocs
    
def Classify(testDocs, DocClassRepresent, DensityPie):
    return np.argmax(np.add(np.dot(testDocs, DocClassRepresent.T), DensityPie), axis = 1)

def CalcError(classify, testLabel):
    return 
    
def Run():
    trainLabelLen = len(trainLabel)
    totalClasses = max(trainLabel)
    testLabelLength = len(testLabel)
    vocabLen = len(vocab)
    densityPie = DensityPie(trainLabel, trainLabelLen, totalClasses)
    trainDocClass = DocClassRepresent(trainDocId, trainWordId, trainCount, trainLabel, vocab, vocabLen, totalClasses, stopVocab)
    #testDocRep = testDocRepresent(testDocId, testWordId, testCount, vocab, vocabLen, testLabelLength)
    #result = Classify(testDocRep, trainDocClass, densityPie)
    #print(result);
    return trainDocClass;

x = Run()