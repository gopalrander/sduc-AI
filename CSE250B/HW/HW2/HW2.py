# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 21:57:30 2016

@author: gopal
"""
import numpy as np

vocab = np.loadtxt('vocabulary.txt', dtype= str, unpack=True)
#stopVocab = np.loadtxt('english.txt', dtype=str, unpack=True)
stopVocab = np.loadtxt('stopWords2', dtype=str, unpack=True)
#stopVocab = np.loadtxt('mystopwords.txt', dtype=str, unpack=True)
trainDocId, trainWordId, trainCount= np.loadtxt('train.data', dtype = int, unpack=True)
trainLabel = np.loadtxt('train.label', dtype=int, unpack=True)

testDocId, testWordId, testCount = np.loadtxt('test.data', dtype=int, unpack=True)
testLabel = np.loadtxt('test.label', dtype=int, unpack=True)

##print (trainDocId[1], wordId[1], trainCount[1])
##print (trainLabel[1])
def DensityPie(trainLabel, trainLabelLen, totalClasses):
    return np.array([np.count_nonzero(trainLabel == i+1)/(trainLabelLen) for i in range(totalClasses)])
##DensityPie = [np.count_nonzero(trainLabel == i+1)/(trainLabelLen) for i in range(totalClasses)]
def DocClassRepresent(trainDocId, trainWordId, trainCount, trainLabel, vocab, vocabLen, totalClasses, stopVocab, indexFixBits):
    for w in stopVocab:
        index = np.where(vocab==w)[0]
        if len(index) != 0 :
            indexFixBits[index] = 1;
            vocab[index] = '-1 stopword'
    indexFix = np.cumsum(indexFixBits)

    modifiedVocab = vocab[np.where(vocab!='-1 stopword')]
    DocClassRepresent = np.ones(shape=(totalClasses, len(modifiedVocab)), dtype=float)
    
    for i in range(len(trainDocId)):
        if vocab[trainWordId[i]-1] != '-1 stopword':
            DocClassRepresent[trainLabel[trainDocId[i]-1]-1][trainWordId[i]-1-indexFix[trainWordId[i]-1]]+= trainCount[i]

    DocClassRepresent = np.log(DocClassRepresent/DocClassRepresent.sum(axis=1, keepdims=True))
    #CanReject = np.flatnonzero((np.amax(DocClassRepresent, axis=0) - np.amin(DocClassRepresent, axis=0)) < 1.5)
    #print(modifiedVocab[CanReject])
    VarianceSort = np.argsort(np.var(DocClassRepresent, axis=0))
    return DocClassRepresent, modifiedVocab, indexFixBits, VarianceSort

def testDocRepresent(testDocId, testWordId, testCount, vocab, vocabLen, indexFix, testLabelLength):
    testDocs = np.zeros(shape=(testLabelLength, vocabLen), dtype=int)
    indexFix = np.cumsum(indexFixBits)
    for i in range(len(testDocId)):
        if(vocab[testWordId[i]-1] != '-1 stopword'):
            testDocs[testDocId[i]-1][testWordId[i]-1-indexFix[testWordId[i]-1]]+= testCount[i]
    invalidDocsId = np.where(np.sum(testDocs, axis=1) == 0)[0]
    return testDocs, invalidDocsId
    
def Classify(testDocs, invalidDocsId, DocClassRepresent, DensityPie):
    validDocsResult = np.argmax(np.add(np.dot(testDocs, DocClassRepresent.T), DensityPie), axis = 1)
    mostLikelyDoc = np.argmax(DensityPie)
    for i in invalidDocsId: 
        validDocsResult[i] = mostLikelyDoc
    return validDocsResult
    
#trainDocClass = np.load('trainDocClass.npy')
#modifiedVocab = np.load('modifiedVocab.npy')
trainLabelLen = len(trainLabel)
totalClasses = max(trainLabel)
testLabelLength = len(testLabel)
vocabLen = len(vocab)
indexFixBits = np.zeros(vocabLen, dtype=int)
densityPie = DensityPie(trainLabel, trainLabelLen, totalClasses)
M = 5000
#Step1
trainDocClass, modifiedVocab, indexFixBits, VarianceSort = DocClassRepresent(trainDocId, trainWordId, trainCount, trainLabel, vocab, vocabLen, totalClasses, stopVocab, indexFixBits)

#testDocRep = testDocRepresent(testDocId, testWordId, testCount, vocab, len(modifiedVocab), indexFixBits, testLabelLength)
#result = Classify(testDocRep, trainDocClass, densityPie)
#print ( np.count_nonzero((result+1)-testLabel));
#trainColnSum = np.sum(trainDocClass, axis=0)
#classLikelyWords = np.array(modifiedVocab[np.argsort(trainDocClass/trainColnSum, axis=1)])

#Step2
stopWordsStep2 = stopVocab

#stopWordsStep2 = np.append(stopWordsStep2, modifiedVocab[VarianceSort[:-M]])
#trainDocClass, modifiedVocab, indexFixBits, VarianceSort = DocClassRepresent(trainDocId, trainWordId, trainCount, trainLabel, vocab, vocabLen, totalClasses, stopWordsStep2, indexFixBits)

#testDocRep, invalidDocsId = testDocRepresent(testDocId, testWordId, testCount, vocab, len(modifiedVocab), indexFixBits, testLabelLength)
#result = Classify(testDocRep, invalidDocsId, trainDocClass, densityPie)
#print (M, np.count_nonzero((result+1)-testLabel));
#trainColnSum = np.sum(trainDocClass, axis=0)
#classLikelyWords = np.array(modifiedVocab[np.argsort(trainDocClass/trainColnSum, axis=1)])
#

#step3
trainColnSum = np.sum(trainDocClass, axis=0)
trainColVar = np.var(trainDocClass, axis=0)
maxDiff = np.zeros(shape=(np.shape(trainDocClass)))
for i in range(totalClasses):
    maxDiff[i] = ((totalClasses*trainDocClass[i] - trainColnSum) / trainColnSum) + trainColVar
res = np.argsort(-maxDiff, axis=1)
flag = np.zeros(len(modifiedVocab), dtype=bool)

m= M/(totalClasses)

for i in range(totalClasses):
    selected=0;
    for j in range(len(modifiedVocab)) :
        if selected >= m : break
        if flag[res[i][j]] == False :
            flag[res[i][j]] = True
            selected+=1
            
wordsToRemove = np.array(modifiedVocab[np.where(flag==False)])
stopWordsStep3 = np.append(stopWordsStep2 , wordsToRemove)
trainDocClass, modifiedVocab, indexFixBits, VarianceSort = DocClassRepresent(trainDocId, trainWordId, trainCount, trainLabel, vocab, vocabLen, totalClasses, stopWordsStep3, indexFixBits)

testDocRep, invalidDocsId = testDocRepresent(testDocId, testWordId, testCount, vocab, len(modifiedVocab), indexFixBits, testLabelLength)
result = Classify(testDocRep, invalidDocsId, trainDocClass, densityPie)
print (M, np.count_nonzero((result+1)-testLabel));
trainColnSum = np.sum(trainDocClass, axis=0)
classLikelyWords = np.array(modifiedVocab[np.argsort(trainDocClass/trainColnSum, axis=1)])

