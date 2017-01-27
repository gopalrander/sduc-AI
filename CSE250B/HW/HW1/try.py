import os;
#import math;
import struct;
import gzip;
from array import array;
#from multiprocessing import Pool;
import numpy as np;
#from scipy import linalg;
from sklearn.neighbors import NearestNeighbors

from numbapro import vectorize, cuda
import numba as nb

os.chdir('D:\\UCSD\\F16\\sduc-AI\\CSE250B\\HW\\HW1')

#distance b/w two vectors

labelTrainingSet = [];
imageTrainingSet = [];
labelTestSet = [];
imageTestSet = [];

#def diff_1NN_p(A, B, p):
#	if (len(A) == len(B)):
#		vectorLength = len(A)
#		dist = 0
#		for i in range(vectorLength):
#			dist += pow(abs(A[i] - B[i]), p)
#		dist = pow(dist, 1/p)
#		return dist
#	else:
#         print ("ERROR in diff_1_NN")
#         print (len(A), len(B))
#         return math.nan;
#
         

#print (diff_1NN_p(A, B, 1))
def loadData():
        with gzip.open("train-labels-idx1-ubyte.gz", 'rb') as file:
            magicNum, size = struct.unpack(">II", file.read(8))
            #print (magicNum, size)
            global labelTrainingSet
            labelTrainingSet = array("B", file.read())
            labelTrainingSet = np.array(labelTrainingSet,dtype=np.int)
            #print (labelTrainingSet[0:10])

        with gzip.open("t10k-labels-idx1-ubyte.gz", 'rb') as file:
            magicNum, size = struct.unpack(">II", file.read(8))
            #print (magicNum, size)
            global labelTestSet
            labelTestSet = array("B", file.read())
            labelTestSet = np.array(labelTestSet,dtype=np.int)
            #print (labelTestSet[0:10])

        with gzip.open("train-images-idx3-ubyte.gz",'rb') as file:
            magicNum, size, rows, cols = struct.unpack(">IIII", file.read(16))
            #print (magicNum, size, rows, cols)

            imagesTemp = array("B", file.read())
            imagesTemp = np.array(imagesTemp)

            global imageTrainingSet
            #for i in range (size):
                #imageTrainingSet.append([0]* rows*cols)
            imageTrainingSet = np.zeros(shape=(size, rows*cols), dtype=np.int) 
                
            for i in range (size):
                imageTrainingSet[i] = imagesTemp[rows*cols*i : rows*cols*(i+1)]
            #print(imageTrainingSet[0])
            
        with gzip.open("t10k-images-idx3-ubyte.gz",'rb') as file:
            magicNum, size, rows, cols = struct.unpack(">IIII", file.read(16))
            #print (magicNum, size, rows, cols)

            imagesTemp = array("B", file.read())
            imagesTemp = np.array(imagesTemp)

            global imageTestSet
            #for i in range (size):
            #    imageTestSet.append([0]* rows*cols)
            imageTestSet = np.zeros(shape=(size, rows*cols), dtype=np.int)
            
            for i in range (size):
                imageTestSet[i] = imagesTemp[rows*cols*i : rows*cols*(i+1)]

            #print(imageTestSet[0])


#def find_NN(testIndex):
#    minDist = math.inf
#    labelIndex = -1
#    for i in range (1000):
        #curDist = diff_1NN_p(imageTestSet[testIndex], imageTrainingSet[i], 2)
#        curDist = linalg.norm(imageTestSet[testIndex] - imageTrainingSet[i])
        #print(curDist);
#        if (curDist < minDist):
#            minDist = curDist
#            labelIndex = i        
    #print ("TestLabel ", "NN-Label")
    #print (testIndex, labelIndex, minDist);
    #print (labelTestSet[testIndex], labelTrainingSet[labelIndex]) 
    #return (labelTrainingSet[labelIndex] == labelTestSet[testIndex])

#class ThreadClass(threading.Thread):
#    def __init__(self, index):
#        threading.Thread.__init__(self)
#        self.index = index
#        #self.q = q;
#    def run(self):
#        nnIndex = find_NN(self.index)
#        print (nnIndex)
        #self.q.put ([self.index, nnIndex])

#def find_testError():
#    threads = []
#    wrong = 0;
#    for i in range (len(imageTestSet)):
#        t = ThreadClass(i)
#        threads.append(t);
#        t.start()
#        
#    for t in threads:
#        t.join();
#        #if (labelTestSet[i] != labelTrainingSet[labelIndex]):
#         #   wrong = wrong+1;
#          #  print (wrong);
            
#find_testError();
#if __name__ == '__main__':
#        with Pool(6) as p:
#            result = np.array(p.map(find_NN, np.arange(10000)))
#        print (np.count_nonzero(result))
loadData();
@vectorize(['float32(float32, float32)'], target="gpu")
def addMe(x ,y):
    return x+y

    
#@vectorize(target='gpu' )
def getErrorPercent(trainingSet, trainingLabel, testSet, testLabel):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(trainingSet);
    result = nbrs.kneighbors(testSet, return_distance=False).flatten()
    guess = np.array(trainingLabel[result])
    errorLabels = np.where(testLabel-guess !=0)[0]
    np.random.shuffle(errorLabels);
    return (np.count_nonzero(testLabel-guess)*100/len(testLabel)) , errorLabels


#@vectorize(['void(int[:,:], int[:])'])
def randomizeInput(trainingSet, trainingLabel):
    tempArray = np.column_stack((trainingSet, trainingLabel))
    np.random.shuffle(tempArray)
    return tempArray[:,range(784)], tempArray[:,784]

#0 1NN (3.09)

#### Prototype Selection #####
#1 Average over all labels. (17.97)
rangeArray = np.arange(len(labelTrainingSet));
#average1 = np.array([imageTrainingSet[i].tolist() for i in rangeArray if labelTrainingSet[i]==1])
PLabelSet = np.arange(10, dtype=int)
#PTrainingSet = np.zeros(shape=(10, 784), dtype= np.float64) 
#for j in np.nditer(PLabelSet):
#    PTrainingSet[j] = np.mean(np.array([imageTrainingSet[i].tolist() for i in rangeArray if labelTrainingSet[i]==j]), axis=0, dtype=np.float64)
#print (PTrainingSet)

#--2 select first n from each class and calculate. (first 1000 =12.69)
#2 random first n from each class 1000 = 10.75
#randomSubsetSize = 10000
#randomSubsetPerClassSize = int(randomSubsetSize/10)
#imageTrainingSet, labelTrainingSet = randomizeInput(imageTrainingSet, labelTrainingSet)
#P2TrainingSet = np.array([imageTrainingSet[i].tolist() for i in rangeArray if labelTrainingSet[i]==0])[:randomSubsetPerClassSize , :]
#P2LabelSet = np.array([0 for i in range(randomSubsetPerClassSize)])

#for j in range(1, len(PLabelSet)):
#    P2TrainingSet = np.append(P2TrainingSet, np.array([imageTrainingSet[i].tolist() for i in rangeArray if labelTrainingSet[i]==j])[:randomSubsetPerClassSize , :], axis=0)
#    P2LabelSet = np.append(P2LabelSet, [j for i in range(randomSubsetPerClassSize)])
#print ("#3")
confidenceRuns = 1
while (confidenceRuns > 0):
    #3 just random n from given training set
    randomSubsetSize = 1000
    imageTrainingSet, labelTrainingSet = randomizeInput(imageTrainingSet, labelTrainingSet)
    P3TrainingSet = imageTrainingSet[:randomSubsetSize, :]
    P3LabelSet = labelTrainingSet[:randomSubsetSize]
    print (getErrorPercent(P3TrainingSet, P3LabelSet, imageTestSet, labelTestSet)[0], end="", flush=True);
    
    #print ("#4")
    #4 do random 100 selctions from initial set. Then one by one add the rows from the training set to set S which are misclassified. incrementally increase the set.
    randomSubsetSize = 100
    #imageTrainingSet, labelTrainingSet = randomizeInput(imageTrainingSet, labelTrainingSet)
    P4TrainingSet = imageTrainingSet[:randomSubsetSize, :]
    P4LabelSet = labelTrainingSet[:randomSubsetSize]
    errorNow = 100
    runCount = 90
    addEveryLoop = 10
    epsilon = 0.1
    while runCount > 0 and errorNow > epsilon:
        print (len(P4LabelSet), errorNow)
        errorNow, errorIndex =  getErrorPercent(P4TrainingSet, P4LabelSet, imageTrainingSet, labelTrainingSet)
        P4TrainingSet = np.append(P4TrainingSet, np.array(imageTrainingSet[errorIndex[0:addEveryLoop]]), axis=0)
        P4LabelSet = np.append(P4LabelSet, labelTrainingSet[errorIndex[0:addEveryLoop]])
        runCount = runCount - 1;
        
    #print (getErrorPercent(imageTrainingSet, labelTrainingSet, imageTestSet, labelTestSet));
    print("\t", end="")
    print (getErrorPercent(P4TrainingSet, P4LabelSet, imageTestSet, labelTestSet)[0], flush=True)
    confidenceRuns = confidenceRuns - 1 