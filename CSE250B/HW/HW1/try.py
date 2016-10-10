import os;
import math;
import struct;
import gzip;
from array import array;
import threading
import queue;
from multiprocessing import Pool;
import numpy as np;
from scipy import linalg;

    
os.chdir('D:\\UCSD\\F16\\sduc-AI\\CSE250B\\HW\\HW1')

#distance b/w two vectors
q = queue.Queue()
labelTrainingSet = [];
imageTrainingSet = [];
labelTestSet = [];
imageTestSet = [];

def diff_1NN_p(A, B, p):
	if (len(A) == len(B)):
		vectorLength = len(A)
		dist = 0
		for i in range(vectorLength):
			dist += pow(abs(A[i] - B[i]), p)

		dist = pow(dist, 1/p)
		return dist
	else:
         print ("ERROR in diff_1_NN")
         print (len(A), len(B))
         return math.nan;

         

#print (diff_1NN_p(A, B, 1))
def loadData():
        with gzip.open("train-labels-idx1-ubyte.gz", 'rb') as file:
            magicNum, size = struct.unpack(">II", file.read(8))
            #print (magicNum, size)
            global labelTrainingSet
            labelTrainingSet = array("B", file.read())
            labelTrainingSet = np.array(labelTrainingSet)
            #print (labelTrainingSet[0:10])

        with gzip.open("t10k-labels-idx1-ubyte.gz", 'rb') as file:
            magicNum, size = struct.unpack(">II", file.read(8))
            #print (magicNum, size)
            global labelTestSet
            labelTestSet = array("B", file.read())
            labelTestSet = np.array(labelTestSet)
            #print (labelTestSet[0:10])

        with gzip.open("train-images-idx3-ubyte.gz",'rb') as file:
            magicNum, size, rows, cols = struct.unpack(">IIII", file.read(16))
            print (magicNum, size, rows, cols)

            imagesTemp = array("B", file.read())
            imagesTemp = np.array(imagesTemp)

            global imageTrainingSet
            #for i in range (size):
                #imageTrainingSet.append([0]* rows*cols)
            imageTrainingSet = np.zeros(shape=(size, rows*cols), dtype= np.int) 
                
            for i in range (size):
                imageTrainingSet[i] = imagesTemp[rows*cols*i : rows*cols*(i+1)]
            #print(imageTrainingSet[0])
            
        with gzip.open("t10k-images-idx3-ubyte.gz",'rb') as file:
            magicNum, size, rows, cols = struct.unpack(">IIII", file.read(16))
            print (magicNum, size, rows, cols)

            imagesTemp = array("B", file.read())
            imagesTemp = np.array(imagesTemp)

            global imageTestSet
            #for i in range (size):
            #    imageTestSet.append([0]* rows*cols)
            imageTestSet = np.zeros(shape=(size, rows*cols), dtype= np.int)
            
            for i in range (size):
                imageTestSet[i] = imagesTemp[rows*cols*i : rows*cols*(i+1)]

            #print(imageTestSet[0])


def find_NN(testIndex):
    minDist = math.inf
    labelIndex = -1
    for i in range (len(imageTrainingSet)):
        #curDist = diff_1NN_p(imageTestSet[testIndex], imageTrainingSet[i], 2)
        curDist = linalg.norm(imageTestSet[testIndex] - imageTrainingSet[i])
        print(curDist);
        if (curDist < minDist):
            minDist = curDist
            labelIndex = i        
    #print ("TestLabel ", "NN-Label")
    print (testIndex, labelIndex, minDist);
    #print (labelTestSet[testIndex], labelTrainingSet[labelIndex]) 
    return (labelTrainingSet[labelIndex] == labelTestSet[testIndex])

class ThreadClass(threading.Thread):
    def __init__(self, index):
        threading.Thread.__init__(self)
        self.index = index
        #self.q = q;
    def run(self):
        nnIndex = find_NN(self.index)
        print (nnIndex)
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
            
   
loadData();
result = [];
#find_testError();
if __name__ == '__main__':
        with Pool(6) as p:
            result = np.array(p.map(find_NN, np.arange(1000)))
        print (np.count_nonzero(result))