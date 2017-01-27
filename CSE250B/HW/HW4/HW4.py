# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:59:12 2016

@author: gopal
"""

from nltk.cluster import KMeansClusterer, euclidean_distance
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from collections import defaultdict
import string
import numpy as np

FullCorpus = np.genfromtxt('brown_corpus.txt', dtype=str)
stopw = np.array(stopwords.words('english'))
stopVocab = np.genfromtxt('stopWords2', dtype=str)
nnList = np.genfromtxt('nnlist.txt', dtype=str)
def startsWithPunc(st):
    for i in range(len(Punctuation)):
        if st.startswith(Punctuation[i]):
            return True
    return False
    
def Preprocess(st):
    if st.startswith('$'):
        return '$dollars'
    elif str.isnumeric(st):
        if int(st) > 1900 and int(st)< 2050:
            return '$year'
        else:
            return '$number'
    return st
    
    
Punctuation = np.array([string.punctuation[i] for i in range(len(string.punctuation))])
Punctuation = np.append(Punctuation, ["``", "''", "--"])

#filtered = np.array([FullCorpus[i].lower() for i in range(len(FullCorpus)) if startsWithPunc(FullCorpus[i].lower()) == False and FullCorpus[i].lower() not in stopw])
filtered = np.array([FullCorpus[i].lower() for i in range(len(FullCorpus)) if FullCorpus[i] not in Punctuation and FullCorpus[i].lower() not in stopw and FullCorpus[i].lower() not in stopVocab])
words = np.array([Preprocess(filtered[i]) for i in range(len(filtered))]);
###-------------------Data Ready-------------------#

frequency = defaultdict(lambda: 0)
for word in words:
    frequency[word]+=1
frequency = sorted(frequency.items(), key= lambda x: x[1] , reverse=True)

V = defaultdict(lambda: 0, frequency[:5000])
C = defaultdict(lambda: 0, frequency[:1000])

V_List = np.array(list(V))
C_List = np.array(list(C))
V_Index = {x:i for i,x in enumerate(V)}
#PC_W = defaultdict(lambda: defaultdict(int))

PC_W = np.zeros(shape=(len(C), len(V)))
contextIndex =  np.where([words == p for p in C_List])

for i in range(len(contextIndex[1])):
    c = words[contextIndex[1][i]]
    w1 = words[contextIndex[1][i] - 2]
    w2 = words[contextIndex[1][i] - 1]
    w3 = words[contextIndex[1][i] + 1]
    w4 = words[contextIndex[1][i] + 2]
    if V[w1]> 0 : PC_W[contextIndex[0][i]][V_Index[w1]]+=1
    if V[w2]> 0 : PC_W[contextIndex[0][i]][V_Index[w2]]+=1
    if V[w3]> 0 : PC_W[contextIndex[0][i]][V_Index[w3]]+=1
    if V[w4]> 0 : PC_W[contextIndex[0][i]][V_Index[w4]]+=1

    
PC_W = PC_W/np.sum(PC_W, axis = 0)
cSum = sum(C.values())
PC = np.array([C[w]/cSum for w in C_List])

PC_W = (np.log(PC_W).T - np.log(PC)).T
for i in range(np.shape(PC_W)[0]):
    for j in range(np.shape(PC_W)[1]):
        if PC_W[i][j] == -np.inf:
            PC_W[i][j] = 0

numberOfClusters = 100            
#####-----------------------------Probbability measure ready-------------#####
#variancePC_W = np.var(PC_W, axis=1)
#clusterer = KMeansClusterer(numberOfClusters, euclidean_distance, repeats=3, avoid_empty_clusters=True);
#clusters= np.array(clusterer.cluster(PC_W.T, True, True))
#clust = {}
#for i in range(numberOfClusters):
#    clust[i] = list(V_List[np.array(np.where(clusters == i)).flatten()])
####----------------PCA------------------------####
reducedDim = 100
pca = PCA(n_components=reducedDim)
pca.fit(PC_W.T)
#print (pca.components_) np.dot(pca_components_, pca.components_) = Identity
PCA_PC_W = np.dot(pca.components_, PC_W)
clusterer = KMeansClusterer(numberOfClusters, euclidean_distance, repeats=3, avoid_empty_clusters=True);
newClusters = np.array(clusterer.cluster(PCA_PC_W.T, True, True))
newClust = {}
for i in range(numberOfClusters):
    newClust[i] = list(V_List[np.array(np.where(newClusters == i)).flatten()])
    
neigh = NearestNeighbors(algorithm='brute', n_neighbors=2, metric='cosine')
neigh.fit(PCA_PC_W.T)
cosineNN = [V_List[neigh.kneighbors([PCA_PC_W.T[V_Index[word]]], return_distance= False)] for word in nnList]