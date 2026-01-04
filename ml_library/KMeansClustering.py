import numpy as np
import pandas as pd
def assignValues(X, centroids):
    c = np.zeros(X.shape[0], dtype = int)
    for i in range (X.shape[0]):
        #to think of some vectorized implementation
        Kzero = X[i] - centroids
        c[i] = np.argmin(np.sum(Kzero**2, axis = 1))
    return c
  def newMeanClusterCentroids(X, K, c):
    centroids = np.zeros((K, X.shape[1]))
    t = np.zeros((K, X.shape[1]))
    sizeForEach = np.zeros(K, dtype = int)
    for i in range (X.shape[0]):
        for j in range (K):
            if (c[i] == j):
                t[j] += X[i]
                sizeForEach[j]+=1
    for i in range(K):
        if (sizeForEach[i] > 0):
            centroids[i] = t[i]/sizeForEach[i]
        else:
           centroids[i] = X[np.random.choice(X.shape[0])]
    return centroids
def J(c, centroid, X):
    J = 0.0
    for i in range (X.shape[0]):
        J += (np.sum((centroid[c[i]] - X[i])**2))/X.shape[0]
    return J
def KMeans(X, K, num_iters = None):
    if (num_iters is None):
        num_iters = 50 + (X.shape[0] * K) % 950

    Centroids = X[np.random.choice(range(X.shape[0]), K, replace=False), :]
    finalIter = num_iters - 1
    JForCost = np.zeros(num_iters)
    Ccopy = []
    CentroidsCopy = []
    prevC = None
    for i in range(num_iters):
        c = assignValues(X, Centroids)
        Centroids = newMeanClusterCentroids(X, K, c)
        Ccopy.append(c)
        CentroidsCopy.append(Centroids)
        JForCost[i] = (J(c, Centroids, X))
        if (prevC is not None and np.array_equal(c, prevC)):
            finalIter = i
            break
        prevC = c
    t = np.argmin(JForCost[:finalIter+1])
    return Ccopy[t], CentroidsCopy[t]    
