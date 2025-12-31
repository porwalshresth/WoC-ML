import numpy as np
import pandas as pd
#Note that the implementation of Gradient, cost. here is different than Linear one, don't put any of that code when compiling all finally!!!
def sigmoid (z):
    return (1/(1 +np.exp(-z)))
def J_Logistic(X, y, w, b, lambda_ = 10):
    m = X.shape[0]
    predictions = sigmoid (np.dot(X, w) + b)
    cost = np.sum((-y * np.log(predictions) - (1-y)*np.log(1 - predictions)))/m + lambda_ * np.sum(w**2)/(2*m)
    return cost
def computeGradientLogistic(X, y, w, b, lambda_ = 10):
    m, n = X.shape
    predictions = sigmoid (np.dot(X, w) + b)
    dj_dw = (np.dot(X.T, predictions - y)) / m + lambda_*w/m
    dj_db = np.sum(predictions - y) / m
    return dj_dw, dj_db
#The following code can be removed and Gradient Descent, normalization, prediction for the three can be like made kinda some other function visible to all functuons and then following code could be removed.
def gradientDescent(num_iters, alpha, X, w, y, b, lambda_ = 10):#Error Warning: y must be of dim(m, 1)
    if y.ndim == 1:#ensure that you apply this if condition everywhere.
        y = y.reshape(-1, 1)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    previousCost = J_Logistic(X, y, w, b, lambda_)
    for i in range (num_iters):
        dj_dw, dj_db = computeGradientLogistic(X, y, w, b)
        while True:
            wtemp =w - alpha * dj_dw
            btemp = b - alpha*dj_db
            newCost = J_Logistic(X, y, wtemp, btemp)
            if (newCost > previousCost):
                alpha/=3
            else:
                previousCost = newCost
                w = wtemp
                b = btemp
                break
    return w, b, alpha
def predict(X_test, wfortest, bfortest):#X_test sent here must be normalized
    return (((sigmoid(np.dot(X_test, wfortest) + bfortest)) >= 0.5).astype(int))
def z_scoreNormalization(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X-mu)/sigma
    return (X_norm, mu, sigma)
