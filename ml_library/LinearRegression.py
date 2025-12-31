import numpy as np
import pandas as pd
def J(w, b, X, y, lambda_ = 10):
    return ((np.sum((np.dot(X, w) + b - y)**2))/(2*X.shape[0] ) + (lambda_ * np.sum(w**2)/(2*X.shape[0])))
def compute_gradient(w, X, y, b, lambda_ = 10):#Note that y must be of dim(m, 1)
    m = np.shape(X)[0]
    dj_db= np.sum(np.dot(X, w) + b - y)/m
    dj_dw = (np.dot(X.T, np.dot(X, w) + b - y))/m  + (lambda_*w)/m
    return dj_dw, dj_db
def gradientDescent(num_iters, alpha, X, w, y, b, lambda_ = 10):#Note that y must be of dim(m, 1)
    previousCost = J(w, b, X, y)
    for i in range (num_iters):
        dj_dw, dj_db = compute_gradient(w, X, y, b, lambda_)
        while True:
            wtemp =w - alpha * dj_dw
            btemp = b - alpha*dj_db
            newCost = J(wtemp, btemp, X, y)
            if (newCost > previousCost):
                alpha/=3
            else:
                previousCost = newCost
                w = wtemp
                b = btemp
                break
    return w, b, alpha
def predict(X_test, wfortest, bfortest, mu, sigma):#X_test sent here must be normalized
    return ((np.dot(X_test, wfortest) + bfortest) * sigma)+mu
def z_scoreNormalization(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X-mu)/sigma
    return (X_norm, mu, sigma)
