import numpy as np
import pandas as pd
def generate_exponents(n, degree, interactionTerm = True):
    exponents = []
    def recurse(currentExponents, remainingDegree):
        if len(currentExponents) == n:
            exponents.append(tuple(currentExponents))
            return     
        for i in range(remainingDegree + 1):
            recurse(currentExponents + [i], remainingDegree - i)        
    #There is a all zero term in the exponents
    recurse([], degree) 
    uniqueExponents = sorted(list(set(exponents)))
    if not interactionTerm:
        # Keep only terms where at most 1 feature has a non-zero power.
        uniqueExponents = [exp for exp in uniqueExponents if np.count_nonzero(exp) <= 1]
    uniqueExponents = [exp for exp in uniqueExponents if sum(exp) > 0]
    return uniqueExponents
def create_polynomial_features(X, degree, interactionTerm = True):
    m, n = X.shape
    exponents = generate_exponents(n, degree, interactionTerm)
    new_n = len(exponents)
    new_X = np.zeros((m, new_n))
    for i, exp in enumerate(exponents):
        # The value of the term x1^e1 * x2^e2 * ... * xn^en is the product of (xj raised to ej)
        col = np.ones(m)
        for j in range(n):
            col *= (X[:, j] ** exp[j])
        new_X[:, i] = col     
    colNames = [f'x^{exp}' for exp in exponents]
    dfNew = pd.DataFrame(new_X, columns=colNames)
    return dfNew
#Entire Copy of the Linear Regression Code - try to make the library such that it the following code is not there and it is being called or maybe some directory thing.
def J(w, b, X, y, lambda_ = 10):
    return ((np.sum((np.dot(X, w) + b - y)**2))/(2*X.shape[0] ) + (lambda_ * np.sum(w**2)/(2*X.shape[0])))
def compute_gradient(w, X, y, b, lambda_ = 10):#Note that y must be of dim(m, 1)
    m = np.shape(X)[0]
    dj_db= np.sum(np.dot(X, w) + b - y)/m
    dj_dw = (np.dot(X.T, np.dot(X, w) + b - y))/m  + (lambda_*w)/m
    return dj_dw, dj_db
def gradientDescent(num_iters, alpha, X, w, y, b, lambda_ = 10):#Note that  y must be of dim(m, 1)
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
