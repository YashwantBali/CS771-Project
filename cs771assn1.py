#############################################################
# File: submit.py <Assignment 1 - Intro to ML: CS771>
# Authors:	
#	220281 - Banoth Anand  <banotha22@iitk.ac.in>
#   220279 - Bali Yaswanth Naidu <baliyn22@iitk.ac.in>
#	230694 - Nidhi Bajpai  <nidhib23@iitk.ac.in>
#	230734 - Pankhuri Sachan  <pankhurisa23@iitk.ac.in>
##############################################################

import numpy as np
import sklearn
from scipy.linalg import khatri_rao
#from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def my_map(X):
    # Convert the elements to -1 and 1
    X = 1 - 2 * X

    n = X.shape[1]     #no.of columns in X
    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((X, ones_column))

    for row in X:
        for i in range(n - 2, -1, -1):
            row[i] = row[i + 1] * row[i]
    n_features = X.shape[1]
    triu_indices = np.triu_indices(n_features, k=1)
    new_vector = X[:, triu_indices[0]] * X[:, triu_indices[1]]

    return new_vector

def my_fit(X_train, y_train0, y_train1):

    X_train = my_map(X_train)
    #model = LogisticRegression(max_iter=10000,penalty="l2")
    model = LinearSVC(max_iter=10000, loss = 'squared_hinge', C = 10, tol=1e-4, penalty = "l2", dual = False)
    model.fit(X_train, y_train0, y_train1)
    

    # Extract the learned weights and bias
    w0 = model.coef_[0]
    b0 = model.intercept_[0] if model.fit_intercept else 0

    w1 = model.coef_[0]
    b1 = model.intercept_[0] if model.fit_intercept else 0

    return w0, b0, w1, b1

