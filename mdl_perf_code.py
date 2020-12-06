# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:13:19 2020

@author: SRDhruba
"""

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR as SupportVectorRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr

## Predictive models...
def RF(X_train, y_train, X_test, seed = 0):
    mdl = RandomForestRegressor(n_estimators = 200, criterion = "mae", min_samples_leaf = 5, random_state = seed, n_jobs = -1)
    y_pred = mdl.fit(X_train, y_train).predict(X_test)
    return y_pred

def SVR(X_train, y_train, X_test):
    mdl = SupportVectorRegressor(kernel = "poly", degree = 3, coef0 = 1.0, gamma = "scale", tol = 1e-3, C = 10)
    y_pred = mdl.fit(X_train, y_train).predict(X_test)
    return y_pred

def EN(X_train, y_train, X_test, seed):
    mdl = ElasticNet(fit_intercept = True, l1_ratio = 0.005, alpha = 0.2, tol = 1e-3, selection = "random", random_state = seed)
    y_pred = mdl.fit(X_train, y_train).predict(X_test)
    return y_pred

def KNN(X_train, y_train, X_test):
    mdl = KNeighborsRegressor(n_neighbors = 5, weights = "distance", algorithm = "auto", metric = "minkowski", p = 1, n_jobs = -1)
    y_pred = mdl.fit(X_train, y_train).predict(X_test)
    return y_pred


## Evaluate model performance...
def EVAL_PERF(y_label, y_pred, alpha = 0.05):
    y_label, y_pred = np.array(y_label).squeeze(), np.array(y_pred).squeeze()
    PCC, pval = pearsonr(y_label, y_pred);     #PCC = PCC if pval < alpha else 0
    SCC, pval = spearmanr(y_label, y_pred);    #SCC = SCC if pval < alpha else 0
    NRMSE = np.sqrt(((y_label - y_pred)**2).mean()) / y_label.std(ddof = 0)
    NMAE  = (np.abs(y_label - y_pred)).mean() / (np.abs(y_label - y_label.mean())).mean()
    return PCC, SCC, NRMSE, NMAE
