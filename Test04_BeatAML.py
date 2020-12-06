# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:28:20 2020

@author: SRDhruba
"""


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import pickle
from time import time
from tqdm import tqdm
# from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR as SupportVectorRegressor
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr, spearmanr

## Path & FILE...
PATH = "%s\\Google Drive\\Study\\ECE 5332-009 - Topics in EE, Data Science\\BeatAML\\" % os.getenv("HOMEPATH")
DIR  = ("Training\\", "Leaderboard\\")
FILE = ("rnaseq.csv", "dnaseq.csv", "clinical_numerical.csv", "clinical_categorical.csv", 
        "clinical_categorical_legend.csv", "aucs.csv", "response.csv")
os.chdir(PATH)

### Training data...
RNA_TR = pd.read_csv(PATH + DIR[0] + FILE[0], header = 0)
# CLN_TR = pd.read_csv(PATH + DIR[0] + FILE[2], header = 0)
# CLC_TR = pd.read_csv(PATH + DIR[0] + FILE[3], header = 0)
AUC_TR = pd.read_csv(PATH + DIR[0] + FILE[5], header = 0)

## Leaderboard data...
RNA_LB = pd.read_csv(PATH + DIR[1] + FILE[0], header = 0)
# CLN_LB = pd.read_csv(PATH + DIR[1] + FILE[2], header = 0)
# CLC_LB = pd.read_csv(PATH + DIR[1] + FILE[3], header = 0)
AUC_LB = pd.read_csv(PATH + DIR[1] + FILE[5], header = 0)

all(AUC_TR.inhibitor.unique()  == AUC_LB.inhibitor.unique())     ## Check if same drugs
all(RNA_TR[["Gene", "Symbol"]] == RNA_LB[["Gene", "Symbol"]])    ## Check if same genes

lab_id_list = dict(TR = RNA_TR.columns, LB = RNA_LB.columns)
gene_list, drug_list = RNA_TR[["Gene", "Symbol"]], AUC_TR.inhibitor.unique().tolist()

## Preprocessing...
RNA_TR.index, RNA_LB.index = gene_list.Symbol, gene_list.Symbol
RNA_TR, RNA_LB = RNA_TR.iloc[:, 2:], RNA_LB.iloc[:, 2:]
# print(RNA_TR.shape, RNA_LB.shape)

var_idx = (-RNA_TR.var(axis = 1)).to_numpy().argsort()[:40000]   ## Filter by gene variability
RNA_TR_filt, RNA_LB_filt = RNA_TR.iloc[var_idx, :], RNA_LB.iloc[var_idx, :]


# In[ ]:


## Scaling...
zscore = lambda data, ref_data: StandardScaler().fit(ref_data).transform(data)

## Feature selection...
def LassoFS(X, y, seed = 0):
    # FS = LassoCV(fit_intercept = True, normalize = False, n_alphas = 100, tol = 1e-3, cv = 5, selection = "random", 
    #              random_state = seed, n_jobs = -1)
    # FS = Lasso(fit_intercept = True, normalize = False, alpha = 0.001, tol = 1e-3, selection = "random", random_state = seed)
    FS  = Lasso(fit_intercept = True, normalize = False, alpha = 0.001, selection = 'random', random_state = seed, max_iter = 3000)
    features = (FS.fit(X, y).coef_ != 0).nonzero()[0]
    return features


## Predictive models...
def RF(X_train, y_train, X_test, seed = 0):
    mdl = RandomForestRegressor(n_estimators = 200, criterion = "mae", min_samples_leaf = 5, random_state = seed, n_jobs = -1)
    y_pred = mdl.fit(X_train, y_train).predict(X_test)
    return y_pred

def SVR(X_train, y_train, X_test):
    mdl = SupportVectorRegressor(kernel = "poly", degree = 3, coef0 = 1.0, gamma = "scale", tol = 1e-3, C = 10, max_iter = 2000)
    y_pred = mdl.fit(X_train, y_train).predict(X_test)
    return y_pred

def EN(X_train, y_train, X_test, seed):
    mdl = ElasticNet(fit_intercept = True, l1_ratio = 0.005, alpha = 0.2, tol = 1e-3, max_iter = 2000, selection = "random", 
                     random_state = seed)
    y_pred = mdl.fit(X_train, y_train).predict(X_test)
    return y_pred

def KNN(X_train, y_train, X_test):
    mdl = KNeighborsRegressor(n_neighbors = 7, weights = "distance", algorithm = "auto", metric = "minkowski", p = 1, n_jobs = -1)
    y_pred = mdl.fit(X_train, y_train).predict(X_test)
    return y_pred

def ADB(X_train, y_train, X_test, seed = 0, base = "tree"):
    if base == "tree":
        base = DecisionTreeRegressor(criterion = "mae", max_depth = 3, min_samples_leaf = 5, max_features = "sqrt", 
                                     min_impurity_decrease = 1e-6, splitter = "best", random_state = seed)
    elif base == "linear":
        base = ElasticNet(fit_intercept = True, l1_ratio = 0.005, alpha = 0.2, tol = 1e-3, max_iter = 2000, selection = "random", 
                          random_state = seed)
    #### Choice loop ends...
    mdl = AdaBoostRegressor(base_estimator = base, n_estimators = 150, learning_rate = 0.8, loss = "exponential", random_state = seed)
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


# In[ ]

### K-FOLD ON TRAINING+LEADERBOARD DATA...
RNA_data = pd.concat((RNA_TR_filt, RNA_LB_filt), axis = 1)
# RNA_mean, RNA_SD = RNA_data.mean(axis = 1), RNA_data.std(axis = 1)
# RNA_data_norm = ((RNA_data.T - RNA_mean) / RNA_SD).T
AUC_data = pd.concat((AUC_TR, AUC_LB), axis = 0, ignore_index = True)

FS_switch = "ReliefF"
if FS_switch == "ReliefF":
    p_top = 20000
    # feature_ranks = {kk: [ ] for kk in drug_list}
    with open("FS_Drugs_122_ReliefF_3_folds.pickle", "rb") as file:
        feature_ranks = pickle.load(file)
elif FS_switch == "Lasso":
    feature_ranks = dict.fromkeys(drug_list)
####

AUC_data_pred = { };      metrics = ["PCC", "SCC", "NRMSE", "NMAE"]
models = ["RF", "SVR", "EN", "KNN", "ADB", "ENS", "ENS1234", "ENS1235", "ENS1245", "ENS1345", "ENS2345", "ENS123", "ENS124", "ENS125", 
          "ENS134", "ENS135", "ENS145", "ENS234", "ENS235", "ENS245", "ENS345"]
RESULTS_all = {MM: pd.DataFrame(dtype = float, index = drug_list, columns = metrics) for MM in models}

N  = len(drug_list);     count = 0
CV = KFold(n_splits = 3, shuffle = False, random_state = None)
dt = time() 
for drug in tqdm(drug_list[:N]):
    count += 1;   # print("\nChosen drug# = %d: %s" % (count, drug))
    
    y_data = AUC_data.iloc[(AUC_data.inhibitor == drug).tolist(), :]
    X_data = RNA_data.loc[:, y_data.lab_id].T
    # X_data = RNA_data_norm.loc[:, y_data.lab_id].T
    
    fold = 0
    ## From saved file...
    if FS_switch == "ReliefF":
        feat_top_set = feature_ranks[drug][fold]
        feat_top = feat_top_set[:p_top]
    elif FS_switch == "Lasso":
        feat_top = LassoFS(X_data, y_data, seed = 2020)
        feature_ranks[drug] = feat_top
    X_data, y_data = X_data.iloc[:, feat_top], y_data["auc"].to_numpy()

    ## CV loop starts...    
    Y_data_pred = pd.DataFrame(dtype = float, index = X_data.index, columns = ["Actual"] + models)
    for train_idx, test_idx in CV.split(X_data):
        X_train, y_train = X_data.iloc[train_idx, :], y_data[train_idx]
        X_test,  y_test  = X_data.iloc[test_idx, :],  y_data[test_idx]
        
        ## Perform prediction...
        test_idx_lab = X_data.index[test_idx]
        Y_data_pred.loc[test_idx_lab, "Actual"] = y_test
        Y_data_pred.loc[test_idx_lab, "RF"]     = RF(X_train, y_train, X_test, seed = 0)
        Y_data_pred.loc[test_idx_lab, "SVR"]    = SVR(X_train, y_train, X_test)
        Y_data_pred.loc[test_idx_lab, "EN"]     = EN(X_train, y_train, X_test, seed = 0)
        Y_data_pred.loc[test_idx_lab, "KNN"]    = KNN(X_train, y_train, X_test)
        Y_data_pred.loc[test_idx_lab, "ADB"]    = ADB(X_train, y_train, X_test, seed = 0)
        
        ## Ensemble prediction...
        Y_data_pred.loc[:, "ENS"]  = Y_data_pred.loc[:, np.array(models)[:5]].mean(axis = 1)
        for MM in models[6:]:
            Y_data_pred.loc[:, MM] = Y_data_pred.loc[:, map(lambda i: models[int(i)-1], MM.split("ENS")[1])].mean(axis = 1)
        #### Ensemble loop ends.
    #### CV loop ends.
    
    ## Performance evaluation & save results...
    for MM in models:
        RESULTS_all[MM].loc[drug, :] = EVAL_PERF(Y_data_pred["Actual"], Y_data_pred[MM])
    
    AUC_data_pred[drug] = Y_data_pred
#### Whole loop ends.

dt = time() - dt;    print("\nElapsed time = %0.4f sec." % dt)

RESULTS_all_MEAN    = pd.DataFrame({MM: RESULTS_all[MM].mean(axis = 0) for MM in models})
RESULTS_all["Mean"] = RESULTS_all_MEAN.T

print("Mean performance for %d inhibitors = \n" % N, RESULTS_all_MEAN.T)









