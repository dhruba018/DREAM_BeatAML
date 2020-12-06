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
# from sklearn.model_selection import KFold
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

### TEST ON LEADERBOARD DATA...
FS_switch = "ReliefF"
if FS_switch == "ReliefF":
    p_top = 15000
    # feature_ranks = {kk: [ ] for kk in drug_list}
    with open("FS_Drugs_122_ReliefF_3_folds.pickle", "rb") as file:
        feature_ranks = pickle.load(file)
elif FS_switch == "Lasso":
    feature_ranks = dict.fromkeys(drug_list)
####

AUC_LB_pred = { };      metrics = ["PCC", "SCC", "NRMSE", "NMAE"]
models = ["RF", "SVR", "EN", "KNN", "ADB", "ENS", "ENS1234", "ENS1235", "ENS1245", "ENS1345", "ENS2345", "ENS123", "ENS124", "ENS125", 
          "ENS134", "ENS135", "ENS145", "ENS234", "ENS235", "ENS245", "ENS345"]
RESULTS_LB = {MM: pd.DataFrame(dtype = float, index = drug_list, columns = metrics) for MM in models}

N = len(drug_list);     count = 0
dt = time() 
for drug in tqdm(drug_list[:N]):
    count += 1;   # print("\nChosen drug# = %d: %s" % (count, drug))
    
    y_TR = AUC_TR.iloc[(AUC_TR.inhibitor == drug).tolist(), :]
    y_LB = AUC_LB.iloc[(AUC_LB.inhibitor == drug).tolist(), :]
    X_TR, y_TR = RNA_TR_filt.loc[:, y_TR.lab_id].T, y_TR["auc"].to_numpy()
    X_LB, y_LB = RNA_LB_filt.loc[:, y_LB.lab_id].T, y_LB["auc"].to_numpy()
    
    Y_LB_pred = pd.DataFrame(dtype = float, index = X_LB.index, columns = ["Actual"] + models)
    
    fold = 0;      # dt = time()
    # ## Perform ReliefF...
    # dt = time();        FS.fit(X_TR.values, y_TR);      dt = time() - dt;    print("Elapsed time = %0.4f sec." % dt)
    # feat_top = FS.top_features_;    feature_ranks[drug].append(feat_top)

    ## From saved file...
    if FS_switch == "ReliefF":
        feat_top_set = feature_ranks[drug][fold]
        feat_top = feat_top_set[:p_top]
    elif FS_switch == "Lasso":
        feat_top = LassoFS(X_TR, y_TR, seed = 2020)
        feature_ranks[drug] = feat_top
    X_TR, X_LB = X_TR.iloc[:, feat_top], X_LB.iloc[:, feat_top]
    
    ## Perform prediction...
    Y_LB_pred.loc[:, "Actual"] = y_LB
    Y_LB_pred.loc[:, "RF"]     =  RF(X_TR, y_TR, X_LB, seed = 0)
    Y_LB_pred.loc[:, "SVR"]    = SVR(X_TR, y_TR, X_LB)
    Y_LB_pred.loc[:, "EN"]     =  EN(X_TR, y_TR, X_LB, seed = 0)
    Y_LB_pred.loc[:, "KNN"]    = KNN(X_TR, y_TR, X_LB)
    Y_LB_pred.loc[:, "ADB"]    = ADB(X_TR, y_TR, X_LB, seed = 0)
    
    RESULTS_LB["RF"].loc[drug, :]  = EVAL_PERF(Y_LB_pred["Actual"], Y_LB_pred["RF"])
    RESULTS_LB["SVR"].loc[drug, :] = EVAL_PERF(Y_LB_pred["Actual"], Y_LB_pred["SVR"])
    RESULTS_LB["EN"].loc[drug, :]  = EVAL_PERF(Y_LB_pred["Actual"], Y_LB_pred["EN"])
    RESULTS_LB["KNN"].loc[drug, :] = EVAL_PERF(Y_LB_pred["Actual"], Y_LB_pred["KNN"])
    RESULTS_LB["ADB"].loc[drug, :] = EVAL_PERF(Y_LB_pred["Actual"], Y_LB_pred["ADB"])
        
    # fold += 1
    # dt = time() - dt;    print("Elapsed time = %0.4f sec." % dt)
    
    ## Ensemble prediction...
    Y_LB_pred.loc[:, "ENS"]         = Y_LB_pred.loc[:, np.array(models)[:5]].mean(axis = 1)
    RESULTS_LB["ENS"].loc[drug, :]  = EVAL_PERF(Y_LB_pred["Actual"], Y_LB_pred["ENS"])
    for MM in models[6:]:
        Y_LB_pred.loc[:, MM]        = Y_LB_pred.loc[:, map(lambda i: models[int(i)-1], MM.split("ENS")[1])].mean(axis = 1)
        RESULTS_LB[MM].loc[drug, :] = EVAL_PERF(Y_LB_pred["Actual"], Y_LB_pred[MM])
    #### Ensemble loop ends...
    
    AUC_LB_pred[drug] = Y_LB_pred
#### Whole loop ends...

dt = time() - dt;    print("Elapsed time = %0.4f sec." % dt)

RESULTS_LB_MEAN    = pd.DataFrame({MM: RESULTS_LB[MM].mean(axis = 0) for MM in models})
RESULTS_LB["Mean"] = RESULTS_LB_MEAN.T

print("Mean performance for %d inhibitors = \n" % N, RESULTS_LB_MEAN.T)


#%%
# from sklearn.linear_model import LassoCV

# FS = Lasso(fit_intercept = False, normalize = False, alpha = 0.001, tol = 1e-3, 
#            selection = "random", random_state = 0)
# # FS = LassoCV(fit_intercept = True, normalize = True, tol = 1e-3, n_alphas = 200, cv = 5, 
# #              selection = "random", random_state = 0)
# aa = FS.fit(X_TR, y_TR).coef_
# print(sum(aa != 0))









   