#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
# import sys
import numpy as np
# import scipy as sp
import pandas as pd
# import scipy.linalg as alg
import pickle
# import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from skrebate import ReliefF
from sklearn.model_selection import KFold
from sklearn.svm import SVR as SupportVectorRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr

## Path & Filename...
PATH     = os.path.join(os.getenv("HOMEPATH"), "Google Drive\\Study\\ECE 5332-009 - Topics in EE, Data Science\\BeatAML\\")
DIRNAME  = ("Training\\", "Leaderboard\\")
FILENAME = ("rnaseq.csv", "dnaseq.csv", "clinical_numerical.csv", "clinical_categorical.csv", 
            "clinical_categorical_legend.csv", "aucs.csv", "response.csv")
os.chdir(PATH)

### Read data...
## Training...
RNA            = pd.read_csv(PATH + DIRNAME[0] + FILENAME[0], header = 0)
# DNA            = pd.read_csv(PATH + DIRNAME[0] + FILENAME[1], header = 0)
# CLI_NUM        = pd.read_csv(PATH + DIRNAME[0] + FILENAME[2], header = 0)
# CLI_CAT        = pd.read_csv(PATH + DIRNAME[0] + FILENAME[3], header = 0)
# CLI_CAT_LEG    = pd.read_csv(PATH + DIRNAME[0] + FILENAME[4], header = 0)

AUC            = pd.read_csv(PATH + DIRNAME[0] + FILENAME[5], header = 0)
# RESP           = pd.read_csv(PATH + DIRNAME[0] + FILENAME[6], header = 0)

## Leaderboard...
RNA_LB         = pd.read_csv(PATH + DIRNAME[1] + FILENAME[0], header = 0)
# DNA_LB         = pd.read_csv(PATH + DIRNAME[1] + FILENAME[1], header = 0)
# CLI_NUM_LB     = pd.read_csv(PATH + DIRNAME[1] + FILENAME[2], header = 0)
# CLI_CAT_LB     = pd.read_csv(PATH + DIRNAME[1] + FILENAME[3], header = 0)
# CLI_CAT_LEG_LB = pd.read_csv(PATH + DIRNAME[1] + FILENAME[4], header = 0)

AUC_LB         = pd.read_csv(PATH + DIRNAME[1] + FILENAME[5], header = 0)
# RESP_LB        = pd.read_csv(PATH + DIRNAME[1] + FILENAME[6], header = 0)

## Processing...
all(AUC.inhibitor.unique()  == AUC_LB.inhibitor.unique())     ## Check if same drugs
all(RNA[["Gene", "Symbol"]] == RNA_LB[["Gene", "Symbol"]])    ## Check if same genes

lab_id_list = dict(TR = RNA.columns[2:], LB = RNA_LB.columns[2:])
gene_list, drug_list = RNA[["Gene", "Symbol"]], AUC.inhibitor.unique().tolist()

RNA.index, RNA_LB.index = gene_list.Symbol, gene_list.Symbol
RNA, RNA_LB = RNA.iloc[:, 2:], RNA_LB.iloc[:, 2:]
# RNA.shape, RNA_LB.shape

var_idx = (RNA.var(axis = 1) > 0.1).to_numpy()                ## Filter genes w/ low variability
RNA_filt, RNA_LB_filt = RNA.iloc[var_idx, :], RNA_LB.iloc[var_idx, :]



# In[ ]:


## Function definitions...
def RF(X_train, y_train, X_test, seed = 0):
    mdl = RandomForestRegressor(n_estimators = 200, criterion = "mae", min_samples_leaf = 5, random_state = seed)
    y_pred = mdl.fit(X_train, y_train).predict(X_test)
    return y_pred

def SVR(X_train, y_train, X_test):
    mdl = SupportVectorRegressor(kernel = "poly", degree = 3, gamma = "scale", tol = 1e-3, C = 100)
    y_pred = mdl.fit(X_train, y_train).predict(X_test)
    return y_pred

def EN(X_train, y_train, X_test, seed):
    mdl = ElasticNet(fit_intercept = True, l1_ratio = 0.8, alpha = 0.2, tol = 1e-3, random_state = seed)
    y_pred = mdl.fit(X_train, y_train).predict(X_test)
    return y_pred

def EVAL_PERF(y_label, y_pred, alpha = 0.05):
    y_label, y_pred = np.array(y_label).squeeze(), np.array(y_pred).squeeze()
    PCC, pval = pearsonr(y_label, y_pred);     #PCC = PCC if pval < alpha else 0
    SCC, pval = spearmanr(y_label, y_pred);    #SCC = SCC if pval < alpha else 0
    NRMSE = np.sqrt(((y_label - y_pred)**2).mean()) / y_label.std(ddof = 0)
    NMAE  = (np.abs(y_label - y_pred)).mean() / (np.abs(y_label - y_label.mean())).mean()
    return PCC, SCC, NRMSE, NMAE


# In[ ]:

### ORIGINAL ANALYSIS CELL...

drug_list = AUC.inhibitor.unique().tolist()
RNA_filt = RNA.iloc[(RNA.var(axis = 1) > 0.1).to_numpy(), :]

FS = ReliefF(n_features_to_select = 1000, n_neighbors = 10, n_jobs = 1);    p_top = 1500
feature_ranks = {kk: [ ] for kk in drug_list}
RESULTS_RF  = pd.DataFrame(dtype = float, index = drug_list, columns = ["PCC", "SCC", "NRMSE", "NMAE"])
RESULTS_SVR = pd.DataFrame(dtype = float, index = drug_list, columns = ["PCC", "SCC", "NRMSE", "NMAE"])
RESULTS_EN  = pd.DataFrame(dtype = float, index = drug_list, columns = ["PCC", "SCC", "NRMSE", "NMAE"])
RESULTS_ENS = pd.DataFrame(dtype = float, index = drug_list, columns = ["PCC", "SCC", "NRMSE", "NMAE"])
N = len(drug_list);     count = 76
for drug in tqdm(drug_list[76:N]):              ## Continue from where left off
    count +=1;    print("\nChosen drug# = %d: %s" % (count, drug))
    y_data = AUC.iloc[(AUC.inhibitor == drug).tolist(), :]
    X_data, y_data = RNA_filt.loc[:, y_data.lab_id].T, y_data["auc"].to_numpy()
    
    Y_pred = pd.DataFrame(dtype = float, index = X_data.index, columns = ["Actual", "RF", "SVR", "EN", "ENS"])
    CV = KFold(n_splits = 3, shuffle = False, random_state = 0)
    dt = time()
    for train_idx, test_idx in CV.split(X_data):
        X_train, y_train = X_data.iloc[train_idx, :], y_data[train_idx]
        X_test,  y_test  = X_data.iloc[test_idx, :],  y_data[test_idx]
        
        ## Perform ReliefF...
        # dt = time();         
        FS.fit(X_train.values, y_train)
        # dt = time() - dt;    print("Elapsed time = %0.4f sec." % dt)
        feat_top = FS.top_features_;    feature_ranks[drug].append(feat_top)
        X_train, X_test = X_train.iloc[:, feat_top[:p_top]], X_test.iloc[:, feat_top[:p_top]]
        
        ## Perform prediction...
        test_idx_lab = X_data.index[test_idx]
        Y_pred.loc[test_idx_lab, "Actual"] = y_test
        Y_pred.loc[test_idx_lab, "RF"]     = RF(X_train, y_train, X_test, seed = 0)
        Y_pred.loc[test_idx_lab, "SVR"]    = SVR(X_train, y_train, X_test)
        Y_pred.loc[test_idx_lab, "EN"]     = EN(X_train, y_train, X_test, seed = 0)
        # Y_pred.loc[test_idx_lab, "ENS"]    = Y_pred.loc[test_idx_lab, ["RF", "SVR", "EN"]].mean(axis = 1)
    #### CV loop ends.
    dt = time() - dt;    print("Elapsed time = %0.4f sec." % dt)
    
    ## Ensemble prediction...
    Y_pred.loc[:, "ENS"] = Y_pred.loc[:, ["RF", "SVR", "EN"]].mean(axis = 1)
    
    ## Save results...
    # PCC, SCC, NEMSE, NMAE = EVAL_PERF(Y_pred["Actual"], Y_pred["RF"])
    RESULTS_RF.loc[drug, :]  = EVAL_PERF(Y_pred["Actual"], Y_pred["RF"])
    RESULTS_SVR.loc[drug, :] = EVAL_PERF(Y_pred["Actual"], Y_pred["SVR"])
    RESULTS_EN.loc[drug, :]  = EVAL_PERF(Y_pred["Actual"], Y_pred["EN"])
    RESULTS_ENS.loc[drug, :] = EVAL_PERF(Y_pred["Actual"], Y_pred["ENS"])
    
    ## Display results...
    RES_TAB = pd.concat((RESULTS_RF.loc[drug, :], RESULTS_SVR.loc[drug, :], RESULTS_EN.loc[drug, :], RESULTS_ENS.loc[drug, :]), 
                        axis = 1, ignore_index = True)
    RES_TAB.columns = ["RF", "SVR", "EN", "ENS"];         print("\n", RES_TAB)
    
    # print(RESULTS_RF.loc[drug, :])
    # print(RESULTS_SVR.loc[drug, :])
    # print(RESULTS_EN.loc[drug, :])
    ####
        


# In[ ]:


# %qtconsole --style monokai

with open("FS1_Drugs_25.pickle", "wb") as file:
    pickle.dump(feature_ranks, file, protocol = pickle.HIGHEST_PROTOCOL)

with open("FS2_Drugs_50.pickle", "wb") as file:
    pickle.dump(feature_ranks, file, protocol = pickle.HIGHEST_PROTOCOL)


with open("FS1_Drugs_25_gn.pickle", "rb") as file:
    aa1 = pickle.load(file)

with open("FS2_Drugs_51.pickle", "rb") as file:
    aa2 = pickle.load(file)

aa3 = {kk: [ ] for kk in aa1.keys()}
for kk in aa2.keys():
    if len(aa1[kk]) == 3:
        aa3[kk] = aa1[kk]
    elif len(aa2[kk]) == 3:
        aa3[kk] = aa2[kk]
aa3["gene_names"] = aa1["gene_names"]


with open("FS_Drugs_76.pickle", "wb") as file:
    pickle.dump(aa3, file, protocol = pickle.HIGHEST_PROTOCOL)

with open("FS_Drugs_76.pickle", "rb") as file:
    feature_ranks = pickle.load(file)


with open("FS_Drugs_122_ReliefF_3_folds.pickle", "wb") as file:
    pickle.dump(feature_ranks, file, protocol = pickle.HIGHEST_PROTOCOL)


# In[ ]:

### CELL FOR AFTER GETTING FEATURE RANKS...

# drug_list = AUC.inhibitor.unique().tolist()
# RNA_filt = RNA.iloc[(RNA.var(axis = 1) > 0.1).to_numpy(), :]

# FS = ReliefF(n_features_to_select = 1000, n_neighbors = 10, n_jobs = 1);    
p_top = 100
# feature_ranks = {kk: [ ] for kk in drug_list}
# with open("FS_Drugs_76.pickle", "rb") as file:
#     feature_ranks = pickle.load(file)
with open("FS_Drugs_122_ReliefF_3_folds.pickle", "rb") as file:
    feature_ranks = pickle.load(file)

RESULTS_RF  = pd.DataFrame(dtype = float, index = drug_list, columns = ["PCC", "SCC", "NRMSE", "NMAE"])
RESULTS_SVR = pd.DataFrame(dtype = float, index = drug_list, columns = ["PCC", "SCC", "NRMSE", "NMAE"])
RESULTS_EN  = pd.DataFrame(dtype = float, index = drug_list, columns = ["PCC", "SCC", "NRMSE", "NMAE"])
RESULTS_ENS = pd.DataFrame(dtype = float, index = drug_list, columns = ["PCC", "SCC", "NRMSE", "NMAE"])
N = len(drug_list);     count = 0
dt = time()
for drug in tqdm(drug_list[:N]):              ## Continue from where left off
    count += 1;   # print("\nChosen drug# = %d: %s" % (count, drug))
    y_data = AUC.iloc[(AUC.inhibitor == drug).tolist(), :]
    X_data, y_data = RNA_filt.loc[:, y_data.lab_id].T, y_data["auc"].to_numpy()
    
    Y_pred = pd.DataFrame(dtype = float, index = X_data.index, columns = ["Actual", "RF", "SVR", "EN", "ENS"])
    CV = KFold(n_splits = 3, shuffle = False, random_state = 0)
    fold = 0;      # dt = time()
    for train_idx, test_idx in CV.split(X_data):
        X_train, y_train = X_data.iloc[train_idx, :], y_data[train_idx]
        X_test,  y_test  = X_data.iloc[test_idx, :],  y_data[test_idx]
        
        ## Perform ReliefF...
        # dt = time();         
        # FS.fit(X_train.values, y_train)
        # dt = time() - dt;    print("Elapsed time = %0.4f sec." % dt)
        # feat_top = FS.top_features_;    feature_ranks[drug].append(feat_top)
        feat_top = feature_ranks[drug][fold]        
        X_train, X_test = X_train.iloc[:, feat_top[:p_top]], X_test.iloc[:, feat_top[:p_top]]
        
        ## Perform prediction...
        test_idx_lab = X_data.index[test_idx]
        Y_pred.loc[test_idx_lab, "Actual"] = y_test
        Y_pred.loc[test_idx_lab, "RF"]     = RF(X_train, y_train, X_test, seed = 0)
        Y_pred.loc[test_idx_lab, "SVR"]    = SVR(X_train, y_train, X_test)
        Y_pred.loc[test_idx_lab, "EN"]     = EN(X_train, y_train, X_test, seed = 0)
        # Y_pred.loc[test_idx_lab, "ENS"]    = Y_pred.loc[test_idx_lab, ["RF", "SVR", "EN"]].mean(axis = 1)
        fold += 1
    #### CV loop ends.
    # dt = time() - dt;    print("Elapsed time = %0.4f sec." % dt)
        
    ## Ensemble prediction...
    Y_pred.loc[:, "ENS"] = Y_pred.loc[:, ["RF", "SVR", "EN"]].mean(axis = 1)
    
    ## Save results...
    # PCC, SCC, NEMSE, NMAE = EVAL_PERF(Y_pred["Actual"], Y_pred["RF"])
    RESULTS_RF.loc[drug, :]  = EVAL_PERF(Y_pred["Actual"], Y_pred["RF"])
    RESULTS_SVR.loc[drug, :] = EVAL_PERF(Y_pred["Actual"], Y_pred["SVR"])
    RESULTS_EN.loc[drug, :]  = EVAL_PERF(Y_pred["Actual"], Y_pred["EN"])
    RESULTS_ENS.loc[drug, :] = EVAL_PERF(Y_pred["Actual"], Y_pred["ENS"])
    
    # ## Display results...
    # RES_TAB = pd.concat((RESULTS_RF.loc[drug, :], RESULTS_SVR.loc[drug, :], RESULTS_EN.loc[drug, :], 
    #                      RESULTS_ENS.loc[drug, :]), axis = 1, ignore_index = True)
    # RES_TAB.columns = ["RF", "SVR", "EN", "ENS"];         print("\n", RES_TAB)
    
    # print(RESULTS_RF.loc[drug, :])
    # print(RESULTS_SVR.loc[drug, :])
    # print(RESULTS_EN.loc[drug, :])
#### Whole loop ends...
dt = time() - dt;    print("Elapsed time = %0.4f sec." % dt)

RESULTS_MEAN = pd.concat(((RESULTS_RF.iloc[:N, :]).mean(axis = 0), (RESULTS_SVR.iloc[:N, :]).mean(axis = 0), 
                          (RESULTS_EN.iloc[:N, :]).mean(axis = 0), (RESULTS_ENS.iloc[:N, :]).mean(axis = 0)), 
                         axis = 1, ignore_index = True)
RESULTS_MEAN.columns = ["RF", "SVR", "EN", "ENS"];         print("Mean performance for %d inhibitors = \n" % N, RESULTS_MEAN)


# In[ ]

### TEST ON LEADERBOARD DATA...
p_top = 500
# feature_ranks = {kk: [ ] for kk in drug_list}
with open("FS_Drugs_122_ReliefF_3_folds.pickle", "rb") as file:
    feature_ranks = pickle.load(file)

RESULTS_LB_RF  = pd.DataFrame(dtype = float, index = drug_list, columns = ["PCC", "SCC", "NRMSE", "NMAE"])
RESULTS_LB_SVR = pd.DataFrame(dtype = float, index = drug_list, columns = ["PCC", "SCC", "NRMSE", "NMAE"])
RESULTS_LB_EN  = pd.DataFrame(dtype = float, index = drug_list, columns = ["PCC", "SCC", "NRMSE", "NMAE"])
RESULTS_LB_ENS = pd.DataFrame(dtype = float, index = drug_list, columns = ["PCC", "SCC", "NRMSE", "NMAE"])
N = len(drug_list);     count = 0
dt = time()
for drug in tqdm(drug_list[:N]):              ## Continue from where left off
    count += 1;   # print("\nChosen drug# = %d: %s" % (count, drug))
    
    y_TR, y_LB = AUC.iloc[(AUC.inhibitor == drug).tolist(), :], AUC_LB.iloc[(AUC_LB.inhibitor == drug).tolist(), :]
    X_TR, y_TR = RNA_filt.loc[:, y_TR.lab_id].T,    y_TR["auc"].to_numpy()
    X_LB, y_LB = RNA_LB_filt.loc[:, y_LB.lab_id].T, y_LB["auc"].to_numpy()
    
    Y_LB_pred = pd.DataFrame(dtype = float, index = X_LB.index, columns = ["Actual", "RF", "SVR", "EN", "ENS"])
    
    fold = 0;      # dt = time()
    # ## Perform ReliefF...
    # dt = time();        FS.fit(X_TR.values, y_TR);      dt = time() - dt;    print("Elapsed time = %0.4f sec." % dt)
    # feat_top = FS.top_features_;    feature_ranks[drug].append(feat_top)

    ## From saved file...
    feat_top = feature_ranks[drug][fold]
    X_TR, X_LB = X_TR.iloc[:, feat_top[:p_top]], X_LB.iloc[:, feat_top[:p_top]]
    
    ## Perform prediction...
    Y_LB_pred.loc[:, "Actual"] = y_LB
    Y_LB_pred.loc[:, "RF"]     = RF(X_TR, y_TR, X_LB, seed = 0)
    Y_LB_pred.loc[:, "SVR"]    = SVR(X_TR, y_TR, X_LB)
    Y_LB_pred.loc[:, "EN"]     = EN(X_TR, y_TR, X_LB, seed = 0)
    # fold += 1
    # dt = time() - dt;    print("Elapsed time = %0.4f sec." % dt)
        
    ## Ensemble prediction...
    Y_LB_pred.loc[:, "ENS"] = Y_LB_pred.loc[:, ["RF", "SVR", "EN"]].mean(axis = 1)
    
    ## Save results...
    RESULTS_LB_RF.loc[drug, :]  = EVAL_PERF(Y_LB_pred["Actual"], Y_LB_pred["RF"])
    RESULTS_LB_SVR.loc[drug, :] = EVAL_PERF(Y_LB_pred["Actual"], Y_LB_pred["SVR"])
    RESULTS_LB_EN.loc[drug, :]  = EVAL_PERF(Y_LB_pred["Actual"], Y_LB_pred["EN"])
    RESULTS_LB_ENS.loc[drug, :] = EVAL_PERF(Y_LB_pred["Actual"], Y_LB_pred["ENS"])
    
    # ## Display results...
    # RES_LB_TAB = pd.concat((RESULTS_LB_RF.loc[drug, :], RESULTS_LB_SVR.loc[drug, :], RESULTS_LB_EN.loc[drug, :], 
    #                         RESULTS_LB_ENS.loc[drug, :]), axis = 1, ignore_index = True)
    # RES_LB_TAB.columns = ["RF", "SVR", "EN", "ENS"];         print("\n", RES_LB_TAB)
    
    # print(RESULTS_RF.loc[drug, :])
    # print(RESULTS_SVR.loc[drug, :])
    # print(RESULTS_EN.loc[drug, :])
#### Whole loop ends...
dt = time() - dt;    print("Elapsed time = %0.4f sec." % dt)

RESULTS_LB_MEAN = pd.concat(((RESULTS_LB_RF.iloc[:N, :]).mean(axis = 0), (RESULTS_LB_SVR.iloc[:N, :]).mean(axis = 0),
                             (RESULTS_LB_EN.iloc[:N, :]).mean(axis = 0), (RESULTS_LB_ENS.iloc[:N, :]).mean(axis = 0)), 
                            axis = 1, ignore_index = True)
RESULTS_LB_MEAN.columns = ["RF", "SVR", "EN", "ENS"];    print("Mean performance for %d inhibitors = \n" % N, RESULTS_LB_MEAN)














   