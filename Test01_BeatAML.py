#!/usr/bin/env python
# coding: utf-8

# #### CTD$^2$ Beat AML Challenge
# ##### Subchallenge 1
# _Data files._ 
# * Genetic: rnaseq.csv, dnaseq.csv 
# * Clinical: clinical_numerical.csv, clinical_categorical.csv, clinical_categorical_legend.csv 
# * Response: aucs.csv, response.csv 
# 
# _Goal._ Predict AUC from the given data.
# 
# _Steps._ 
# * Read RNA-seq, DNA-seq, AUC datasets 
# * Used DNA-seq for feature selection on RNA-seq 
# * Model AUC using RNA-seq 
# 
# 

# In[33]:


import os
# import sys
import numpy as np
import scipy as sp
import pandas as pd
import scipy.linalg as alg
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
PATH = "%s\\Google Drive\\Study\\ECE 5332-009 - Topics in EE, Data Science\\BeatAML\\" % os.getenv("HOMEPATH")
os.chdir(PATH)

## Read data...
RNA = pd.read_csv(PATH + "rnaseq.csv", header = 0)
DNA = pd.read_csv(PATH + "dnaseq.csv", header = 0)
AUC = pd.read_csv(PATH + "aucs.csv",   header = 0)
# RESP = pd.read_csv(PATH + "response.csv", header = 0)

# print("Dataset sizes = \n", pd.DataFrame([RNA.shape, DNA.shape, AUC.shape], index = ["RNA", "DNA", "AUC"], columns = ["row", "col"]))


# In[2]:


## Feature selection...
variant_count = pd.DataFrame([[vv, (DNA.Hugo_Symbol.to_numpy() == vv).sum()] for vv in DNA.Hugo_Symbol.unique()], 
                             columns = ["gene", "counts"])
variant_count = variant_count.iloc[variant_count.counts.argsort()[::-1], :].reset_index(drop = True)
print("#genes with somatic variants = %d\n#genes with variant count > 1 = %d" % 
      (variant_count.shape[0], (variant_count.counts > 1).sum()))
# print("Top 5 genes =\n", variant_count[:5])

# Check if all genes are in RNA-seq...
all([(gg == RNA.Symbol).sum() > 0 for gg in variant_count.gene])
genes, gene_idx, _ = np.intersect1d(RNA.Symbol, variant_count.gene, assume_unique = False, return_indices = True)
RNA_filt = RNA.iloc[gene_idx, :].copy();    print("#genes used = %d" % len(genes))

## Drug information...
drugs, drug_info, drug_sample_count = AUC.inhibitor.unique().tolist(), [ ], [ ]
for dd in drugs:
    drug_info.append((dd, AUC.iloc[(AUC.inhibitor == dd).to_numpy(), :]))
    drug_sample_count.append([dd, (AUC.inhibitor == dd).sum()])
drug_info, drug_sample_count = dict(drug_info), pd.DataFrame(drug_sample_count, columns = ["inhibitor", "lab_id_count"])
print("\n#inhibitors used = %d" % len(drug_info))
# print("Top 5 inhibitors = \n", drug_sample_count[:5])

gene_var = RNA_filt.iloc[:, 2:].var(axis = 1)
RNA_filt2 = RNA_filt.iloc[(gene_var > 0.3).to_numpy(), :]
genes2 = genes[(gene_var > 0.3).to_numpy()]

print(RNA_filt2.shape)
print(genes2.shape)


# In[*]

from skrebate import ReliefF

gene_ranks = pd.DataFrame(dtype = int, columns = drugs, index = RNA.Symbol)
FS = ReliefF(n_features_to_select = RNA.shape[0], n_neighbors = 20)
for dd in tqdm(drugs):
    yy = drug_info[dd]





# In[3]:


## Function definitions...
def RF(X_train, y_train, X_test, seed = None):
    mdl = RandomForestRegressor(n_estimators = 100, criterion = 'mse', random_state = seed)
    mdl.fit(X_train, y_train);     y_pred = mdl.predict(X_test)
    return y_pred

def SVM(X_train, y_train, X_test, seed):
    mdl = SupportVectorRegressor(kernel = "poly", degree = 3, gamma = "auto", tol = 1e-3)
    mdl.fit(X_train, y_train);     y_pred = mdl.predict(X_test)
    return y_pred

def PerfEval(y, y_hat, corr = "PCC", err = "NRMSE", alpha = 0.05, return_pval = False):
    ## Base functions...
    CORR  = lambda y, y_hat, c_type: pearsonr(y, y_hat) if c_type == "PCC" else spearmanr(y, y_hat)
    ERROR = lambda y, y_hat, e_ord: alg.norm(y - y_hat, ord = e_ord) ** e_ord / y.size
    
    ## Format input...
    y, y_hat = np.array(y, dtype = float), np.array(y_hat, dtype = float)
    if y.ndim == 2 and any([ss == 1 for ss in y.shape]):
        y = y.squeeze()
    elif y.ndim > 2:
        print("Error! y should be a ndarray of shape (n, ) or (n, 1) or (1, n).")
    ####
    if y_hat.ndim == 2 and any([ss == 1 for ss in y_hat.shape]):
        y_hat = y_hat.squeeze()
    elif y_hat.ndim > 2:
        print("Error! y_hat should be a ndarray of shape (n, ) or (n, 1) or (1, n).")
    ####
    
    #### Calculate metrics...
    if corr.upper() == "PCC" or "SCC":
        rho, pval = CORR(y, y_hat, c_type = corr.upper())
        rho = 0 if pval > alpha else rho
        print("No significant correlation found!") if pval > alpha else None
    else:
        print('Not valid! Use "PCC" or "SCC" for corr option.')
    ####
    
    if "MAE" in err.upper():
        eps = ERROR(y, y_hat, e_ord = 1)
        if err.upper() == "NMAE":
            eps /= ERROR(y, y.mean(), e_ord = 1)
        else:
            print('Invalid option! Use a variation of "MSE" or "MAE" for err option.')
    elif "MSE" in err.upper():
        eps = ERROR(y, y_hat, e_ord = 2)
        if err.upper() == "RMSE":
            eps = sp.sqrt(eps)
        elif err.upper() == "NRMSE":
            eps = sp.sqrt(eps) / y.std(ddof = 0)
        else:
            print('Invalid option! Use a variation of "MSE" or "MAE" for err option.')
    else:
        print('Invalid option! Use a variation of "MSE" or "MAE" for err option.')
    ####
    
    return (rho, eps, pval) if return_pval else (rho, eps)
####


# In[4]:


## Pick drug and build dataset for modeling...
drug_picked = list(drug_info.keys())[0];    sample_ids = drug_info[drug_picked]["lab_id"]
print("Drug picked for analysis = %s\n#samples = %d" % (drug_picked, len(sample_ids)))

RNA_drug = RNA.loc[:, sample_ids];    RNA_drug.index = RNA.Symbol
AUC_drug = AUC.iloc[(AUC.inhibitor == drug_picked).to_numpy(), 2]
p_top = 100
FS = ReliefF(n_features_to_select = p_top, n_neighbors = 20)
FS.fit(RNA_drug.T.values, AUC_drug.values)

X_data = RNA_drug.iloc[FS.top_features_, :].T
y_data = AUC_drug

# X_data = pd.DataFrame(RNA_filt2.loc[:, sample_ids].to_numpy().T, index = sample_ids, columns = genes2)
# y_data = pd.DataFrame(drug_info[drug_picked]["auc"].to_numpy(), index = sample_ids, columns = ["auc"])
print("Analysis dataset size = \n", pd.DataFrame([X_data.shape, y_data.shape], index = ["X", "y"], columns = ["row", "col"]))

## Perform prediction...
models = ["RF", "SVM"];    n_sample, n_model = len(sample_ids), len(models)
Y_data = pd.DataFrame(dtype = float, index = sample_ids, columns = ["Actual"] + models)
Y_data["Actual"] = y_data
CV = KFold(n_splits = 5, shuffle = False, random_state = None)
for train_idx, test_idx in tqdm(CV.split(X_data)):
    train_id, test_id = sample_ids.iloc[train_idx], sample_ids.iloc[test_idx]
    X_train, y_train = X_data.loc[train_id, :], y_data.iloc[train_idx]
    X_test,  y_test  = X_data.loc[test_id, :],  y_data.iloc[test_idx]
    
    Y_data.loc[test_id, "RF"]  =  RF(X_train, y_train, X_test, seed = None)
    Y_data.loc[test_id, "SVM"] = SVM(X_train, y_train, X_test, seed = None)
####

metrics = ["SCC", "NRMSE"]
PERF = pd.DataFrame(dtype = float, index = metrics, columns = models)
for mm in models:
    PERF.loc[:, mm] = PerfEval(y_data, Y_data[mm], corr = metrics[0], err = metrics[1], return_pval = False)
####
print("Model performances for 5-fold cross validation = \n", PERF)


# In[32]:


p_top = 50
X_data = RNA_drug.iloc[FS.top_features_[:p_top], :].T
y_data = AUC_drug

# X_data = pd.DataFrame(RNA_filt2.loc[:, sample_ids].to_numpy().T, index = sample_ids, columns = genes2)
# y_data = pd.DataFrame(drug_info[drug_picked]["auc"].to_numpy(), index = sample_ids, columns = ["auc"])
# print("Analysis dataset size = \n", pd.DataFrame([X_data.shape, y_data.shape], index = ["X", "y"], columns = ["row", "col"]))

## Perform prediction...
models = ["RF", "SVM"];    n_sample, n_model = len(sample_ids), len(models)
Y_data = pd.DataFrame(dtype = float, index = sample_ids, columns = ["Actual"] + models)
Y_data["Actual"] = y_data
CV = KFold(n_splits = 5, shuffle = False, random_state = None)
for train_idx, test_idx in tqdm(CV.split(X_data)):
    train_id, test_id = sample_ids.iloc[train_idx], sample_ids.iloc[test_idx]
    X_train, y_train = X_data.loc[train_id, :], y_data.iloc[train_idx]
    X_test,  y_test  = X_data.loc[test_id, :],  y_data.iloc[test_idx]
    
    Y_data.loc[test_id, "RF"]  =  RF(X_train, y_train, X_test, seed = 0)
    Y_data.loc[test_id, "SVM"] = SVM(X_train, y_train, X_test, seed = 0)
####

metrics = ["SCC", "NRMSE"]
PERF = pd.DataFrame(dtype = float, index = metrics, columns = models)
for mm in models:
    PERF.loc[:, mm] = PerfEval(y_data, Y_data[mm], corr = metrics[0], err = metrics[1], return_pval = False)
####
print("Model performances for 5-fold cross validation = \n", PERF)


# In[ ]:


# %qtconsole --style monokai

