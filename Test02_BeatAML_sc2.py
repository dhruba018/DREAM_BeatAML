# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:23:25 2020

@author: SRDhruba
"""


import os
os.chdir("C:\\Users\\SRDhruba\\Google Drive\\Study\\ECE 5332-009 - Topics in EE, Data Science\\BeatAML\\")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from beataml_example2_master.input_manager import InputManager
from beataml_example2_master.input_manager import RawInputs
from beataml_example2_master.model import makeFullFeatureVector
from sksurv.datasets import get_x_y
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored


# In[ ]:
raw_inputs = RawInputs('training')
raw_inputs.load()

im = InputManager(raw_inputs)
im.prepInputs()
im.printStats()

raw_inputs_lb = RawInputs("Leaderboard")
raw_inputs_lb.load()

im_lb = InputManager(raw_inputs_lb)
im_lb.prepInputs()
im_lb.printStats()

# In[ ]:
most_variant_genes = im.rnaseq_by_spec.var().nlargest(500).index
inhibitors = im.aucs.columns

lab_ids = im.getAllSpecimens()
feature_matrix = pd.DataFrame()
for lab_id in lab_ids:
    feature_vector = makeFullFeatureVector(im, most_variant_genes, inhibitors, lab_id)
    feature_series = pd.Series(data = feature_vector, name = lab_id)
    feature_matrix = feature_matrix.append(feature_series)

lab_ids_lb = im_lb.getAllSpecimens()
feature_matrix_lb = pd.DataFrame()
for lab_id in lab_ids_lb:
    feature_vector = makeFullFeatureVector(im_lb, most_variant_genes, inhibitors, lab_id)
    feature_series = pd.Series(data = feature_vector, name = lab_id)
    feature_matrix_lb = feature_matrix_lb.append(feature_series)

# In[ ]:
if 1:
    feature_means = feature_matrix.mean()
    feature_stds = feature_matrix.std()
    normed_features = (feature_matrix - feature_means) / feature_stds
    normed_features_lb = (feature_matrix_lb - feature_means) / feature_stds
else:
    normed_features = feature_matrix
    normed_features_lb = feature_matrix_lb
normed_features = normed_features.fillna(0.0)
normed_features_lb = normed_features_lb.fillna(0.0)

full_dataset = pd.read_csv('training/response.csv').set_index('lab_id').join(normed_features)
X, Y = get_x_y(full_dataset, ['vitalStatus', 'overallSurvival'], pos_label = 'Dead')

full_dataset_lb = pd.read_csv('leaderboard/response.csv').set_index('lab_id').join(normed_features_lb)
X_lb, Y_lb = get_x_y(full_dataset_lb, ['vitalStatus', 'overallSurvival'], pos_label = 'Dead')

# In[ ]
def plot_gridcv_results(gcv, alphas):
    scores = gcv.cv_results_['mean_test_score']
    scores_std = gcv.cv_results_['std_test_score']
    std_error = scores_std / np.sqrt(5)
    
    plt.figure().set_size_inches(8, 6)
    plt.semilogx(alphas, scores)
    plt.fill_between(alphas, scores + std_error, scores - std_error, alpha = 0.2)
    plt.xlabel('alpha')
    plt.ylabel('Concordance Index')
    plt.axvline(gcv.best_params_['alphas'][0], color = 'r', ls = '--', 
                label = ('Best alpha, CI = %0.3f' % gcv.best_score_))
    plt.legend()
    plt.title('Cross Validation Concordance Index')

def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y['vitalStatus'], y['overallSurvival'], prediction)
    return result[0]

# In[ ]:
# This package allows general elastic net tuning, but by setting l1_ratio = 1, we restrict to LASSO.
regr = CoxnetSurvivalAnalysis(l1_ratio = 0.8, alpha_min_ratio = 0.1, max_iter = 3e5)

n_folds = 10

alphas = np.logspace(-1.3, 1.5, num = 50)
cv = KFold(n_splits = 5, shuffle = True, random_state = 0)
gcv = GridSearchCV(regr, {"alphas": [[v] for v in alphas]}, cv = cv, n_jobs = -1).fit(X, Y)

plot_gridcv_results(gcv, alphas)
regr_best = CoxnetSurvivalAnalysis(alphas = gcv.best_params_["alphas"], l1_ratio = 0.8, alpha_min_ratio = 0.1, max_iter = 3e5).fit(X, Y)
y_regr = regr_best.predict(X_lb)

ci_lb = concordance_index_censored(Y_lb["vitalStatus"], Y_lb["overallSurvival"], y_regr)[0]
print("concordance index = %0.4f" % ci_lb)


# In[ ]
zero_mask = np.array([Y[ii][1] == 0 for ii in range(len(Y))])

surv_mdl = FastSurvivalSVM(rank_ratio = 0.8, fit_intercept = True, optimizer = "rbtree", tol = 1e-4, max_iter = 100, random_state = 0)

param_grid = {'alpha': np.logspace(-2, 2, num = 100)}
cv = KFold(n_splits = 5, shuffle = True, random_state = 0)
grid_cv = GridSearchCV(surv_mdl, param_grid, scoring = score_survival_model, n_jobs = -1, cv = cv)
grid_cv.fit(X[~zero_mask], Y[~zero_mask])

plot_gridcv_results(grid_cv, param_grid["alpha"])
surv_mdl_best = FastSurvivalSVM(alpha = grid_cv.best_params_["alpha"], rank_ratio = 0.8, fit_intercept = True, optimizer = "rbtree", tol = 1e-4, max_iter = 100, random_state = 0).fit(X, Y)
y_pred = surv_mdl_best.predict(X_lb)

ci_lb = concordance_index_censored(Y_lb["vitalStatus"], Y_lb["overallSurvival"], y_pred)[0]
print("concordance index = %0.4f" % ci_lb)


# import matplotlib.pyplot as plt
# from sksurv.nonparametric import kaplan_meier_estimator

# time, survival_prob = kaplan_meier_estimator(Y["vitalStatus"], Y["overallSurvival"])
# plt.step(time, survival_prob, where = "post")
# plt.ylabel("est. probability of survival $\hat{S}(t)$")
# plt.xlabel("time $t$")



