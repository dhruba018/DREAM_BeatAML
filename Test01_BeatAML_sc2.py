# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 03:32:29 2020

@author: SRDhruba
"""


# In[ ]:
# %pip install scikit-survival synapseclient
# In[ ]:
# import getpass
# import synapseclient
# import synapseutils

# syn = synapseclient.Synapse()
# syn.login(input(prompt="Enter Synapse Username"), getpass.getpass("Enter Synapse Password"))
# downloaded_files = synapseutils.syncFromSynapse(syn, 'syn21212904', path='training')
# Now, load the data, and train a model!
# In[ ]:
# Auto-reload the custom python modules, for easy development.

# %load_ext autoreload
# %aimport input_manager
# %aimport model
# %autoreload 1

import os
os.chdir("C:\\Users\\SRDhruba\\Google Drive\\Study\\ECE 5332-009 - Topics in EE, Data Science\\BeatAML\\")

from beataml_example2_master.input_manager import RawInputs

raw_inputs = RawInputs('training')
raw_inputs.load()
# In[ ]:
from beataml_example2_master.input_manager import InputManager

im = InputManager(raw_inputs)
im.prepInputs()
im.printStats()
# In[ ]:
most_variant_genes = im.rnaseq_by_spec.var().nlargest(1000).index
inhibitors = im.aucs.columns
# In[ ]:
import numpy as np
import pandas as pd

from beataml_example2_master.model import makeFullFeatureVector

lab_ids = im.getAllSpecimens()
feature_matrix = pd.DataFrame()
for lab_id in lab_ids:
    feature_vector = makeFullFeatureVector(im, most_variant_genes, inhibitors, lab_id)
    feature_series = pd.Series(data = feature_vector, name = lab_id)
    feature_matrix = feature_matrix.append(feature_series)
# In[ ]:
feature_means = feature_matrix.mean()
feature_stds = feature_matrix.std()
normed_features = (feature_matrix - feature_means) / feature_stds
normed_features = normed_features.fillna(0.0)
# In[ ]:
from sksurv.datasets import get_x_y
full_dataset = pd.read_csv('training/response.csv').set_index('lab_id').join(normed_features)
X, Y = get_x_y(full_dataset, ['vitalStatus', 'overallSurvival'], pos_label = 'Dead')
# In[ ]:
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# This package allows general elastic net tuning, but by setting
# l1_ratio=1, we restrict to LASSO.
regr = CoxnetSurvivalAnalysis(l1_ratio = 1, alpha_min_ratio = 0.05, max_iter = 3e5)

n_folds = 10

alphas = np.logspace(-1.3, 0, num = 100)
cv = KFold(n_splits = 5, shuffle = True, random_state = 328)
gcv = GridSearchCV(regr, {"alphas": [[v] for v in alphas]}, cv = cv).fit(X, Y)
#In[ ]:
import matplotlib.pyplot as plt

scores = gcv.cv_results_['mean_test_score']
scores_std = gcv.cv_results_['std_test_score']
std_error = scores_std / np.sqrt(n_folds)

plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha = 0.2)
plt.xlabel('alpha')
plt.ylabel('Concordance Index')
plt.axvline(gcv.best_params_['alphas'][0], color = 'r', ls = '--', label = ('Best alpha, CI = %0.3f' % gcv.best_score_))
plt.legend()
plt.title('Cross Validation Concordance Index, LASSO')
# In[ ]:
# TODO: Investigate which features were chosen...
pd.Series(gcv.best_estimator_.coef_[:, 0]).to_numpy().nonzero()
# Pickle the data for evaluation.
# We have:
# 
# feature mean and variance (for computing z-score)
# feature weights
# most variable genes
# In[ ]:
np.save('beataml_example2_master/model/feature_means.npy', feature_means.to_numpy())
np.save('beataml_example2_master/model/feature_stds.npy', feature_stds.to_numpy())
np.save('beataml_example2_master/model/estimator_coef.npy', gcv.best_estimator_.coef_[:, 0])
np.save('beataml_example2_master/model/most_variant_genes.npy', most_variant_genes.to_numpy())
np.save('beataml_example2_master/model/inhibitors.npy', np.array(inhibitors))
# Run the model
# We run the model with

# SYNAPSE_PROJECT_ID=<your project ID>
# docker build -t docker.synapse.org/$SYNAPSE_PROJECT_ID/sc2_model .
# docker run \
#     -v "$PWD/training/:/input/" \
#     -v "$PWD/output:/output/" \
#     docker.synapse.org/$SYNAPSE_PROJECT_ID/sc2_model

# Maybe push to Synapse.
# docker login docker.synapse.org
# docker push docker.synapse.org/$SYNAPSE_PROJECT_ID/sc2_model
# Note: you must pass the Synapse certified user test before you can push to the Synapse docker hub.

# Look at predictions vs goldstandard for training data
# Assumes predictions are in output/predictions.csv. Note that the performance on training dataset is going to be better than on the leaderboard data. Therefore, this is a good test of formatting / sanity check, but not of predictive performance.

# In[ ]:
from sksurv.metrics import concordance_index_censored

groundtruth = pd.read_csv('training/response.csv').set_index('lab_id')
predictions = pd.read_csv('beataml_example2_master/output/predictions.csv').set_index('lab_id')
data = groundtruth.join(predictions)
# data = data[data.vitalStatus == 'Dead']
cindex = concordance_index_censored(
    data.vitalStatus == 'Dead', data.overallSurvival, -data.survival)[0]
print(cindex)
# In[ ]:
import seaborn
seaborn.scatterplot(
    x='overallSurvival',
    y='survival',
    data=data,
    hue='vitalStatus',
    alpha=1)
plt.title('SC2 baseline predictor')