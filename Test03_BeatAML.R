# ctrl+shift+F10

PATH     = sprintf("%s\\Google Drive\\Study\\ECE 5332-009 - Topics in EE, Data Science\\BeatAML\\", Sys.getenv("HOMEPATH"))
DIRNAME  = c("Training\\", "Leaderboard\\")
FILENAME = c("rnaseq.csv", "dnaseq.csv", "clinical_numerical.csv", "clinical_categorical.csv", 
             "clinical_categorical_legend.csv", "aucs.csv", "response.csv")
setwd(PATH)

library(Rfast)
library(stringr)
library(IntegratedMRF)
library(stats)


### Read data...
## Training...
RNA = read.csv(paste0(PATH, DIRNAME[1], FILENAME[1]), header = TRUE, as.is = TRUE)
AUC = read.csv(paste0(PATH, DIRNAME[1], FILENAME[6]), header = TRUE, as.is = TRUE)

## Leaderboard...
RNA_LB = read.csv(paste0(PATH, DIRNAME[2], FILENAME[1]), header = TRUE, as.is = TRUE)
AUC_LB = read.csv(paste0(PATH, DIRNAME[2], FILENAME[6]), header = TRUE, as.is = TRUE)

## Processing...
all(unique(AUC$inhibitor)    == unique(AUC_LB$inhibitor))         ## Check if same drugs
all(RNA[c("Gene", "Symbol")] == RNA_LB[c("Gene", "Symbol")])      ## Check if same genes

lab_id_list = list(TR = colnames(RNA)[3:ncol(RNA)], LB = colnames(RNA_LB)[3:ncol(RNA_LB)])
gene_list = RNA[c("Gene", "Symbol")];   drug_list = unique(AUC$inhibitor)

rownames(RNA)    = gene_list$Gene;    RNA    = RNA[, 3:ncol(RNA)]
rownames(RNA_LB) = gene_list$Gene;    RNA_LB = RNA_LB[, 3:ncol(RNA_LB)]
# dim(RNA); dim(RNA_LB)

var_idx = rowVars(as.matrix(RNA)) > 0.1            ## Filter genes w/ low variability
RNA_filt = RNA[var_idx, ];        RNA_LB_filt = RNA_LB[low_var_idx, ]


### Process cluster info...
FILENAME.2   = "drug_clusters_max_clust.csv"
cluster.info = read.csv(paste0(PATH, FILENAME.2), header = TRUE, row.names = 1, as.is = TRUE)
colnames(cluster.info) = str_replace(colnames(cluster.info), pattern = "X", replacement = "N")
cls_sz = 9;   col_idx = grepl(colnames(cluster.info), pattern = as.character(cls_sz))

drugs_c1 = rownames(cluster.info)[cluster.info[, col_idx] == 2]
print(sprintf("# drugs in this cluster = %d", length(drugs_c1)))

# AUC_c1 = AUC[AUC$inhibitor == drugs_c1, ]

## Training...
AUC_c1 = lapply(drugs_c1, function(d) AUC[AUC$inhibitor == d, ]);         names(AUC_c1)    = drugs_c1
lab_ids_common = Reduce(lapply(drugs_c1, function(d) AUC_c1[[d]]$lab_id), f = intersect)
lab_ids_idx = sapply(drugs_c1, function(d) d = match(AUC_c1[[d]]$lab_id, table = lab_ids_common, nomatch = FALSE))

## Leaderboard...
AUC_LB_c1 = lapply(drugs_c1, function(d) AUC_LB[AUC_LB$inhibitor == d, ]);   names(AUC_LB_c1) = drugs_c1
lab_ids_union_LB = Reduce(lapply(drugs_c1, function(d) AUC_LB_c1[[d]]$lab_id), f = union)

## Make datasets...
X_train = t(RNA_filt[, str_replace(paste0("X", lab_ids_common), pattern = "-", replacement = ".")])
y_train = sapply(drugs_c1, function(d) AUC_c1[[d]]$auc[lab_ids_idx[[d]]])
rownames(X_train) = lab_ids_common;     rownames(y_train) = lab_ids_common
# dim(X_train);   dim(y_train)

X_test = t(RNA_LB_filt[, str_replace(paste0("X", lab_ids_union_LB), pattern = "-", replacement = ".")])
rownames(X_test) = lab_ids_union_LB
# dim(X_train);    dim(X_test);     all(colnames(X_train) == colnames(X_test))

y_pred = build_forest_predict(X_train, y_train, n_tree = 100, m_feature = 10000, min_leaf = 5, X_test)
dimnames(y_pred) = list(rownames(X_test), drugs_c1)

y_test_LB = lapply(drugs_c1, function(d) AUC_LB_c1[[d]]$auc)
y_pred_LB = lapply(drugs_c1, function(d) y_pred[rownames(y_pred) %in% AUC_LB_c1[[d]]$lab_id, d])
names(y_test_LB) = drugs_c1;     names(y_pred_LB) = drugs_c1

cor_c1 = sapply(c("pearson", "spearman"), 
                function(m) sapply(drugs_c1, function(d) cor(y_test_LB[[d]], y_pred_LB[[d]], method = m)))
mean_cor_c1 = colmeans(cor_c1);   names(mean_cor_c1) = colnames(cor_c1);    print(mean_cor_c1)
