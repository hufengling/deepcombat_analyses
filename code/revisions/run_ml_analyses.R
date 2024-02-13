# Load libraries

library(tidyverse)
library(DeepComBat)
library(neuroCombat)
library(CovBat)

library(latex2exp)
library(DescTools)
library(GGally)
library(ggfortify)
library(ggpubr)

library(umap)

library(caret)
library(matrixStats)
library(twosamples)
library(kBET)

library(gt)
library(gtsummary)

source("/home/fengling/Documents/nnbatch/code/eval_utils_revisions.R")
source("/home/fengling/Documents/nnbatch/code/revisions/combat_from_train.R")

all_cov <- read.csv("./data/covariates.csv")
all_cov$SEX <- as.factor(all_cov$SEX)
all_cov$DIAGNOSIS <- as.factor(all_cov$DIAGNOSIS)
all_cov$Random <- as.factor(all_cov$Random)
all_cov$manufac <- as.factor(all_cov$manufac)
all_cov$Manufacturer <- as.factor(all_cov$Manufacturer)

unique_cov <- all_cov %>% 
  mutate(X.1 = X.1 + 1) %>% 
  group_by(subid) %>% 
  filter(VISIT == max(VISIT)) %>% 
  as.data.frame()
rm(all_cov)

raw <- as.matrix(read.csv("./data/raw.csv", header = F))
unique_raw <- raw[unique_cov$X.1, ]
rm(raw)

## Feature matrix list
internal_files_list <- lapply(list.files("./data/revisions/internal_harmonization", 
                                         full.names = TRUE, pattern = "*.csv"), 
                              function(i) {print(i); read_csv(i)})
internal_list <- list(raw = internal_files_list[[4]],
                      combat = internal_files_list[[1]][, -1],
                      covbat = internal_files_list[[2]][, -1],
                      deepcombat = internal_files_list[[3]][, -1]) %>% 
  lapply(as.matrix)
internal_list %>% lapply(dim)

external_files_list <- lapply(list.files("./data/revisions/external_harmonization", 
                                         full.names = TRUE, pattern = "*.csv"), 
                              function(i) {print(i); read_csv(i)})
external_list <- list(raw = internal_files_list[[4]],
                      combat = external_files_list[[1]][, -1],
                      covbat = external_files_list[[2]][, -1],
                      deepcombat = external_files_list[[3]][, -1]) %>% 
  lapply(as.matrix)
external_list %>% lapply(dim)

lambdas_files_list <- lapply(list.files("./data/revisions/internal_lambdas", 
                                        full.names = TRUE, pattern = "*.csv"), 
                             function(i) {print(i); read_csv(i)})
lambdas_list <- list(raw = internal_files_list[[4]],
                     deepcombat_00625 = lambdas_files_list[[1]],
                     deepcombat_025 = lambdas_files_list[[2]],
                     deepcombat_01 = internal_files_list[[3]][, -1],
                     deepcombat_4 = lambdas_files_list[[4]],
                     deepcombat_16 = lambdas_files_list[[3]]) %>% 
  lapply(as.matrix)
lambdas_list %>% lapply(dim)

rm(internal_files_list, external_files_list, lambdas_files_list)

## Declarations
methods_names <- c("Raw", "ComBat", "CovBat", "DeepComBat")
lambdas_names <-  c("Raw",
                    TeX(r"(\lambda = 0.00625)"),
                    TeX(r"(\lambda = 0.025)"), 
                    TeX(r"(\lambda = 0.1)"),
                    TeX(r"(\lambda = 0.4)"), 
                    TeX(r"(\lambda = 1.6)"))
lambdas_names_table <- c("Raw", 
                         "Lambda = 0.00625", 
                         "Lambda = 0.025", 
                         "Lambda = 0.1",
                         "Lambda = 0.4", 
                         "Lambda = 1.6")
methods_range <- 1:4
lambdas_range <- 1:6
my_colors <- c("#1F78B4", "#B2DF8A", "#33A02C", "#CAB2D6")
my_colors_lambdas <- c("#1F78B4", "#B2DF8A", "#33A02C", "#CAB2D6", "#FB9A99", "#E31A1C")

#ML Analyses
#Internal
print("Starting analyses")
internal_batch <- ml_cv_tester(internal_list, covariates = unique_cov,
                               outcome = "Manufacturer", 
                               k_fold = 10, repeats = 10, verbose = TRUE)
write.csv(internal_batch, "./data/revisions/ml/internal_batch.csv")
internal_sex <- ml_cv_tester(internal_list, covariates = unique_cov,
                             outcome = "SEX", 
                             k_fold = 10, repeats = 10, verbose = TRUE)
write.csv(internal_sex, "./data/revisions/ml/internal_sex.csv")
internal_diagnosis <- ml_cv_tester(internal_list, covariates = unique_cov,
                                   outcome = "DIAGNOSIS", 
                                   k_fold = 10, repeats = 10, verbose = TRUE)
write.csv(internal_diagnosis, "./data/revisions/ml/internal_diagnosis.csv")
internal_age <- ml_cv_tester(internal_list, covariates = unique_cov,
                             outcome = "AGE", 
                             k_fold = 10, repeats = 10, verbose = TRUE)
write.csv(internal_age, "./data/revisions/ml/internal_age.csv")

#External
external_batch <- ml_cv_tester(external_list, covariates = unique_cov,
                               outcome = "Manufacturer", 
                               k_fold = 10, repeats = 10, verbose = TRUE)
write.csv(external_batch, "./data/revisions/ml/external_batch.csv")
external_sex <- ml_cv_tester(external_list, covariates = unique_cov,
                             outcome = "SEX", 
                             k_fold = 10, repeats = 10, verbose = TRUE)
write.csv(external_sex, "./data/revisions/ml/external_sex.csv")
external_diagnosis <- ml_cv_tester(external_list, covariates = unique_cov,
                                   outcome = "DIAGNOSIS", 
                                   k_fold = 10, repeats = 10, verbose = TRUE)
write.csv(external_diagnosis, "./data/revisions/ml/external_diagnosis.csv")
external_age <- ml_cv_tester(external_list, covariates = unique_cov,
                             outcome = "AGE",
                             k_fold = 10, repeats = 10, verbose = TRUE)
write.csv(external_age, "./data/revisions/ml/external_age.csv")

#Lambdas
lambdas_batch <- ml_cv_tester(lambdas_list, covariates = unique_cov,
                              outcome = "Manufacturer", 
                              k_fold = 10, repeats = 10, verbose = TRUE)
write.csv(lambdas_batch, "./data/revisions/ml/lambdas_batch.csv")
lambdas_sex <- ml_cv_tester(lambdas_list, covariates = unique_cov,
                            outcome = "SEX", 
                            k_fold = 10, repeats = 10, verbose = TRUE)
write.csv(lambdas_sex, "./data/revisions/ml/lambdas_sex.csv")
lambdas_diagnosis <- ml_cv_tester(lambdas_list, covariates = unique_cov,
                                  outcome = "DIAGNOSIS", 
                                  k_fold = 10, repeats = 10, verbose = TRUE)
write.csv(lambdas_diagnosis, "./data/revisions/ml/lambdas_diagnosis.csv")
lambdas_age <- ml_cv_tester(lambdas_list, covariates = unique_cov,
                            outcome = "AGE", 
                            k_fold = 10, repeats = 10, verbose = TRUE)
write.csv(lambdas_age, "./data/revisions/ml/lambdas_age.csv")