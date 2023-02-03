source("~/Documents/nnbatch/code/load_packages.R")
setwd("~/Documents/nnbatch")
#ML
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(MASS)
library(tidyverse)

all_cov <- read.csv("./data/covariates.csv")
all_cov$SEX <- as.factor(all_cov$SEX)
all_cov$DIAGNOSIS <- as.factor(all_cov$DIAGNOSIS)
all_cov$Random <- as.factor(all_cov$Random)
all_cov$manufac <- as.factor(all_cov$manufac)

unique_cov <- all_cov %>% 
  group_by(subid) %>% 
  filter(VISIT == max(VISIT)) %>% 
  as.data.frame()

files_list <- lapply(list.files("./data/for_paper", full.names = TRUE), read_csv)
unique_data_list <- list(unique_raw = files_list[[8]],
                         unique_combat = files_list[[4]],
                         unique_covbat = files_list[[5]],
                         unique_cvae = files_list[[6]],
                         unique_gcvae = files_list[[7]],
                         optimal = files_list[[2]],
                         small = files_list[[3]],
                         waysmall = files_list[[10]],
                         big = files_list[[1]],
                         waybig = files_list[[9]])
unique_data_list <- lapply(unique_data_list, as.matrix)

set.seed(1)
manufac_results <- ml_cv_tester(unique_data_list, 1:10, unique_cov, 
                                outcome = "manufac",
                                k_fold = 10, repeats = 5, verbose = T)
age_results <- ml_cv_tester(unique_data_list, 1:10, unique_cov, 
                            outcome = "AGE",
                            k_fold = 10, repeats = 5, verbose = T)
sex_results <- ml_cv_tester(unique_data_list, 1:10, unique_cov, 
                            outcome = "SEX",
                            k_fold = 10, repeats = 5, verbose = T)
diagnosis_results <- ml_cv_tester(unique_data_list, 1:10, unique_cov, 
                                  outcome = "DIAGNOSIS",
                                  k_fold = 10, repeats = 5, verbose = T)

write.csv(manufac_results, "./data/for_paper/manufac.csv")
write.csv(age_results, "./data/for_paper/age.csv")
write.csv(sex_results, "./data/for_paper/sex.csv")
write.csv(diagnosis_results, "./data/for_paper/diagnosis.csv")
