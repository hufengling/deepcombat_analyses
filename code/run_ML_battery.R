#bsub -q taki_normal -J "ml[1-4]" -M 40000 -o ~/Documents/nnbatch/run_ml.txt bash -c "Rscript /home/fengling/Documents/nnbatch/code/run_ML_battery.R"
source("~/Documents/nnbatch/code/load_packages.R")
setwd("~/Documents/nnbatch")
#ML
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(MASS)
library(tidyverse)
library(pROC)
library(parallel)

if (Sys.getenv("LSB_JOBINDEX") == "" | Sys.getenv("LSB_JOBINDEX") == "0") {
  i <- 3
} else {
  i <- as.numeric(Sys.getenv("LSB_JOBINDEX"))
}

ml_cv_tester_individual <- function(data_list, ml_interval = NULL, covariates, outcome = "manufac",
                                    k_fold = 10, repeats = 1, verbose = FALSE, cores = 10) {
  if (is.null(ml_interval)) {
    ml_interval <- 1:length(data_list)
  }
  tmp_list <- data_list[ml_interval]
  outcome_vec <- covariates[outcome]
  names(outcome_vec) <- "outcome"
  tmp_list <- lapply(tmp_list, function(item) {
    joined_df <- cbind(as.data.frame(item), outcome = outcome_vec)
    return(joined_df)
  })
  
  if (outcome %in% c("SEX", "manufac")) { #two class prediction tasks
    tc <- trainControl(method = "repeatedcv", 
                       number = k_fold, repeats = repeats,
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE,
                       savePredictions = TRUE,
                       verboseIter = verbose,
                       sampling = "up")
  }
  
  if (outcome %in% c("DIAGNOSIS")) { #multiclass prediction
    tc <- trainControl(method = "repeatedcv", 
                       number = k_fold, repeats = repeats,
                       savePredictions = TRUE,
                       verboseIter = verbose,
                       sampling = "up")
  }
  
  if (outcome %in% c("AGE")) { #regression
    tc <- trainControl(method = "repeatedcv", 
                       number = k_fold, repeats = repeats,
                       savePredictions = TRUE,
                       verboseIter = verbose)
  }
  
  data_results <- lapply(tmp_list, function(item) {
    if (outcome == "manufac") {
      tmp_rf <- train(outcome ~ ., data = item, method = "rf",
                      trControl = tc,
                      tuneGrid = data.frame(.mtry = 20),
                      metric = "ROC")$pred %>% 
        group_by(Resample) %>%
        summarize(validation_roc = as.numeric(roc(obs ~ True)$auc)) %>% 
        cbind(ml = "rf")
      tmp_svm <- train(outcome ~ ., data = item, method = "svmRadial", 
                       trControl = tc, 
                       tuneGrid = data.frame(.sigma = 1/62, .C = 1),
                       metric = "ROC")$pred %>% 
        group_by(Resample)#%>%
        summarize(validation_roc = as.numeric(roc(obs ~ True)$auc)) %>% 
        cbind(ml = "svm")
      tmp_knn <- train(outcome ~ ., data = item, method = "knn", 
                       trControl = tc, 
                       tuneGrid = data.frame(.k = 5),
                       metric = "ROC")$pred %>% 
        group_by(Resample) %>%
        summarize(validation_roc = as.numeric(roc(obs ~ True)$auc)) %>% 
        cbind(ml = "knn")
      tmp_qda <- train(outcome ~ ., data = item, method = "qda", 
                       trControl = tc, metric = "ROC")$pred %>% 
        group_by(Resample) %>%
        summarize(validation_roc = as.numeric(roc(obs ~ True)$auc)) %>% 
        cbind(ml = "qda")
      tmp_xgb <- train(outcome ~ ., data = item, method = "xgbTree",
                       trControl = tc,
                       tuneGrid = data.frame(.nrounds = 100,
                                             .max_depth = 6,
                                             .eta = 0.1,
                                             .gamma = 0,
                                             .colsample_bytree = .5,
                                             .min_child_weight = 1,
                                             .subsample = 1),
                       metric = "ROC")$pred %>% 
        group_by(Resample) %>%
        summarize(validation_roc = as.numeric(roc(obs ~ True)$auc)) %>% 
        cbind(ml = "xgb")
      tmp_results <- rbind(tmp_rf,
                           tmp_svm,
                           tmp_knn,
                           tmp_qda,
                           tmp_xgb)
    }
    
    if (outcome == "SEX") {
      tmp_rf <- train(outcome ~ ., data = item, method = "rf",
                      trControl = tc,
                      tuneGrid = data.frame(.mtry = 20),
                      metric = "ROC")$pred %>% 
        group_by(Resample) %>%
        summarize(validation_roc = as.numeric(roc(obs ~ M)$auc)) %>% 
        cbind(ml = "rf")
      tmp_svm <- train(outcome ~ ., data = item, method = "svmRadial", 
                       trControl = tc, 
                       tuneGrid = data.frame(.sigma = 1/62, .C = 1),
                       metric = "ROC")$pred %>% 
        group_by(Resample) %>%
      summarize(validation_roc = as.numeric(roc(obs ~ M)$auc)) %>% 
        cbind(ml = "svm")
      tmp_knn <- train(outcome ~ ., data = item, method = "knn", 
                       trControl = tc, 
                       tuneGrid = data.frame(.k = 5),
                       metric = "ROC")$pred %>% 
        group_by(Resample) %>%
        summarize(validation_roc = as.numeric(roc(obs ~ M)$auc)) %>% 
        cbind(ml = "knn")
      tmp_qda <- train(outcome ~ ., data = item, method = "qda", 
                       trControl = tc, metric = "ROC")$pred %>% 
        group_by(Resample) %>%
        summarize(validation_roc = as.numeric(roc(obs ~ M)$auc)) %>% 
        cbind(ml = "qda")
      tmp_xgb <- train(outcome ~ ., data = item, method = "xgbTree",
                       trControl = tc,
                       tuneGrid = data.frame(.nrounds = 100,
                                             .max_depth = 6,
                                             .eta = 0.1,
                                             .gamma = 0,
                                             .colsample_bytree = .5,
                                             .min_child_weight = 1,
                                             .subsample = 1),
                       metric = "ROC")$pred %>% 
        group_by(Resample) %>%
        summarize(validation_roc = as.numeric(roc(obs ~ M)$auc)) %>% 
        cbind(ml = "xgb")
      tmp_results <- rbind(tmp_rf,
                           tmp_svm,
                           tmp_knn,
                           tmp_qda,
                           tmp_xgb)
    }
    
    if (outcome == "DIAGNOSIS") {
      tmp_rf <- train(outcome ~ ., data = item, method = "rf", 
                      trControl = tc, 
                      tuneGrid = data.frame(.mtry = 20))$pred %>% 
        group_by(Resample) %>% 
        summarize(validation_accuracy = mean(pred == obs)) %>% 
        cbind(ml = "rf")
      tmp_svm <- train(outcome ~ ., data = item, method = "svmRadial", 
                       trControl = tc, 
                       tuneGrid = data.frame(.sigma = 1/62, .C = 1))$pred %>% 
        group_by(Resample) %>% 
        summarize(validation_accuracy = mean(pred == obs)) %>% 
        cbind(ml = "svm")
      tmp_knn <- train(outcome ~ ., data = item, method = "knn", 
                       trControl = tc, 
                       tuneGrid = data.frame(.k = 5))$pred %>% 
        group_by(Resample) %>% 
        summarize(validation_accuracy = mean(pred == obs)) %>% 
        cbind(ml = "knn")
      tmp_qda <- train(outcome ~ ., data = item, method = "qda", 
                       trControl = tc)$pred %>% 
        group_by(Resample) %>% 
        summarize(validation_accuracy = mean(pred == obs)) %>% 
        cbind(ml = "qda")
      tmp_xgb <- train(outcome ~ ., data = item, method = "xgbTree",
                       trControl = tc, 
                       tuneGrid = data.frame(.nrounds = 100, 
                                             .max_depth = 6, 
                                             .eta = 0.1, 
                                             .gamma = 0, 
                                             .colsample_bytree = .5,
                                             .min_child_weight = 1, 
                                             .subsample = .75))$pred %>% 
        group_by(Resample) %>% 
        summarize(validation_accuracy = mean(pred == obs)) %>% 
        cbind(ml = "xgb")
      tmp_results <- rbind(tmp_rf,
                           tmp_svm,
                           tmp_knn,
                           tmp_qda,
                           tmp_xgb)
    }
    
    if (outcome == "AGE") {
      tmp_rf <- train(outcome ~ ., data = item, method = "rf", 
                      trControl = tc, 
                      tuneGrid = data.frame(.mtry = 20))$pred %>% 
        group_by(Resample) %>% 
        summarize(validation_r2 = cor(pred, obs)^2) %>% 
        cbind(ml = "rf")
      tmp_svm <- train(outcome ~ ., data = item, method = "svmRadial", 
                       trControl = tc, 
                       tuneGrid = data.frame(.sigma = 1/62, .C = 1))$pred %>% 
        group_by(Resample) %>% 
        summarize(validation_r2 = cor(pred, obs)^2) %>% 
        cbind(ml = "svm")
      tmp_knn <- train(outcome ~ ., data = item, method = "knn", 
                       trControl = tc, 
                       tuneGrid = data.frame(.k = 5))$pred %>% 
        group_by(Resample) %>% 
        summarize(validation_r2 = cor(pred, obs)^2) %>% 
        cbind(ml = "knn")
      tmp_xgb <- train(outcome ~ ., data = item, method = "xgbTree",
                       trControl = tc, 
                       tuneGrid = data.frame(.nrounds = 100, 
                                             .max_depth = 6, 
                                             .eta = 0.1, 
                                             .gamma = 0, 
                                             .colsample_bytree = .5,
                                             .min_child_weight = 1, 
                                             .subsample = .75))$pred %>% 
        group_by(Resample) %>% 
        summarize(validation_r2 = cor(pred, obs)^2) %>% 
        cbind(ml = "xgb")
      tmp_results <- rbind(tmp_rf,
                           tmp_svm,
                           tmp_knn,
                           tmp_xgb)
    }
    return(tmp_results)
  })
  
  data_results <- lapply(ml_interval, function(i) {
    return(cbind(data_results[[i]], 
                 names(tmp_list)[i]))
  })
  
  bind_rows(data_results)
}

all_cov <- read.csv("/home/fengling/Documents/nnbatch/data/covariates.csv")
all_cov$SEX <- as.factor(all_cov$SEX)
all_cov$DIAGNOSIS <- as.factor(all_cov$DIAGNOSIS)
all_cov$Random <- as.factor(all_cov$Random)
all_cov$manufac <- as.factor(all_cov$manufac)

unique_cov <- all_cov %>% 
  group_by(subid) %>% 
  filter(VISIT == max(VISIT)) %>% 
  as.data.frame()

files_list <- lapply(list.files("/home/fengling/Documents/nnbatch/data/for_paper", 
                                full.names = TRUE, pattern = "*.csv"), 
                     read_csv)
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
if (i == 1) {
  manufac_results <- ml_cv_tester_individual(unique_data_list, 1:10, unique_cov, 
                                             outcome = "manufac",
                                             k_fold = 10, repeats = 10, verbose = T)
  write.csv(manufac_results, "/home/fengling/Documents/nnbatch/data/for_paper/ml/manufac_100.csv")
}
if (i == 2) {
  age_results <- ml_cv_tester_individual(unique_data_list, 1:10, unique_cov, 
                                         outcome = "AGE",
                                         k_fold = 10, repeats = 10, verbose = T)
  write.csv(age_results, "/home/fengling/Documents/nnbatch/data/for_paper/ml/age_100.csv")
}
if (i == 3) {
  sex_results <- ml_cv_tester_individual(unique_data_list, 1:10, unique_cov, 
                                         outcome = "SEX",
                                         k_fold = 10, repeats = 10, verbose = T)
  write.csv(sex_results, "/home/fengling/Documents/nnbatch/data/for_paper/ml/sex_100.csv")
}
if (i == 4) {
  diagnosis_results <- ml_cv_tester_individual(unique_data_list, 1:10, unique_cov, 
                                               outcome = "DIAGNOSIS",
                                               k_fold = 10, repeats = 10, verbose = T)
  write.csv(diagnosis_results, "/home/fengling/Documents/nnbatch/data/for_paper/ml/diagnosis_100.csv")
}
