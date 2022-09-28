umap_plotter <- function(data_list, covariates, n_neighbors, n_epochs) {
  umap_plot_list <- vector("list", length(data_list))
  for (i in 1:length(data_list)) {
    data <- data_list[[i]]
    umap_custom <- umap.defaults
    umap_custom$n_neighbors <- n_neighbors
    umap_custom$n_epochs <- n_epochs
    data_umap <- umap(data, config = umap_custom)
    layout_cov <- cbind(data_umap$layout, covariates)
    names(layout_cov)[1:2] <- c(1, 2)
    
    subid <- ggplot(layout_cov) + geom_point(aes(`1`, `2`, color = subid, shape = ".")) +
      theme(legend.position = "none")
    age <- ggplot(layout_cov) + geom_point(aes(`1`, `2`, color = AGE, shape = "."))
    sex <- ggplot(layout_cov) + geom_point(aes(`1`, `2`, color = SEX, shape = "."))
    diagnosis <- ggplot(layout_cov) + geom_point(aes(`1`, `2`, color = DIAGNOSIS, shape = "."))
    random <- ggplot(layout_cov) + geom_point(aes(`1`, `2`, color = Random, shape = ".")) +
      theme(legend.position = "none")
    manufac <- ggplot(layout_cov) + geom_point(aes(`1`, `2`, color = manufac, shape = ".")) +
      theme(legend.position = "none")
    
    umap_list <- list(subid, age, sex, diagnosis, random, manufac)
    umap_plot_list[[i]] <- ggarrange(plotlist = umap_list, ncol = 3, nrow = 2, 
                                     labels = c(names(data_list)[i], "AGE", "SEX", 
                                                "DIAGNOSIS", "Random", "manufac"))
    gc()
  }
  
  return(umap_plot_list)
}

anova_tester <- function(data_list, covariates) {
  anova_list <- vector("list", length(data_list))
  for (i in 1:length(data_list)) {
    mod_list <- summary(lm(as.matrix(data_list[[i]]) ~ AGE + SEX + DIAGNOSIS + Random + manufac, 
                           data = covariates))
    t_stats <- sapply(mod_list, function(mod) {
      mod$coefficients[-1, 3]
    }, simplify = T)
    
    mean_abs_t <- (t_stats) %*% rep(1 / ncol(t_stats), ncol(t_stats))
    mean_sq_t <- diag(t_stats %*% t(t_stats) / ncol(t_stats))
    
    anova_list[[i]] <- cbind(mean_abs_t, mean_sq_t)
  }
  
  abs_df <- data.frame(matrix(rep(0, length(anova_list) * nrow(anova_list[[1]])), 
                              nrow = length(anova_list)))
  names(abs_df) <- rownames(anova_list[[1]])
  mean_sq_t_df <- abs_df
  for (i in 1:length(anova_list)) {
    abs_df[i, ] <- signif(anova_list[[i]][, 1], 4)
    mean_sq_t_df[i, ] <- signif(anova_list[[i]][, 2], 4)
  }
  row.names(abs_df) <- names(data_list)
  row.names(mean_sq_t_df) <- names(data_list)
  
  return(list(abs_df, mean_sq_t_df))
}

rf_splitter <- function(data_list, covariates) {
  test_ids <- sample(unique(covariates$subid), floor(length(unique(covariates$subid)) / 4), replace = F)
  train_ids <- unique(covariates$subid)[!(unique(covariates$subid) %in% test_ids)]
  
  test_list <- lapply(data_list, function(data) {
    data[which(covariates$subid %in% test_ids), ]
  })
  train_list <- lapply(data_list, function(data) {
    data[which(covariates$subid %in% train_ids), ]
  })
  
  test_covariates <- covariates[which(covariates$subid %in% test_ids), ]
  train_covariates <- covariates[which(covariates$subid %in% train_ids), ]
  
  return(list(train_list = train_list,
              test_list = test_list,
              train_covariates = train_covariates,
              test_covariates = test_covariates))
}

rf_tester <- function(rf_list, 
                      ml_type = c("rf", "svm"), interval,
                      kernel = "radial") {
  ml_type <- match.arg(ml_type, c("rf", "svm"))
  train_list <- rf_list$train_list
  train_covariates <- rf_list$train_covariates
  test_list <- rf_list$test_list
  test_covariates <- rf_list$test_covariates
  test_error_list <- vector("list", length(train_list))
  
  for (i in 1:length(train_list)) {
    if (ml_type == "rf") {
      cat(paste0("Running RF on item ", i, "\n"))
      age_rf <- randomForest(x = train_list[[i]], y = train_covariates$AGE,
                             xtest = test_list[[i]], ytest = test_covariates$AGE,
                             ntree = 100)
      sex_rf <- randomForest(x = train_list[[i]], y = train_covariates$SEX,
                             xtest = test_list[[i]], ytest = test_covariates$SEX,
                             ntree = 100)
      diag_rf <- randomForest(x = train_list[[i]], y = train_covariates$DIAGNOSIS,
                              xtest = test_list[[i]], ytest = test_covariates$DIAGNOSIS,
                              ntree = 100)
      random_rf <- randomForest(x = train_list[[i]], y = train_covariates$Random,
                                xtest = test_list[[i]], ytest = test_covariates$Random,
                                ntree = 100)
      manufac_rf <- randomForest(x = train_list[[i]], y = train_covariates$manufac,
                                 xtest = test_list[[i]], ytest = test_covariates$manufac,
                                 ntree = 1000)
      age_mse <- crossprod(age_rf$test$predicted - test_covariates$AGE) / length(test_covariates$AGE)
      rf_list <- list(sex_rf, diag_rf, random_rf, manufac_rf)
      tmp_test_error <- lapply(rf_list, function(rf) {
        conf_mat <- rf$test$confusion
        return(1 - sum(diag(conf_mat)) / sum(conf_mat[, 1:(dim(conf_mat)[2] - 1)]))
      }) %>% unlist()
      test_error_list[[i]] <- c(age_mse, tmp_test_error)
      gc()
    }
    if (ml_type == "svm") {
      cat(paste0("Running SVM on item ", i, "\n"))
      age_svm <- svm(x = train_list[[i]], y = train_covariates$AGE, kernel = kernel)
      sex_svm <- svm(x = train_list[[i]], y = train_covariates$SEX, kernel = kernel) %>% 
        predict(test_list[[i]]) %>% 
        table(test_covariates$SEX)
      diag_svm <- svm(x = train_list[[i]], y = train_covariates$DIAGNOSIS, kernel = kernel) %>% 
        predict(test_list[[i]]) %>% 
        table(test_covariates$DIAGNOSIS)
      random_svm <- svm(x = train_list[[i]], y = train_covariates$Random, kernel = kernel) %>% 
        predict(test_list[[i]]) %>% 
        table(test_covariates$Random)
      manufac_svm <- svm(x = train_list[[i]], y = train_covariates$manufac, kernel = kernel) %>% 
        predict(test_list[[i]]) %>% 
        table(test_covariates$manufac)
      age_mse <- crossprod(predict(age_svm, test_list[[i]]) - test_covariates$AGE) / length(test_covariates$AGE)
      confusion_list <- list(sex_svm, diag_svm, random_svm, manufac_svm)
      tmp_test_error <- lapply(confusion_list, function(confusion) {
        return(1 - sum(diag(confusion)) / sum(confusion))
      }) %>% unlist()
      test_error_list[[i]] <- c(age_mse, tmp_test_error)
      gc()
    }
  }
  print("done")
  test_error_df <- test_error_list %>% unlist() %>% 
    matrix(nrow = length(test_error_list), byrow = T) %>% 
    as.data.frame %>% 
    rename(AGE = V1, SEX = V2, DIAGNOSIS = V3, Random = V4, manufac = V5)
  row.names(test_error_df) <- names(data_list[interval])
  return(test_error_df)
}

subject_silhouette_tester <- function(data_list, covariates) {
  cluster_codes <- covariates$subid %>% as.numeric()
  mean_sil_scores <- rep(0, length(data_list))
  for (i in 1:length(data_list)) {
    pca_coords <- PCA(data_list[[i]], scale.unit = T, ncp = 30, graph = F)$ind$coord
    pca_dist <- dist(pca_coords, method = "manhattan")
    subject_sil_score <- silhouette(x = cluster_codes, 
                                    dist = pca_dist)
    mean_sil_scores[[i]] <- mean(subject_sil_score[, 3])
  }
  
  return(mean_sil_scores)
}

cov_generator <- function(data_list, covariates) {
  cov_list <- vector("list", length(data_list))
  for (i in 1:length(data_list)) {
    data <- data_list[[i]]
    siemens_data <- data[which(covariates$manufac == "True"), ]
    other_data <- data[which(covariates$manufac == "False"), ]
    all_cov <- cov(data)
    siemens_cov <- cov(siemens_data)
    other_cov <- cov(other_data)
    diff_cov <- siemens_cov - other_cov
    
    cov_list[[i]] <- list(all_cov, siemens_cov, other_cov, diff_cov)
  }
  
  return(cov_list)
}

cov_plotter <- function(cov_list) {
  ordered_cormat <- reorder_cormat(cov_list[[1]][[1]])
  hclust_ordering <- rownames(ordered_cormat) %>% substring(2) %>% as.numeric()
  
  cov_plot_list <- vector("list", length(cov_list))
  for (i in 1:length(cov_list)) {
    tmp_cov_plot_list <- vector("list", length(cov_list[[i]]))
    for (j in 1:length(cov_list[[i]])) {
      tmp_ordered_cov <- cov_list[[i]][[j]][hclust_ordering, hclust_ordering]
      tmp_cov_plot_list[[j]] <- ggcorrplot(abs(tmp_ordered_cov)) +
        scale_fill_gradient2(low = "white", high = "red", 
                             breaks=c(0, 1), limit=c(0, 1))
    }
    cov_plot_list[[i]] <- ggarrange(plotlist = tmp_cov_plot_list, 
                                    ncol = 2, nrow = 2, 
                                    labels = c("Overall", "Siemens", 
                                               "GE/Phillips", "Difference"), 
                                    legend = "none")
  }
  
  return(cov_plot_list)
}

get_n_max <- function(data_col, n_max, get_value = F) {
  index_list <- rep(0, n_max)
  value_list <- rep(0, n_max)
  for (i in 1:n_max) {
    tmp_max <- which.max(data_col)
    value_list[i] <- data_col[tmp_max]
    index_list[i] <- tmp_max
    data_col[tmp_max] <- NA
  }
  if (get_value)
    return(value_list)
  else
    return(index_list)
}

get_pre_post <- function(data_list, similarity_type = c("correlation", "manhattan", "euclidean"), covariates, n_max) {
  max_cor_matrix_list <- vector("list", length(data_list))
  for (i in 1:(length(data_list))) {
    if (similarity_type == "correlation")
      raw_vs_corrected_cor <- cor(t(data_list[[1]]), t(data_list[[i]])) %>% as.data.frame()
    else if (similarity_type == "manhattan") 
      raw_vs_corrected_cor <- -cdist(data_list[[1]], data_list[[i]], metric = "manhattan") %>% as.data.frame()
    else if (similarity_type == "euclidean")
      raw_vs_corrected_cor <- -cdist(data_list[[1]], data_list[[i]], metric = "euclidean") %>% as.data.frame()
    max_index <- lapply(raw_vs_corrected_cor, function(data_col) {get_n_max(data_col, n_max, get_value = F)})
    max_matrix <- matrix(unlist(max_index), ncol = n_max, byrow = T)
    
    # for each subject, mark index of correlation between raw and corrected self as -1
    id_num_matrix <- matrix(rep(1:nrow(max_matrix), n_max), ncol = n_max, byrow = F)
    self_index <- which(max_matrix - id_num_matrix == 0)
    
    # label other indices with subid
    rownames(max_matrix) <- covariates$subid
    for (j in 1:length(max_matrix)) {
      max_matrix[j] <- as.numeric(as.character(covariates$subid[max_matrix[j]]))
    }
    
    # label corresponding longitudinal scans with -2
    for (j in 1:nrow(max_matrix)) {
      as.numeric(rownames(max_matrix)[j])
      max_matrix[j, max_matrix[j, ] == as.numeric(rownames(max_matrix)[j])] <- -2
    }
    max_matrix[self_index] <- -1 # last step of marking raw vs corrected self as -1
    max_cor_matrix_list[[i]] <- max_matrix %>% as.data.frame()
  }
  return(max_cor_matrix_list)
}

pre_post_analyzer <- function(pre_post_matrix_list, covariates) {
  count_neg_1s_2s <- function(matrix) {
    n = nrow(matrix)
    num_1s <- sum(matrix == -1)
    num_2s <- sum(matrix == -2) / (n - 1)
    return(c(n, num_1s, num_2s))
  }
  
  summary_mat_list <- vector("list")
  unique_subid <- unique(covariates$subid)
  for (i in 1:length(pre_post_matrix_list)) {
    tmp_df <- cbind(pre_post_matrix_list[[i]], subid = covariates$subid)
    
    summary_mat <- matrix(rep(0, length(unique_subid) * 3), ncol = 3)
    for (j in 1:length(unique_subid))
      summary_mat[j, ] <- count_neg_1s_2s(tmp_df[tmp_df$subid == unique_subid[j], ])
    summary_mat_list[[i]] <- summary_mat
  }
  
  return(summary_mat_list)
}