umap_plotter <- function(data_list, covariates, plot_names, n_neighbors, n_epochs) {
  if (length(data_list) != 6) {
    stop("Length of data_list must be 6")
  }
  umap_plot_list <- vector("list", length(data_list))
  for (i in 1:length(data_list)) {
    data <- data_list[[i]]
    umap_custom <- umap.defaults
    umap_custom$n_neighbors <- n_neighbors
    umap_custom$n_epochs <- n_epochs
    data_umap <- umap(data, config = umap_custom)
    layout_cov <- cbind(data_umap$layout, covariates)
    names(layout_cov)[1:2] <- c(1, 2)
    
    # subid <- ggplot(layout_cov) + geom_point(aes(`1`, `2`, color = subid, shape = ".")) +
    #   theme(legend.position = "none")
    # age <- ggplot(layout_cov) + geom_point(aes(`1`, `2`, color = AGE, shape = "."))
    # sex <- ggplot(layout_cov) + geom_point(aes(`1`, `2`, color = SEX, shape = "."))
    # diagnosis <- ggplot(layout_cov) + geom_point(aes(`1`, `2`, color = DIAGNOSIS, shape = "."))
    # random <- ggplot(layout_cov) + geom_point(aes(`1`, `2`, color = Random, shape = ".")) +
    #   theme(legend.position = "none")
    umap_plot_list[[i]] <- ggplot(layout_cov) + 
      geom_point(aes(`1`, `2`, color = manufac, shape = ".")) +
      xlab("UMAP 1") +
      ylab("UMAP 2") +
      labs(title = plot_names[i]) +
      theme_classic() +
      theme(legend.position = "none",
            plot.title = element_text(hjust = 0.5))
    gc()
  }
  umap_plot_arrange <- ggarrange(plotlist = umap_plot_list, 
                                 ncol = 3, nrow = 2)
  
  return(umap_plot_arrange)
}

pca_plotter <- function(data_list, covariates, plot_names) {
  if (length(data_list) != 6) {
    stop("Length of data_list must be 6")
  }
  pca_plot_list <- vector("list", length(data_list))
  for (i in 1:length(data_list)) {
    tmp_pca <- prcomp(data_list[[i]], scale. = TRUE)
    pca_plot_list[[i]] <- autoplot(tmp_pca, data = covariates, colour = "manufac", 
                                   frame = TRUE, frame.type = 'norm') + 
      theme_classic() + 
      labs(title = plot_names[i]) +
      theme(legend.position = "none",
            plot.title = element_text(hjust = 0.5))
  }
  pca_plot_arrange <- ggarrange(plotlist = pca_plot_list, 
                                ncol = 3, nrow = 2)
  return(pca_plot_arrange)
}

cor_plotter <- function(data_list, covariates, plot_names) {
  if (length(data_list) != 6) {
    stop("Length of data_list must be 6")
  }
  ordered_cormat <- reorder_cormat(cor(data_list[[1]]))
  hclust_ordering <- rownames(ordered_cormat) %>% 
    substring(2) %>% 
    as.numeric()
  
  siemens_cor_list <- vector("list", length(data_list))
  other_cor_list <- vector("list", length(data_list))
  diff_cor_list <- vector("list", length(data_list))
  
  for (i in 1:length(data_list)) {
    # Correlation plots for Siemens
    tmp_siemens <- cor(data_list[[i]][covariates$manufac == "True", ])[hclust_ordering, hclust_ordering]
    rownames(tmp_siemens) <- NULL
    colnames(tmp_siemens) <- NULL
    siemens_cor_list[[i]] <- autoplot(tmp_siemens) + 
      theme(axis.text = element_blank(), 
            axis.ticks = element_blank(), 
            legend.position = "none",
            plot.title = element_text(hjust = 0.5)) + 
      labs(x = NULL, y = NULL, title = plot_names[i]) +
      scale_fill_continuous(limit = c(0, 1), 
                            high = "mediumblue", 
                            low = "white")
    
    # Correlation plots for Other
    tmp_other <- cor(data_list[[i]][covariates$manufac == "False", ])[hclust_ordering, hclust_ordering]
    rownames(tmp_other) <- NULL
    colnames(tmp_other) <- NULL
    other_cor_list[[i]] <- autoplot(tmp_other) + 
      theme(axis.text = element_blank(), 
            axis.ticks = element_blank(), 
            legend.position = "none") + 
      labs(x = NULL, y = NULL, title = "") +
      scale_fill_continuous(limit = c(0, 1), 
                            high = "mediumblue", 
                            low = "white")
    
    diff_cor_list[[i]] <- autoplot(abs(tmp_siemens - tmp_other)) + 
      theme(axis.text = element_blank(), 
            axis.ticks = element_blank(), 
            legend.position = "none") + 
      labs(x = NULL, y = NULL, title = "") +
      scale_fill_continuous(limit = c(0, 1), 
                            high = "mediumblue", 
                            low = "white")
  }
  
  all_cor_list <- siemens_cor_list %>%
    append(other_cor_list) %>% 
    append(diff_cor_list)
  
  cor_plots <- ggarrange(plotlist = all_cor_list,
                         ncol = 6, nrow = 3)
  cor_plots
}

feature_plotter <- function(data_list, covariates, plot_names, col_num = NULL) {
  if (length(data_list) != 6) {
    stop("Length of data_list must be 6")
  }
  if (is.null(col_num)) {
    col_num <- sample(1:62, 1)
    cat(paste0("Column number is: ", col_num))
  }
  feature_plot_list <- vector("list", length(data_list))
  for (i in 1:length(data_list)) {
    tmp_df <- as.data.frame(data_list[[i]][, col_num]) %>% cbind(covariates$manufac)
    names(tmp_df) <- c("feature", "manufac")
    tmp_df$manufac <- as.factor(tmp_df$manufac)
    feature_plot_list[[i]] <- ggplot(tmp_df) + 
      geom_density(aes(feature, fill = manufac), alpha = 0.5) +
      theme_classic() +
      theme(legend.position = "none",
            plot.title = element_text(hjust = 0.5)) +
      labs(x = "", y = "", title = plot_names[i])
  }
  
  ggarrange(plotlist = feature_plot_list, nrow = 2, ncol = 3)
}

stats_tester <- function(data_list, covariates) {
  p_summary <- as.data.frame(matrix(rep(0, 21 * length(data_list)), nrow = length(data_list)))
  manova_df <- as.data.frame(matrix(rep(0, 9 * length(data_list)), nrow = length(data_list)))
  ad_df <- as.data.frame(matrix(rep(0, 5 * length(data_list)), nrow = length(data_list)))
  
  for (i in 1:length(data_list)) {
    # Univariate linear regression
    mod_mat <- summary(lm(as.matrix(data_list[[i]]) ~ AGE + SEX + DIAGNOSIS + manufac, 
                          data = covariates))
    p_vals <- sapply(mod_mat, function(mod) {
      mod$coefficients[-1, 4]
    }, simplify = T)
    
    mean_p <- rowMeans(p_vals)
    sd_p <- rowSds(p_vals)
    mean_log_p <- rowMeans(log(p_vals, base = 10), na.rm = T)
    sd_log_p <- rowSds(log(p_vals, base = 10), na.rm = T)
    p_summary[i, ] <- c(as.data.frame(names(data_list)[[i]]), 
                        mean_p, sd_p, 
                        mean_log_p, sd_log_p)
    
    # Multivariate MANOVA
    tmp_manova <- summary(manova(as.matrix(data_list[[i]]) ~ AGE + SEX + DIAGNOSIS + manufac, 
                                 data = covariates))$stats
    manova_df[i, ] <- c(as.data.frame(names(data_list)[[i]]), 
                        tmp_manova[-5, 6], 
                        log(tmp_manova[-5, 6], base = 10))
    
    # Univariate Anderson-Darling
    ad_a <- data_list[[i]][covariates$manufac == "True", ]
    ad_b <- data_list[[i]][covariates$manufac == "False", ]
    tmp_ad_pvals <- rep(0, ncol(data_list[[i]]))
    for (j in 1:ncol(data_list[[i]])) {
      tmp_ad_pvals[j] <- ad_test(ad_a[, j], ad_b[, j], keep.boots = F)[2]
    }
    ad_df[i, ] <- c(as.data.frame(names(data_list)[[i]]), 
                    mean(tmp_ad_pvals), 
                    sd(tmp_ad_pvals), 
                    mean(log(tmp_ad_pvals, base = 10), na.rm = T), 
                    sd(log(tmp_ad_pvals, base = 10), na.rm = T))
  }
  colnames(p_summary) <- c("dataset", names(mean_p), paste0(names(mean_p), "_SD"),
                           paste0(names(mean_p), "_log"), paste0(names(mean_p), "_log_SD"))
  colnames(manova_df) <- c("dataset", names(tmp_manova[-5, 6]), paste0(names(tmp_manova[-5, 6]), "_log"))
  colnames(ad_df) <- c("dataset", "pval", "pval_SD", "pval_log", "pval_log_SD")
  
  
  return(list("lr" = p_summary, "manova" = manova_df, "ad" = ad_df))
}

ml_cv_tester <- function(data_list, ml_interval = NULL, covariates, outcome = "manufac",
                         k_fold = 10, repeats = 1, verbose = FALSE) {
  if (is.null(ml_interval)) {
    ml_interval <- 1:length(data_list)
  }
  tmp_list <- data_list[ml_interval]
  outcome_vec <- covariates[outcome]
  names(outcome_vec) <- "outcome"
  tmp_list <- lapply(tmp_list, function(item) {
    cbind(as.data.frame(item), outcome = outcome_vec)
  })
  
  if (outcome %in% c("SEX", "manufac")) { #two class prediction tasks
    tc <- trainControl(method = "repeatedcv", number = k_fold, repeats = repeats,
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE,
                       savePredictions = TRUE, 
                       verboseIter = verbose)
  }
  
  if (outcome %in% c("DIAGNOSIS", "AGE")) { #regression/multiclass prediction
    tc <- trainControl(method = "repeatedcv", number = k_fold, repeats = repeats,
                       verboseIter = verbose)
  }
  
  data_results <- vector("list", length(tmp_list))
  names(data_results) <- names(tmp_list)
  
  for (i in 1:length(tmp_list)) {
    if (verbose) {
      cat(paste0("Correcting dataset ", i))
    }
    item <- tmp_list[[i]]
    if (outcome %in% c("SEX", "manufac")) {
      tmp_rf <- train(outcome ~ ., data = item, method = "rf", 
                      trControl = tc, tuneGrid = data.frame(.mtry = 20), 
                      metric = "ROC")$results
      tmp_svm <- train(outcome ~ ., data = item, method = "svmRadial", 
                       trControl = tc, tuneGrid = data.frame(.sigma = 1/62, .C = 1),
                       metric = "ROC")$results
      tmp_knn <- train(outcome ~ ., data = item, method = "knn", 
                       trControl = tc, tuneGrid = data.frame(.k = 5),
                       metric = "ROC")$results
      tmp_lda <- train(outcome ~ ., data = item, method = "lda", 
                       trControl = tc, metric = "ROC")$results
      tmp_qda <- train(outcome ~ ., data = item, method = "qda", 
                       trControl = tc, metric = "ROC")$results
      tmp_logitboost <- train(outcome ~ ., data = item, method = "LogitBoost",
                              trControl = tc, tuneGrid = data.frame(.nIter = 100),
                              metric = "ROC")$results
      tmp_glm <- train(outcome ~ ., data = item, method = "glm",
                       trControl = tc, metric = "ROC")$results
      tmp_nn <- train(outcome ~ ., data = item, method = "nnet",
                      trControl = tc, tuneGrid = data.frame(.size = 8, .decay = 0),
                      metric = "ROC")$results
      tmp_results <- rbind(tmp_rf[(length(tmp_rf) - 5):length(tmp_rf)],
                           tmp_svm[(length(tmp_svm) - 5):length(tmp_svm)],
                           tmp_knn[(length(tmp_knn) - 5):length(tmp_knn)],
                           tmp_lda[(length(tmp_lda) - 5):length(tmp_lda)],
                           tmp_qda[(length(tmp_qda) - 5):length(tmp_qda)],
                           tmp_logitboost[(length(tmp_logitboost) - 5):length(tmp_logitboost)],
                           tmp_glm[(length(tmp_glm) - 5):length(tmp_glm)],
                           tmp_nn[(length(tmp_nn) - 5):length(tmp_nn)])
      names_vec <- c("rf", "svmRadial", "knn", "lda", "qda", "logitboost", "glm", "nnet")
    }
    
    if (outcome == "DIAGNOSIS") {
      tmp_rf <- train(outcome ~ ., data = item, method = "rf", 
                      trControl = tc, tuneGrid = data.frame(.mtry = 20))$results
      tmp_svm <- train(outcome ~ ., data = item, method = "svmRadial", 
                       trControl = tc, tuneGrid = data.frame(.sigma = 1/62, .C = 1))$results
      tmp_knn <- train(outcome ~ ., data = item, method = "knn", 
                       trControl = tc, tuneGrid = data.frame(.k = 5))$results
      tmp_lda <- train(outcome ~ ., data = item, method = "lda", 
                       trControl = tc)$results
      tmp_qda <- train(outcome ~ ., data = item, method = "qda", 
                       trControl = tc)$results
      tmp_logitboost <- train(outcome ~ ., data = item, method = "LogitBoost",
                              trControl = tc, tuneGrid = data.frame(.nIter = 100))$results
      tmp_nn <- train(outcome ~ ., data = item, method = "nnet",
                      trControl = tc, tuneGrid = data.frame(.size = 8, .decay = 0))$results
      tmp_results <- rbind(tmp_rf[(length(tmp_rf) - 3):length(tmp_rf)],
                           tmp_svm[(length(tmp_svm) - 3):length(tmp_svm)],
                           tmp_knn[(length(tmp_knn) - 3):length(tmp_knn)],
                           tmp_lda[(length(tmp_lda) - 3):length(tmp_lda)],
                           tmp_qda[(length(tmp_qda) - 3):length(tmp_qda)],
                           tmp_logitboost[(length(tmp_logitboost) - 3):length(tmp_logitboost)],
                           tmp_nn[(length(tmp_nn) - 3):length(tmp_nn)])
      names_vec <- c("rf", "svmRadial", "knn", "lda", "qda", "logitboost", "nnet")
    }
    
    if (outcome == "AGE") {
      tmp_rf <- train(outcome ~ ., data = item, method = "rf", 
                      trControl = tc, tuneGrid = data.frame(.mtry = 20))$results
      tmp_svm <- train(outcome ~ ., data = item, method = "svmRadial", 
                       trControl = tc, tuneGrid = data.frame(.sigma = 1/62, .C = 1))$results
      tmp_knn <- train(outcome ~ ., data = item, method = "knn", 
                       trControl = tc, tuneGrid = data.frame(.k = 5))$results
      tmp_glm <- train(outcome ~ ., data = item, method = "glm",
                       trControl = tc)$results
      tmp_results <- rbind(tmp_rf[(length(tmp_rf) - 5):length(tmp_rf)],
                           tmp_svm[(length(tmp_svm) - 5):length(tmp_svm)],
                           tmp_knn[(length(tmp_knn) - 5):length(tmp_knn)],
                           tmp_glm[(length(tmp_glm) - 5):length(tmp_glm)])
      names_vec <- c("rf", "svmRadial", "knn", "glm")
    }
    
    rownames(tmp_results) <- names_vec
    data_results[[i]] <- tmp_results
  }
  
  data_results
}

plot_against_raw <- function(data_list, raw, plot_names, 
                             col_num = 10, row_num = 200) {
  sample_cols <- sample(1:ncol(raw), col_num)
  sample_rows <- sample(1:nrow(raw), row_num)
  
  tmp_raw <- as.matrix(raw[sample_rows, sample_cols])
  tmp_raw_vec <- as.vector(tmp_raw)
  names_vec <- rep(names(raw)[sample_cols], each = row_num)
  
  against_raw_list <- vector("list", length(data_list))
  for (i in 1:length(data_list)) {
    tmp <- as.matrix(data_list[[i]][sample_rows, sample_cols])
    tmp_vec <- as.vector(tmp)
    
    tmp_df <- as.data.frame(cbind(harmonized = tmp_vec, raw = tmp_raw_vec)) %>% 
      cbind(col_name = names_vec)
    against_raw_list[[i]] <- ggplot(tmp_df) + 
      geom_point(aes(raw, harmonized, color = col_name), size = 0.7) +
      geom_abline(intercept = 0, slope = 1) +
      scale_x_continuous(limits = c(0.5, 4.5)) +
      scale_y_continuous(limits = c(0.5, 4.5)) +
      theme_classic() + 
      labs(title = plot_names[i]) + 
      theme(legend.position="none",
            plot.title = element_text(hjust = 0.5)) +
      scale_color_brewer(palette = "Paired")
  }
  
  ggarrange(plotlist = against_raw_list, 
            ncol = 3, nrow = 2)
}
