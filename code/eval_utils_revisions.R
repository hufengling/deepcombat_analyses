coord_flipper <- function(coord_list, flip_vec) {
  flipped_coord_list <- vector("list", length(coord_list))
  
  for (i in 1:length(coord_list)) {
    tmp_coord <- coord_list[[i]]
    
    # No flip
    if (flip_vec[i] == 0)
      flipped_coord_list[[i]] <- tmp_coord
    # Flip X
    if (flip_vec[i] == 1) {
      tmp_coord[, 1] <- tmp_coord[, 1] * -1
      flipped_coord_list[[i]] <- tmp_coord
    }
    # Flip Y
    if (flip_vec[i] == 2) {
      tmp_coord[, 2] <- tmp_coord[, 2] * -1
      flipped_coord_list[[i]] <- tmp_coord
    }
    # Flip both
    if (flip_vec[i] == 3) {
      tmp_coord[, 1] <- tmp_coord[, 1] * -1
      tmp_coord[, 2] <- tmp_coord[, 2] * -1
      flipped_coord_list[[i]] <- tmp_coord
    }
  }
  
  return(flipped_coord_list)
}

umap_plotter <- function(data_list, covariates, plot_names, 
                         n_neighbors = 20, n_epochs = 100,
                         is_umap_coords = FALSE, ...) {
  umap_plot_list <- vector("list", length(data_list))
  layout_cov_list <- vector("list", length(data_list))
  
  # To plot pre-calculated UMAP coords that have been arbitrarily flipped so signs match
  if (is_umap_coords) {
    for (i in 1:length(data_list)) {
      umap_plot_list[[i]] <- ggplot(data_list[[i]]) + 
        geom_point(aes(`1`, `2`, color = Manufacturer, shape = ".")) +
        xlab("UMAP 1") +
        ylab("UMAP 2") +
        labs(title = plot_names[i]) +
        theme_classic() +
        theme(legend.position = "none",
              plot.title = element_text(hjust = 0.5))
    }
    
    return(ggarrange(plotlist = umap_plot_list, ...))
  }
  
  # Calculate and plot UMAP
  for (i in 1:length(data_list)) {
    data <- data_list[[i]]
    umap_custom <- umap.defaults
    umap_custom$n_neighbors <- n_neighbors
    umap_custom$n_epochs <- n_epochs
    data_umap <- umap(data, config = umap_custom)
    layout_cov <- cbind(data_umap$layout, covariates)
    names(layout_cov)[1:2] <- c(1, 2)
    
    layout_cov_list[[i]] <- layout_cov
    umap_plot_list[[i]] <- ggplot(layout_cov) + 
      geom_point(aes(`1`, `2`, color = Manufacturer, shape = ".")) +
      xlab("UMAP 1") +
      ylab("UMAP 2") +
      labs(title = plot_names[i]) +
      theme_classic() +
      theme(legend.position = "none",
            plot.title = element_text(hjust = 0.5))
    gc()
  }
  return(list(plot = ggarrange(plotlist = umap_plot_list, ...),
              coords = layout_cov_list))
}

pca_plotter <- function(data_list, covariates, plot_names, ...) {
  # if (length(data_list) != 6) {
  #   stop("Length of data_list must be 6")
  # }
  pca_plot_list <- vector("list", length(data_list))
  
  # Calculate and plot PCA
  for (i in 1:length(data_list)) {
    tmp_pca <- princomp(data_list[[i]], cor = TRUE)
    
    pca_plot_list[[i]] <- autoplot(tmp_pca, data = covariates, colour = "Manufacturer", 
                                   frame = TRUE, frame.type = 'norm') + 
      theme_classic() + 
      labs(title = plot_names[i]) +
      theme(legend.position = "none",
            plot.title = element_text(hjust = 0.5))
  }
  return(ggarrange(plotlist = pca_plot_list, ...))
}

cor_plotter <- function(data_list, covariates, plot_names) {
  # if (length(data_list) != 6) {
  #   stop("Length of data_list must be 6")
  # }
  ordered_cormat <- reorder_cormat(cor(data_list[[1]]))
  hclust_ordering <- rownames(ordered_cormat) %>% 
    substring(2) %>% 
    as.numeric()
  
  siemens_cor_list <- vector("list", length(data_list))
  other_cor_list <- vector("list", length(data_list))
  diff_cor_list <- vector("list", length(data_list))
  
  for (i in 1:length(data_list)) {
    # Correlation plots for Siemens
    tmp_siemens <- cor(data_list[[i]][covariates$Manufacturer == "True", ])[hclust_ordering, hclust_ordering]
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
    tmp_other <- cor(data_list[[i]][covariates$Manufacturer == "False", ])[hclust_ordering, hclust_ordering]
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

feature_plotter <- function(data_list, covariates, plot_names, col_num = NULL, ...) {
  # if (length(data_list) != 6) {
  #   stop("Length of data_list must be 6")
  # }
  if (is.null(col_num)) {
    col_num <- sample(1:62, 1)
    cat(paste0("Column number is: ", col_num))
  }
  feature_plot_list <- vector("list", length(data_list))
  for (i in 1:length(data_list)) {
    tmp_df <- as.data.frame(data_list[[i]][, col_num]) %>% cbind(covariates$Manufacturer)
    names(tmp_df) <- c("feature", "Manufacturer")
    tmp_df$Manufacturer <- as.factor(tmp_df$Manufacturer)
    feature_plot_list[[i]] <- ggplot(tmp_df) + 
      geom_density(aes(feature, fill = Manufacturer), alpha = 0.5) +
      theme_classic() +
      theme(legend.position = "none",
            plot.title = element_text(hjust = 0.5)) +
      labs(x = "", y = "", title = plot_names[i])
  }
  
  ggarrange(plotlist = feature_plot_list, ...)
}

stats_tester <- function(data_list, covariates, nboots = 5000, raw_null = FALSE) {
  p_summary <- as.data.frame(matrix(rep(0, 21 * length(data_list)), nrow = length(data_list)))
  manova_df <- as.data.frame(matrix(rep(0, 9 * length(data_list)), nrow = length(data_list)))
  ad_df <- as.data.frame(matrix(rep(0, 5 * length(data_list)), nrow = length(data_list)))
  
  lr_p <- vector("list", length(data_list))
  ad_p <- vector("list", length(data_list))
  
  if (raw_null) {
    tmp_raw <- data_list[[1]]
    data_list[[1]] <- tmp_raw[sample(1:nrow(tmp_raw)), ] 
  }
  
  for (i in 1:length(data_list)) {
    # Univariate linear regression
    mod_mat <- summary(lm(as.matrix(data_list[[i]]) ~ AGE + SEX + DIAGNOSIS + Manufacturer, 
                          data = covariates))
    p_vals <- sapply(mod_mat, function(mod) {
      mod$coefficients[-1, 4]
    }, simplify = T)
    p_vals <- p_vals[-nrow(p_vals), ]
    
    for (j in 1:ncol(data_list[[i]])) {
      mod_mat <- lm(as.matrix(data_list[[i]][, j]) ~ AGE + SEX + DIAGNOSIS + Manufacturer, 
                    data = covariates)
      mod_mat_reduced <- lm(as.matrix(data_list[[i]][, j]) ~ AGE + SEX + DIAGNOSIS, 
                            data = covariates)
      p_vals[nrow(p_vals), j] <- anova(mod_mat, mod_mat_reduced, test = "LRT")[2, 5]
    }
    
    lr_p[[i]] <- p_vals
    mean_p <- rowMeans(p_vals)
    sd_p <- rowSds(p_vals)
    mean_log_p <- rowMeans(log(p_vals, base = 10), na.rm = T)
    sd_log_p <- rowSds(log(p_vals, base = 10), na.rm = T)
    p_summary[i, ] <- c(as.data.frame(names(data_list)[[i]]), 
                        mean_p, sd_p, 
                        mean_log_p, sd_log_p)
    
    # Multivariate MANOVA
    tmp_manova <- summary(manova(as.matrix(data_list[[i]]) ~ AGE + SEX + DIAGNOSIS + Manufacturer, 
                                 data = covariates))$stats
    manova_df[i, ] <- c(as.data.frame(names(data_list)[[i]]), 
                        tmp_manova[-5, 6], 
                        log(tmp_manova[-5, 6], base = 10))
    
    # Univariate Anderson-Darling
    ad_a <- data_list[[i]][covariates$Manufacturer == "GE MEDICAL SYSTEMS", ]
    ad_b <- data_list[[i]][covariates$Manufacturer == "Philips Medical Systems", ]
    ad_c <- data_list[[i]][covariates$Manufacturer == "SIEMENS", ]
    tmp_ad_ab <- rep(0, ncol(data_list[[i]]))
    tmp_ad_ac <- rep(0, ncol(data_list[[i]]))
    tmp_ad_bc <- rep(0, ncol(data_list[[i]]))
    for (j in 1:ncol(data_list[[i]])) {
      tmp_ad_ab[j] <- ad_test(ad_a[, j], ad_b[, j], 
                              keep.boots = F, nboots = nboots)[2]
      tmp_ad_ac[j] <- ad_test(ad_a[, j], ad_c[, j], 
                              keep.boots = F, nboots = nboots)[2]
      tmp_ad_bc[j] <- ad_test(ad_b[, j], ad_c[, j], 
                              keep.boots = F, nboots = nboots)[2]
    }
    ad_pvals <- c(tmp_ad_ab, tmp_ad_ac, tmp_ad_bc)
    ad_p[[i]] <- ad_pvals
    ad_df[i, ] <- c(as.data.frame(names(data_list)[[i]]), 
                    mean(ad_pvals), 
                    sd(ad_pvals),
                    mean(log(ad_pvals, base = 10), na.rm = T), 
                    sd(log(ad_pvals, base = 10), na.rm = T))
    
  }
  colnames(p_summary) <- c("dataset", names(mean_p), paste0(names(mean_p), "_SD"),
                           paste0(names(mean_p), "_log"), paste0(names(mean_p), "_log_SD"))
  colnames(manova_df) <- c("dataset", names(tmp_manova[-5, 6]), paste0(names(tmp_manova[-5, 6]), "_log"))
  colnames(ad_df) <- c("dataset", 
                       "pval", "pval_SD",
                       "pval_log", "pval_log_SD")
  
  
  return(list("lr" = p_summary, "manova" = manova_df, "ad" = ad_df,
              "lr_p" = lr_p, "ad_p" = ad_p))
}

ml_cv_tester <- function(data_list, ml_interval = NULL, covariates, outcome = "Manufacturer",
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
  
  if (outcome %in% c("SEX")) { #two class prediction tasks
    tc <- trainControl(method = "repeatedcv", 
                       number = k_fold, repeats = repeats,
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE,
                       savePredictions = TRUE,
                       verboseIter = verbose,
                       sampling = "up")
  }
  
  if (outcome %in% c("DIAGNOSIS", "Manufacturer")) { #multiclass prediction
    tc <- trainControl(method = "repeatedcv", 
                       number = k_fold, repeats = repeats,
                       verboseIter = verbose,
                       sampling = "up")
  }
  
  if (outcome %in% c("AGE")) { #regression
    tc <- trainControl(method = "repeatedcv", 
                       number = k_fold, repeats = repeats,
                       verboseIter = verbose)
  }
  
  data_results <- lapply(tmp_list, function(item) {
    if (verbose)
      print(name(item))
    if (outcome == "SEX") {
      tmp_rf <- caret::train(outcome ~ ., data = item, method = "rf",
                             trControl = tc,
                             tuneGrid = data.frame(.mtry = 20),
                             metric = "ROC")$results
      tmp_svm <- caret::train(outcome ~ ., data = item, method = "svmRadial", 
                              trControl = tc, 
                              tuneGrid = data.frame(.sigma = 1/62, .C = 1),
                              metric = "ROC")$results
      tmp_knn <- caret::train(outcome ~ ., data = item, method = "knn", 
                              trControl = tc, 
                              tuneGrid = data.frame(.k = 5),
                              metric = "ROC")$results
      tmp_qda <- caret::train(outcome ~ ., data = item, method = "qda", 
                              trControl = tc, metric = "ROC")$results
      tmp_xgb <- caret::train(outcome ~ ., data = item, method = "xgbTree",
                              trControl = tc,
                              tuneGrid = data.frame(.nrounds = 100,
                                                    .max_depth = 6,
                                                    .eta = 0.1,
                                                    .gamma = 0,
                                                    .colsample_bytree = .5,
                                                    .min_child_weight = 1,
                                                    .subsample = 1),
                              metric = "ROC")$results
      tmp_results <- rbind(tmp_rf[(length(tmp_rf) - 5):length(tmp_rf)],
                           tmp_svm[(length(tmp_svm) - 5):length(tmp_svm)],
                           tmp_knn[(length(tmp_knn) - 5):length(tmp_knn)],
                           tmp_qda[(length(tmp_qda) - 5):length(tmp_qda)],
                           tmp_xgb[(length(tmp_xgb) - 5):length(tmp_xgb)])
      names_vec <- c("rf", "svmRadial", "knn", "qda", "xgb")
    }
    
    if (outcome %in% c("DIAGNOSIS", "Manufacturer")) {
      tmp_rf <- caret::train(outcome ~ ., data = item, method = "rf", 
                             trControl = tc, 
                             tuneGrid = data.frame(.mtry = 20))$results
      tmp_svm <- caret::train(outcome ~ ., data = item, method = "svmRadial", 
                              trControl = tc, 
                              tuneGrid = data.frame(.sigma = 1/62, .C = 1))$results
      tmp_knn <- caret::train(outcome ~ ., data = item, method = "knn", 
                              trControl = tc, 
                              tuneGrid = data.frame(.k = 5))$results
      tmp_qda <- caret::train(outcome ~ ., data = item, method = "qda", 
                              trControl = tc)$results
      tmp_xgb <- caret::train(outcome ~ ., data = item, method = "xgbTree",
                              trControl = tc, 
                              tuneGrid = data.frame(.nrounds = 100, 
                                                    .max_depth = 6, 
                                                    .eta = 0.1, 
                                                    .gamma = 0, 
                                                    .colsample_bytree = .5,
                                                    .min_child_weight = 1, 
                                                    .subsample = .75))$results
      tmp_results <- rbind(tmp_rf[(length(tmp_rf) - 3):length(tmp_rf)],
                           tmp_svm[(length(tmp_svm) - 3):length(tmp_svm)],
                           tmp_knn[(length(tmp_knn) - 3):length(tmp_knn)],
                           tmp_qda[(length(tmp_qda) - 3):length(tmp_qda)],
                           tmp_xgb[(length(tmp_xgb) - 3):length(tmp_xgb)])
      names_vec <- c("rf", "svmRadial", "knn", "qda", "xgb")
    }
    
    if (outcome == "AGE") {
      tmp_rf <- caret::train(outcome ~ ., data = item, method = "rf", 
                             trControl = tc, 
                             tuneGrid = data.frame(.mtry = 20))$results
      tmp_svm <- caret::train(outcome ~ ., data = item, method = "svmRadial", 
                              trControl = tc, 
                              tuneGrid = data.frame(.sigma = 1/62, .C = 1))$results
      tmp_knn <- caret::train(outcome ~ ., data = item, method = "knn", 
                              trControl = tc, 
                              tuneGrid = data.frame(.k = 5))$results
      tmp_xgb <- caret::train(outcome ~ ., data = item, method = "xgbTree",
                              trControl = tc, 
                              tuneGrid = data.frame(.nrounds = 100, 
                                                    .max_depth = 6, 
                                                    .eta = 0.1, 
                                                    .gamma = 0, 
                                                    .colsample_bytree = .5,
                                                    .min_child_weight = 1, 
                                                    .subsample = .75))$results
      tmp_results <- rbind(tmp_rf[(length(tmp_rf) - 5):length(tmp_rf)],
                           tmp_svm[(length(tmp_svm) - 5):length(tmp_svm)],
                           tmp_knn[(length(tmp_knn) - 5):length(tmp_knn)],
                           tmp_xgb[(length(tmp_xgb) - 5):length(tmp_xgb)])
      names_vec <- c("rf", "svmRadial", "knn", "xgb")
    }
    
    rownames(tmp_results) <- names_vec
    return(tmp_results)
  })
  
  names(data_results) <- names(tmp_list)
  bind_rows(data_results)
}

plot_against_raw <- function(data_list, raw, plot_names, 
                             col_num = 10, row_num = 200, ...) {
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
      scale_x_continuous(limits = c(0.25, 5.5)) +
      scale_y_continuous(limits = c(0.25, 5.5)) +
      theme_classic() + 
      labs(title = plot_names[i]) + 
      theme(legend.position="none",
            plot.title = element_text(hjust = 0.5)) +
      scale_color_brewer(palette = "Paired")
  }
  
  ggarrange(plotlist = against_raw_list, ...)
}
gg_qqplot <- function(ps, ci = 0.95, test) {
  n  <- length(ps)
  df <- data.frame(
    observed = -log10(sort(ps)),
    expected = -log10(ppoints(n)),
    clower   = -log10(qbeta(p = (1 - ci) / 2, shape1 = 1:n, shape2 = n:1)),
    cupper   = -log10(qbeta(p = (1 + ci) / 2, shape1 = 1:n, shape2 = n:1))
  )
  if (test == "lr") {
    log10Pe <- expression(paste("Expected Uniform -log"[10], plain(p)))
    log10Po <- expression(paste("Observed LR -log"[10], plain(p)))
  }
  if (test == "ad") {
    log10Pe <- expression(paste("Expected Uniform -log"[10], plain(p)))
    log10Po <- expression(paste("Observed AD -log"[10], plain(p)))
  }
  ggplot(df) +
    geom_ribbon(
      mapping = aes(x = expected, ymin = clower, ymax = cupper),
      alpha = 0.1
    ) +
    geom_point(aes(expected, observed), shape = 1, size = 3) +
    geom_abline(intercept = 0, slope = 1, alpha = 0.5) +
    # geom_line(aes(expected, cupper), linetype = 2, size = 0.5) +
    # geom_line(aes(expected, clower), linetype = 2, size = 0.5) +
    xlab(log10Pe) +
    ylab(log10Po)
}

plot_p_distributions <- function(manufac_stats, 
                                 index_range, plot_names, 
                                 test = c("lr", "ad"), y_scale = c(NA, NA, NA), 
                                 use_neg_log = FALSE, 
                                 plot_type = c("density", "qq"), ...) {
  if (test == "lr") {
    p_dist_df <- manufac_stats$lr_p[index_range]
  }
  if (test == "ad") {
    p_dist_df <- manufac_stats$ad_p[index_range]
  }
  if (!test %in% c("lr", "ad")) {
    stop("test must be 'lr' or 'ad'")
  }
  if (!plot_type %in% c("density", "qq")) {
    stop("plot_type must be 'density' or 'qq'")
  }
  
  p_dist_list <- sapply(1:length(p_dist_df), function(i) {
    if (test == "lr") {
      tmp <- as.data.frame(t(p_dist_df[[i]]))
      names(tmp) <- c("age", "sex", "diagnosiscn", "diagnosislmci", "batch")
    }
    if (test == "ad") {
      tmp <- as.data.frame(p_dist_df[[i]])
      names(tmp) <- "batch"
    }
    tmp
  }, simplify = FALSE)
  
  p_plot_list <- vector("list", length(p_dist_list))
  # if (plot_type == "density") {
  #   for (i in 1:length(p_dist_list)) {
  #     p_plot_list[[i]] <- ggplot(p_dist_list[[i]], aes(x = manufacTrue)) +
  #       geom_density() +
  #       ylab("Density") + 
  #       xlim(0, 1) +
  #       theme_classic() + 
  #       labs(title = plot_names[i]) + 
  #       theme(legend.position="none",
  #             plot.title = element_text(hjust = 0.5))
  #     if (test == "lr") {
  #       p_plot_list[[i]] <- p_plot_list[[i]] + 
  #         xlab("Regression p-value")
  #     }
  #     if (test == "ad") {
  #       p_plot_list[[i]] <- p_plot_list[[i]] + 
  #         xlab("Anderson-Darling p-value")
  #     }
  #     if (!is.na(y_scale[1])) {
  #       if (i == 1)
  #         p_plot_list[[i]] <- p_plot_list[[i]] + ylim(0, y_scale[1])
  #     }
  #     if (!is.na(y_scale[2])) {
  #       if (i %in% c(2, 3, 6))
  #         p_plot_list[[i]] <- p_plot_list[[i]] + ylim(0, y_scale[2])
  #     }
  #     if (!is.na(y_scale[3])) {
  #       if (i %in% c(4, 5))
  #         p_plot_list[[i]] <- p_plot_list[[i]] + ylim(0, y_scale[3])
  #     }
  #   }
  # }
  if (plot_type == "qq") {
    for (i in 1:length(p_dist_list)) {
      p_plot_list[[i]] <- gg_qqplot(p_dist_list[[i]]$batch, test = test) +
        theme_classic() +
        labs(title = plot_names[i]) +
        theme(plot.title = element_text(hjust = 0.5))
      
      if (!is.na(y_scale[1])) {
        if (i == 1)
          p_plot_list[[i]] <- p_plot_list[[i]] + ylim(0, y_scale[1])
      }
      if (!is.na(y_scale[2])) {
        if (i %in% c(2, 3, 6))
          p_plot_list[[i]] <- p_plot_list[[i]] + ylim(0, y_scale[2])
      }
      if (!is.na(y_scale[3])) {
        if (i %in% c(4, 5))
          p_plot_list[[i]] <- p_plot_list[[i]] + ylim(0, y_scale[3])
      }
    }
  }
  ggarrange(plotlist = p_plot_list, ...)
}
