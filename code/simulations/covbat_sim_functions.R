# Functions for CovBat simulations
# by Andrew Chen

#### Data-Generating Function: Additive Covariance Effect ####
# modified in this code to allow for binary covariates
# df_gen_add <- function(mu, # p-dimension intercept vector
#                        Sigma, # p x p covariance matrix for error terms
#                        delta.alpha, # alpha for igamma multiplication factor per site
#                        delta.beta, # beta igamma multiplication factor per site
#                        cov.mat, # p x p x m covariance matrix addition base
#                        cov.dist, # distribution function to draw from to multiply cov.mat by
#                        cov.par, # parameters for cov.dist
#                        cov.fixed = NULL, # fixed constants instead of draws
#                        gamma.mu, # vector of mu for gamma
#                        gamma.Sigma, # covariance matrix for gamma
#                        co.mu, # vector of covariate means
#                        co.Sigma, # covariance matrix for covariate
#                        co.prob = NULL, # covariate probabilities, binomial dist if ne NULL
#                        beta, # matrix of coefficients for covariates
#                        force.pd = TRUE, # use nearPD to force cov.mat to be PD
#                        m = 2, # number of sites
#                        n = rep(100, 2) # vector of number of subjects per site
# ) {
#   
#   x <- vector(mode = "list", length = m)
#   
#   if (is.null(co.prob)) {
#     for (i in 1:m) {
#       x[[i]] <- rmvnorm(n[i], co.mu, co.Sigma)
#     }
#   } else {
#     for (i in 1:m) {
#       x[[i]] <- matrix(rbinom(n[i], 1, co.prob[i]), n[i], 1)
#     }
#   }
#   
#   delta <- array(0, dim = c(m, length(mu), length(mu)))
#   invg <- matrix(0, 3, length(mu))
#   for (i in 1:m) {
#     invg[i,] <- sqrt(rinvgamma(62, delta.alpha[i], delta.beta[i]))
#   }
#   
#   gamma <- matrix(0, m, length(mu))
#   for (i in 1:m) {
#     gamma[i, ] <- rmvnorm(1, gamma.mu, gamma.Sigma)
#   }
#   
#   delta.diag = matrix(0, m, length(mu))
#   
#   covs <- NULL
#   y <- vector(mode = "list", length = m)
#   for (i in 1:m) {
#     y_mean <- mu + gamma[i, ]
#     
#     if (is.null(cov.fixed)) {
#       cov_add <- cov.dist(1, cov.par) * cov.mat[,,i] # multiply cov.mat by draw
#     } else {
#       cov_add <- cov.fixed[i] * cov.mat[,,i] 
#     }
#     diag(cov_add) <- abs(diag(cov_add))
#     
#     y_cov <- Sigma + cov_add
#     
#     y_cov <- diag(invg[i,]) %*% y_cov %*% diag(invg[i,])
#     
#     if (force.pd) {
#       y_cov <- as.matrix(nearPD(y_cov, maxit = 1000)$mat)
#     }
#     
#     covs[[i]] <- y_cov
#     
#     y[[i]] <- rmvnorm(n[i], y_mean, y_cov)
#     y[[i]] <- y[[i]] + x[[i]] %*% beta
#   }
#   
#   bat <- c(sapply(1:m, function(x) rep(x, n[x])))
#   
#   x_out <- x[[1]]
#   y_out <- y[[1]]
#   for (i in 2:m) {
#     x_out <- rbind(x_out, x[[i]])
#     y_out <- rbind(y_out, y[[i]])
#   }
#   
#   list(x = x_out, y = y_out, bat = bat, delta = delta, gamma = gamma,
#        cov = covs)
# }

#### Data-Generating Function: Additive Confounded ####
# ties covariate to amount of covariance effect
# association same across site, each site still has its own covariance addition
df_gen_confounded <- function(mu, # p-dimension intercept vector
                              Sigma, # p x p covariance matrix for error terms
                              delta.alpha, # alpha for igamma multiplication factor per site
                              delta.beta, # beta igamma multiplication factor per site
                              cov.mat, # p x p x m covariance matrix addition base
                              cov.dist, # distribution function to draw from to multiply cov.mat by
                              cov.par, # parameters for cov.dist
                              cov.fixed = NULL, # fixed constants instead of draws
                              gamma.mu, # vector of mu for gamma
                              gamma.Sigma, # covariance matrix for gamma
                              co.mu, # vector of covariate means
                              co.Sigma, # covariance matrix for covariate
                              co.prob = NULL, # covariate probabilities, binomial dist if ne NULL
                              beta, # matrix of coefficients for covariates
                              beta.var = NULL, # beta variance scaling factor
                              beta.cov.mat = NULL, # p x p x q confound covariance matrix addition base
                              beta.cov = NULL, # q-dimensional vector for amount of each covariance scaling matrix (beta.cov.mat) per covariate
                              force.pd = TRUE, # force PD by setting negative eigenvalues equal to small constant
                              pd.tol = 1e-12, # small constant for negative eigenvalues
                              no.gamma = FALSE, # no mean site effect
                              no.delta = FALSE, # no variance site effect
                              fix.var = FALSE, # scale variance back to that of Sigma after covariance manipulation
                              m = 2, # number of sites
                              n = rep(100, 2) # vector of number of subjects per site
) {
  p <- length(mu)
  
  x <- vector(mode = "list", length = m)
  
  if (is.null(co.prob)) {
    for (i in 1:m) {
      x[[i]] <- rmvnorm(n[i], co.mu, co.Sigma)
    }
  } else {
    for (i in 1:m) {
      x[[i]] <- matrix(rbinom(n[i], 1, co.prob[i]), n[i], 1)
    }
  }
  
  delta <- array(0, dim = c(m, p, p))
  invg <- matrix(0, 3, p)
  for (i in 1:m) {
    invg[i,] <- sqrt(rinvgamma(p, delta.alpha[i], delta.beta[i]))
    if (no.delta) {invg[i,] <- rep(1, p)}
  }
  
  gamma <- matrix(0, m, p)
  for (i in 1:m) {
    gamma[i, ] <- rmvnorm(1, gamma.mu, gamma.Sigma)
    if (no.gamma) {gamma[i,] <- 0}
  }
  
  delta.diag = matrix(0, m, p)
  
  covs <- vector(mode = "list", length = m)
  y <- vector(mode = "list", length = m)
  for (i in 1:m) {
    y_cov <- vector(mode = "list", length = 2)
    
    y_mean <- mu + gamma[i, ]
    
    if (is.null(cov.fixed)) {
      cov_add <- cov.dist(1, cov.par) * cov.mat[,,i] # multiply cov.mat by draw
    } else {
      cov_add <- cov.fixed[i] * cov.mat[,,i]
    }
    
    conf_cov <- matrix(0, p, p)
    for (j in 1:dim(beta.cov.mat)[3]) {
      conf_cov <- conf_cov + beta.cov[j] * beta.cov.mat[,,j]
    }
    
    y_cov[[1]] <- Sigma + cov_add
    y_cov[[2]] <- Sigma + cov_add + conf_cov
    
    # get nearest positive definite matrix
    if (force.pd) {
      y_cov[[1]] <- nearPD_eig(y_cov[[1]])
      y_cov[[2]] <- nearPD_eig(y_cov[[2]])
    }
    
    # rescale variances to match original variances
    if (fix.var) {
      var_scale <- sqrt(diag(Sigma)/diag(y_cov[[1]]))
      y_cov[[1]] <- diag(var_scale) %*% y_cov[[1]] %*% diag(var_scale)
      var_scale <- sqrt(diag(Sigma)/diag(y_cov[[2]]))
      y_cov[[2]] <- diag(var_scale) %*% y_cov[[2]] %*% diag(var_scale)
    }
    
    # covariate effect on variance
    if (is.null(beta.var)) {beta.var <- rep(1, p)}
    y_cov[[2]] <- diag(sqrt(beta.var)) %*% y_cov[[2]] %*% diag(sqrt(beta.var))
    
    # site effect on variance
    y_cov[[1]] <- diag(invg[i,]) %*% y_cov[[1]] %*% diag(invg[i,])
    y_cov[[2]] <- diag(invg[i,]) %*% y_cov[[2]] %*% diag(invg[i,])
    
    covs[[i]] <- y_cov
    
    y[[i]] <- matrix(0, n[i], p)
    
    if (sum(x[[i]] == 0) > 0) {
      y[[i]][x[[i]] == 0,] <- rmvnorm(sum(x[[i]] == 0), y_mean, y_cov[[1]])
    }
    if (sum(x[[i]] == 1) > 0) {
      y[[i]][x[[i]] == 1,] <- rmvnorm(sum(x[[i]] == 1), y_mean, y_cov[[2]])
    }
    y[[i]] <- y[[i]] + x[[i]] %*% beta
  }
  
  bat <- c(sapply(1:m, function(x) rep(x, n[x])))
  
  x_out <- x[[1]]
  y_out <- y[[1]]
  for (i in 2:m) {
    x_out <- rbind(x_out, x[[i]])
    y_out <- rbind(y_out, y[[i]])
  }
  
  list(x = x_out, y = y_out, bat = bat, delta = delta, gamma = gamma,
       cov = covs, mu = mu, Sigma = Sigma)
}

#### Main Simulation Function ####
# takes simulated dataset and returns MVPA result
# target.cov: pick site that is treated as ground truth covariance to compare to
mvpa_sim <- function(df_in, no.mvpa = FALSE, residualize = TRUE, target.cov = 2, n.pc = NULL) {
  site <- as.factor(df_in$bat == 1) # binary Siemens vs not
  bat <- as.factor(df_in$bat)
  bat <- droplevels(bat)
  
  covt_mod <- as.factor(df_in$x)
  dx <- as.factor(df_in$x)
  
  ## Get harmonized dataset not residualized
  y_combat <- combat(t(df_in$y), bat = bat, mod = covt_mod)
  y_covbat <- covbat(t(df_in$y), bat = bat, mod = covt_mod, n.pc = n.pc)
  
  data <- NULL
  data[[1]] <- as.data.frame(df_in$y)
  data[[2]] <- as.data.frame(t(y_combat$dat.combat))
  data[[3]] <- as.data.frame(t(y_covbat$dat.covbat))
  
  ## Get harmonized dataset residualized
  y_combat <- combat(t(df_in$y), bat = bat, mod = covt_mod, resid = TRUE)
  y_covbat <- covbat(t(df_in$y), bat = bat, mod = covt_mod, resid = TRUE,
                     n.pc = n.pc)
  
  res_data <- NULL
  res_data[[1]] <- as.data.frame(df_in$y - t(y_combat$stand.mean))
  res_data[[2]] <- as.data.frame(t(y_combat$dat.combat))
  res_data[[3]] <- as.data.frame(t(y_covbat$dat.covbat))
  
  # # check sample matrices
  # ggarrange(annotate_figure(plot_mat(cor(data[[1]][df_in$bat == 1,])) + theme(legend.position = "none"), left = "Unharmonized", top = "Site 1"),
  #           annotate_figure(plot_mat(cor(data[[1]][df_in$bat == 2,])) + theme(legend.position = "none"), top = "Site 2"),
  #           annotate_figure(plot_mat(cor(data[[1]][df_in$bat == 3,])) + theme(legend.position = "none"), top = "Site 3"),
  #           annotate_figure(plot_mat(cor(data[[2]][df_in$bat == 1,])) + theme(legend.position = "none"), left = "ComBat-Adjusted"),
  #           plot_mat(cor(data[[2]][df_in$bat == 2,])),
  #           plot_mat(cor(data[[2]][df_in$bat == 3,])),
  #           annotate_figure(plot_mat(cor(data[[3]][df_in$bat == 1,])) + theme(legend.position = "none"), left = "CovBat-Adjusted"),
  #           plot_mat(cor(data[[3]][df_in$bat == 2,])),
  #           plot_mat(cor(data[[3]][df_in$bat == 3,])),
  #           nrow = 3, ncol = 3,
  #           widths = c(1.1, 1, 1),
  #           heights = c(1.1, 1, 1),
  #           legend = "right", common.legend = TRUE)
  # 
  # ggarrange(annotate_figure(plot_mat(cor(res_data[[1]][df_in$bat == 1,])) + theme(legend.position = "none"), left = "Unharmonized", top = "Site 1"),
  #           annotate_figure(plot_mat(cor(res_data[[1]][df_in$bat == 2,])) + theme(legend.position = "none"), top = "Site 2"),
  #           annotate_figure(plot_mat(cor(res_data[[1]][df_in$bat == 3,])) + theme(legend.position = "none"), top = "Site 3"),
  #           annotate_figure(plot_mat(cor(res_data[[2]][df_in$bat == 1,])) + theme(legend.position = "none"), left = "ComBat-Adjusted"),
  #           plot_mat(cor(res_data[[2]][df_in$bat == 2,])),
  #           plot_mat(cor(res_data[[2]][df_in$bat == 3,])),
  #           annotate_figure(plot_mat(cor(res_data[[3]][df_in$bat == 1,])) + theme(legend.position = "none"), left = "CovBat-Adjusted"),
  #           plot_mat(cor(res_data[[3]][df_in$bat == 2,])),
  #           plot_mat(cor(res_data[[3]][df_in$bat == 3,])),
  #           nrow = 3, ncol = 3,
  #           widths = c(1.1, 1, 1),
  #           heights = c(1.1, 1, 1),
  #           legend = "right", common.legend = TRUE)
  
  ## Get target covariance matrix
  # covs <- lapply(df_in$cov, "[[", 1)
  avg_cov <- df_in$cov[[target.cov]][[1]]
  
  ## Store covariance/correlation matrix distances
  all_pair <- combn(1:m, 2)
  corr <- NULL
  corr[[1]] <- lapply(1:3, function(x) cor(data[[1]][df_in$bat == x,]))
  corr[[2]] <- lapply(1:3, function(x) cor(data[[2]][df_in$bat == x,]))
  corr[[3]] <- lapply(1:3, function(x) cor(data[[3]][df_in$bat == x,]))
  
  cor_dist <- matrix(0, ncol(all_pair), 3, 
                     dimnames = list(
                       apply(all_pair, 2, 
                             function(x) paste(x, collapse = ",")), 
                       c("Unharmonized", "ComBat", "CovBat"))) # row = pair ID
  for (j in 1:3) {
    for (i in 1:ncol(all_pair)) {
      cor_dist[i, j] <- norm(corr[[j]][[all_pair[1, i]]] - 
                               corr[[j]][[all_pair[2, i]]], "f")
    }
  }
  
  cor_truth <- matrix(0, 4, 3, 
                      dimnames = list(
                        c("Site 1", "Site 2", "Site 3", "Overall"), 
                        c("Unharmonized", "ComBat", "CovBat"))) # row = pair ID
  for (j in 1:3) {
    for (i in 1:3) {
      cor_truth[i, j] <- norm(corr[[j]][[i]] - cov2cor(avg_cov), "f")
    }
  }
  for (j in 1:3) {
    cor_truth[4, j] <- norm(cor(data[[j]]) - cov2cor(avg_cov), "f")
  }
  
  covv <- NULL
  covv[[1]] <- lapply(1:3, function(x) cov(data[[1]][df_in$bat == x,]))
  covv[[2]] <- lapply(1:3, function(x) cov(data[[2]][df_in$bat == x,]))
  covv[[3]] <- lapply(1:3, function(x) cov(data[[3]][df_in$bat == x,]))
  
  cov_dist <- matrix(0, ncol(all_pair), 3, 
                     dimnames = list(
                       apply(all_pair, 2, 
                             function(x) paste(x, collapse = ",")), 
                       c("Unharmonized", "ComBat", "CovBat"))) # row = pair ID
  for (j in 1:3) {
    for (i in 1:ncol(all_pair)) {
      cov_dist[i, j] <- norm(covv[[j]][[all_pair[1, i]]] - 
                               covv[[j]][[all_pair[2, i]]], "f")
    }
  }
  
  cov_truth <- matrix(0, 4, 3, 
                      dimnames = list(
                        c("Site 1", "Site 2", "Site 3", "Overall"), 
                        c("Unharmonized", "ComBat", "CovBat"))) # row = pair ID
  
  for (j in 1:3) {
    for (i in 1:3) {
      cov_truth[i, j] <- norm(covv[[j]][[i]] - avg_cov, "f")
    }
  }
  # get overall covariance compare to truth
  for (j in 1:3) {
    cov_truth[4, j] <- norm(cov(data[[j]]) - avg_cov, "f")
  }
  
  ### MANOVA results for site and dx
  man_site <- lapply(data, function(x) manova(as.matrix(x) ~ bat))
  names(man_site) <- c("Unharmonized", "ComBat", "CovBat")
  man_site_p <- try(sapply(man_site, function(x) summary(x)$stats[1,6]))
  
  if (is(man_site_p, "try-error")) {return(NULL)}
  
  man_dx <- lapply(data, function(x) manova(as.matrix(x) ~ dx))
  names(man_dx) <- c("Unharmonized", "ComBat", "CovBat")
  man_dx_p <- try(sapply(man_dx, function(x) summary(x)$stats[1,6]))
  
  if (is(man_dx_p, "try-error")) {return(NULL)}
  
  # return only data results if not performing MVPA
  if (no.mvpa) {
    return(list(corr.dist = cor_dist, 
                cov.dist = cov_dist, corr.truth = cor_truth, cov.truth = cov_truth,
                manova.site = man_site_p, manova.dx = man_dx_p,
                combat = y_combat, covbat = y_covbat,
                resid.data = res_data,
                corr = corr, covv = covv))
  }
  
  ## MVPA for site
  aucs <- NULL
  if (residualize) {
    for (sim in 1:nsim) {
      aucs <- rbind(aucs, c(auc_test(res_data[[1]], site, trainpct = 0.5),
                            auc_test(res_data[[2]], site, trainpct = 0.5),
                            auc_test(res_data[[3]], site, trainpct = 0.5)))
    }
  } else {
    for (sim in 1:nsim) {
      aucs <- rbind(aucs, c(auc_test(data[[1]], site, trainpct = 0.5),
                            auc_test(data[[2]], site, trainpct = 0.5),
                            auc_test(data[[3]], site, trainpct = 0.5)))
    }
  }
  if (anyNA(aucs)) {
    message("Single class in training/testing sample")
    return(NULL)
  }
  colnames(aucs) <- c("Unharmonized", "ComBat", "CovBat")
  aucs_melt <- melt(aucs)
  aucs_melt$grp <- as.factor(aucs_melt$Var2)
  aucs_site <- aucs

  ## MVPA for diagnosis
  aucs <- NULL
  for (sim in 1:nsim) {
    aucs <- rbind(aucs, c(auc_test(data[[1]], dx, trainpct = 0.5),
                          auc_test(data[[2]], dx, trainpct = 0.5),
                          auc_test(data[[3]], dx, trainpct = 0.5)))
  }
  if (anyNA(aucs)) {
    message("Single class in training/testing sample")
    return(NULL)
  }
  colnames(aucs) <- c("Unharmonized", "ComBat", "CovBat")
  aucs_dx <- aucs
  aucs_melt <- melt(aucs)
  aucs_melt$grp <- as.factor(aucs_melt$Var2)
  aucs_dx <- aucs
  
  return(list(site = aucs_site, dx = aucs_dx, corr.dist = cor_dist, 
              cov.dist = cov_dist, corr.truth = cor_truth, cov.truth = cov_truth,
              manova.site = man_site_p, manova.dx = man_dx_p,
              combat = y_combat, covbat = y_covbat,
              resid.data = res_data,
              corr = corr, covv = covv))
}

#### AUC test function ####
auc_test <- function(data, class, trainpct = 0.8, train = NULL # input sampling
) {
  data$class <- class
  if (is.null(train)) {train <- sample(nrow(data), trainpct*nrow(data), replace = FALSE)}
  TrainSet <- data[train,]
  ValidSet <- data[-train,]
  
  # handle situations with no classes in training set
  if ((length(unique(TrainSet$class)) == 1) || 
      (length(unique(ValidSet$class)) == 1)) {
    return(NA)
  }
  
  model <- randomForest(class ~ ., data = TrainSet)
  predValid <- predict(model, ValidSet, type = "prob")
  rf_roc <- roc(ValidSet$class, predValid[,1], quiet = TRUE)
  auc(rf_roc)
}

#### AUC test with unharmonized test set ####
auc_test_valid <- function(data, # harmonized subset
                           raw.data, # unharmonized data
                           class,
                           raw.class # unharmonized class
) {
  data$class <- class
  raw.data$class <- raw.class
  TrainSet <- data
  ValidSet <- raw.data
  model <- randomForest(class ~ ., data = TrainSet)
  predValid <- predict(model, ValidSet, type = "prob")
  rf_roc <- roc(ValidSet$class, predValid[,1], quiet = TRUE)
  auc(rf_roc)
}

# plot covariance/correlation matrix
plot_mat <- function(cov, lims = c(-max(abs(cov_melt$value)), max(abs(cov_melt$value)))) {
  cov_melt <- melt(cov)
  ggplot(data = cov_melt, aes(x=Var1, y=Var2, fill=value)) + 
    geom_tile() + 
    scale_fill_gradientn(
      colours = c(wong_colors[6], "white", wong_colors[5]), 
      limits = lims) +
    labs(fill = "") + 
    theme_classic() +
    theme(axis.line = element_blank(),
          axis.ticks = element_blank(),
          axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank())
}


# nearest positive definite matrix by changing negative eigenvalues
nearPD_eig <- function (x, tol = 1e-12) {
  eig <- eigen(x)
  ev <- eig$values
  ev[ev < 0] <- tol
  eig$vectors %*% diag(ev) %*% t(eig$vectors)
}



