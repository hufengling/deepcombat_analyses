##### Confounding: 
# Site effect in covariance
# Covariate effect in mean/covariance, confounded with site effect
library(reshape2)
library(mvtnorm)
library(invgamma)
library(randomForest)
library(pROC)
library(ROCR)
library(caret)
library(Matrix)

library(CovBat)
source("covbat_sim_functions.R")

load("animats_add.Rdata")
load("covbat_sim_params.Rdata")

# simulation parameters
ndf <- 1000
nsim <- 1 # number of AUC test runs
m = 3
n = rep(250, 3) # vector of subject number per site
mu <- apply(adni_ct_33, 2, mean)
sigma <- cor(adni_ct_33) # correlation matrix base to have var 1 errors

# 1 = low, 2 = med, 3 = high, 4 = none
s <- 1 # gamma
d <- 1 # delta
c <- 1 # covariance site effect
b <- 1 # beta mean effect
bv <- 4 # beta var effect
bc <- 1 # beta covariance effect

# store results as all_sim$sf_bf
# where sf is site effect level and bf is covariate effect level and cf is
# site covariance effect level
all_sim <- list()

set.seed(888)
for (sf in 1:3) {
  for (bf in 1:3) {
    for (cf in 1:3) {
      s <- sf
      d <- sf
      c <- cf
      b <- bf
      bc <- bf
      
      site_all <- list()
      dx_all <- list()
      corr_all <- list()
      cov_all <- list()
      
      for (i in 1:ndf) {
        df_in <- df_gen_confounded(mu,
                                   sigma,
                                   gamma.mu = rep(0, 62),
                                   gamma.Sigma = diag(gamma.sigma[s], 62),
                                   delta.alpha = alpha[,d],
                                   delta.beta = beta[,d],
                                   cov.mat = allmat_0,
                                   cov.fixed = site.cov[,c],
                                   co.prob = rep(0.25, 3),
                                   beta = betas[,b],
                                   beta.var = betas.var[,bv],
                                   beta.cov.mat = allmat_0,
                                   beta.cov = betas.cov[,bc],
                                   force.pd = TRUE,
                                   m = m,
                                   n = n,
                                   no.gamma = FALSE,
                                   no.delta = FALSE,
                                   fix.var = TRUE)
        out <- mvpa_sim(df_in)
        
        site_all[[i]] <- out$site
        dx_all[[i]] <- out$dx
        corr_all[[i]] <- out$corr.dist
        cov_all[[i]] <- out$cov.dist
      }
      
      site_med <- do.call(rbind, lapply(site_all, apply, 2, median))
      dx_med <- do.call(rbind, lapply(dx_all, apply, 2, median))
      
      all_sim[[paste(sf, bf, cf, sep = "_")]] <- list(site.AUC = site_all, 
                                                      dx.AUC = dx_all, 
                                                      corr.dist = corr_all,
                                                      cov.dist = cov_all, 
                                                      site.AUC.median = site_med, 
                                                      dx.AUC.median = dx_med)
    }
  }
}

save(all_sim, file = "sim_confounding.Rdata")