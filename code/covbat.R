source("./CovBat_Harmonization/R/R/utils.R")

covbat_fh <- function (dat, bat, mod = NULL, percent.var = 0.95, n.pc = NULL, 
                       train = NULL, mean.only = FALSE, std.var = TRUE, resid = FALSE, 
                       eb = TRUE, parametric = TRUE, score.eb = FALSE, score.parametric = TRUE, 
                       verbose = FALSE) 
{
  dat <- as.matrix(dat)
  .checkConstantRows <- function(dat) {
    sds <- rowSds(dat)
    ns <- sum(sds == 0)
    if (ns > 0) {
      message <- paste0(ns, " rows (features) were found to be constant \n                        across samples. Please remove these rows before \n                        running ComBat.")
      stop(message)
    }
  }
  .checkConstantRows(dat)
  if (eb) {
    if (verbose) 
      cat("[combat] Performing ComBat with empirical Bayes\n")
  }
  else {
    if (verbose) 
      cat("[combat] Performing ComBat without empirical Bayes \n                     (L/S model)\n")
  }
  batch <- as.factor(bat)
  batchmod <- model.matrix(~-1 + batch)
  if (verbose) 
    cat("[combat] Found", nlevels(batch), "batches\n")
  n.batch <- nlevels(batch)
  batches <- lapply(levels(batch), function(x) which(batch == 
                                                       x))
  n.batches <- sapply(batches, length)
  n.array <- sum(n.batches)
  design <- cbind(batchmod, mod)
  check <- apply(design, 2, function(x) all(x == 1))
  design <- as.matrix(design[, !check])
  if (verbose) 
    cat("[combat] Adjusting for", ncol(design) - ncol(batchmod), 
        "covariate(s) or covariate level(s)\n")
  if (qr(design)$rank < ncol(design)) {
    if (ncol(design) == (n.batch + 1)) {
      stop("[combat] The covariate is confounded with batch. Remove the \n           covariate and rerun ComBat.")
    }
    if (ncol(design) > (n.batch + 1)) {
      if ((qr(design[, -c(1:n.batch)])$rank < ncol(design[, 
                                                          -c(1:n.batch)]))) {
        stop("The covariates are confounded. Please remove one or more of the \n             covariates so the design is not confounded.")
      }
      else {
        stop("At least one covariate is confounded with batch. Please remove \n             confounded covariates and rerun ComBat.")
      }
    }
  }
  if (verbose) 
    cat("[combat] Standardizing Data across features\n")
  if (!is.null(train)) {
    design_tr <- design[train, ]
    B.hat1 <- solve(crossprod(design_tr))
    B.hat1 <- tcrossprod(B.hat1, design_tr)
    B.hat <- tcrossprod(B.hat1, dat[, train])
  }
  else {
    B.hat1 <- solve(crossprod(design))
    B.hat1 <- tcrossprod(B.hat1, design)
    B.hat <- tcrossprod(B.hat1, dat)
  }
  grand.mean <- crossprod(n.batches/n.array, B.hat[1:n.batch, 
  ])
  var.pooled <- ((dat - t(design %*% B.hat))^2) %*% rep(1/n.array, 
                                                        n.array)
  stand.mean <- crossprod(grand.mean, t(rep(1, n.array)))
  if (!is.null(design)) {
    tmp <- design
    tmp[, c(1:n.batch)] <- 0
    stand.mean <- stand.mean + t(tmp %*% B.hat)
  }
  s.data <- (dat - stand.mean)/(tcrossprod(sqrt(var.pooled), 
                                           rep(1, n.array)))
  if (eb) {
    if (verbose) 
      cat("[combat] Fitting L/S model and finding priors\n")
  }
  else {
    if (verbose) 
      cat("[combat] Fitting L/S model\n")
  }
  batch.design <- design[, 1:n.batch]
  gamma.hat <- tcrossprod(solve(crossprod(batch.design, batch.design)), 
                          batch.design)
  gamma.hat <- tcrossprod(gamma.hat, s.data)
  delta.hat <- NULL
  for (i in batches) {
    delta.hat <- rbind(delta.hat, rowVars(s.data[, i], na.rm = TRUE))
  }
  gamma.star <- delta.star <- NULL
  gamma.bar <- t2 <- a.prior <- b.prior <- NULL
  if (eb) {
    gamma.bar <- rowMeans(gamma.hat)
    t2 <- rowVars(gamma.hat)
    a.prior <- apriorMat(delta.hat)
    b.prior <- bpriorMat(delta.hat)
    if (parametric) {
      if (verbose) 
        cat("[combat] Finding parametric adjustments\n")
      for (i in 1:n.batch) {
        temp <- it.sol(s.data[, batches[[i]]], gamma.hat[i, 
        ], delta.hat[i, ], gamma.bar[i], t2[i], a.prior[i], 
        b.prior[i])
        gamma.star <- rbind(gamma.star, temp[1, ])
        delta.star <- rbind(delta.star, temp[2, ])
      }
    }
    else {
      if (verbose) 
        cat("[combat] Finding non-parametric adjustments\n")
      for (i in 1:n.batch) {
        temp <- int.eprior(as.matrix(s.data[, batches[[i]]]), 
                           gamma.hat[i, ], delta.hat[i, ])
        gamma.star <- rbind(gamma.star, temp[1, ])
        delta.star <- rbind(delta.star, temp[2, ])
      }
    }
  }
  if (mean.only) {
    delta.star <- array(1, dim = dim(delta.star))
  }
  if (verbose) 
    cat("[combat] Adjusting the Data\n")
  bayesdata <- s.data
  j <- 1
  for (i in batches) {
    if (eb) {
      bayesdata[, i] <- (bayesdata[, i] - t(batch.design[i, 
      ] %*% gamma.star))/tcrossprod(sqrt(delta.star[j, 
      ]), rep(1, n.batches[j]))
    }
    else {
      bayesdata[, i] <- (bayesdata[, i] - t(batch.design[i, 
      ] %*% gamma.hat))/tcrossprod(sqrt(delta.hat[j, 
      ]), rep(1, n.batches[j]))
    }
    j <- j + 1
  }
  comdata <- (bayesdata * (tcrossprod(sqrt(var.pooled), rep(1, 
                                                            n.array)))) + stand.mean
  bayesdata <- bayesdata * (tcrossprod(sqrt(var.pooled), rep(1, 
                                                             n.array)))
  x_pc <- prcomp(t(bayesdata), center = TRUE, scale. = std.var)
  npc <- which(cumsum(x_pc$sdev^2/sum(x_pc$sdev^2)) > percent.var)[1]
  if (!is.null(n.pc)) {
    npc <- n.pc
  }
  scores <- x_pc$x[, 1:npc]
  scores_com <- combat(t(scores), bat, eb = score.eb, parametric = score.parametric)
  full_scores <- x_pc$x
  full_scores[, 1:npc] <- t(scores_com$dat.combat)
  if (std.var) {
    x.covbat <- t(full_scores %*% t(x_pc$rotation)) * matrix(x_pc$scale, 
                                                             dim(bayesdata)[1], dim(bayesdata)[2]) + matrix(x_pc$center, 
                                                                                                            dim(bayesdata)[1], dim(bayesdata)[2])
  }
  else {
    x.covbat <- t(full_scores %*% t(x_pc$rotation)) + matrix(x_pc$center, 
                                                             dim(bayesdata)[1], dim(bayesdata)[2])
  }
  if (resid == FALSE) {
    x.covbat <- x.covbat + stand.mean
  }
  return(list(dat.covbat = x.covbat, 
              combat.out = list(dat.combat = comdata, 
                                s.data = s.data, gamma.hat = gamma.hat, delta.hat = delta.hat, 
                                gamma.star = gamma.star, delta.star = delta.star, gamma.bar = gamma.bar, 
                                t2 = t2, a.prior = a.prior, b.prior = b.prior, batch = batch, 
                                mod = mod, stand.mean = stand.mean, stand.sd = sqrt(var.pooled)[, 1], B.hat = B.hat), 
              combat.scores = scores_com, 
              npc = npc, 
              x.pc = x_pc,
              bayesdata = bayesdata))
}
