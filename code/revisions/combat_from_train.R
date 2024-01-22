combat_from_train <- function (dat, batch, mod = NULL, estimates = NULL, 
                               verbose = FALSE) {
  dat <- as.matrix(dat)
  batch <- as.factor(batch)
  n.array <- length(batch)
  n.batch <- nlevels(batch)
  batches <- lapply(levels(batch), function(x) which(batch == x))
  n.batches <- sapply(batches, length)
  
  design <- cbind(model.matrix(~-1 + batch), mod)
  
  batch_train <- as.factor(estimates$batch)
  batches_train <- lapply(levels(batch_train), function(x) which(batch_train == x))
  n.proportions_train <- sapply(batches_train, length) / length(batch_train)
  
  B.hat <- estimates$beta.hat
  var.pooled <- estimates$var.pooled
  
  if (!is.null(design)) {
    tmp <- design
    if (is.null(estimates$ref.batch)) {
      tmp[, c(1:n.batch)] <- matrix(n.proportions_train, 
                                    nrow = nrow(design), 
                                    ncol = n.batch, byrow = T)
    } else {
      tmp[, c(1:n.batch)] <- 0
      tmp[, which(levels(batch) == estimates$ref.batch)] <- 1
    }
    stand.mean <- t(tmp %*% B.hat)
  }
  s.data <- (dat - stand.mean)/(tcrossprod(sqrt(var.pooled), 
                                           rep(1, n.array)))
  batch.design <- design[, 1:n.batch]
  gamma.star <- estimates$gamma.star
  delta.star <- estimates$delta.star
  
  bayesdata <- s.data
  j <- 1
  for (i in batches) {
    bayesdata[, i] <- (bayesdata[, i] - t(batch.design[i, ] %*% gamma.star)) / 
      tcrossprod(sqrt(delta.star[j, ]), rep(1, n.batches[j]))
    j <- j + 1
  }
  bayesdata <- (bayesdata * (tcrossprod(sqrt(var.pooled), rep(1, n.array)))) + stand.mean
  
  return(list(dat.combat = bayesdata, stand.mean = stand.mean, estimates = estimates))
}

covbat_from_train <- function (dat, batch, mod = NULL, percent.var = 0.95,
                               covbat_train,
                               verbose = FALSE) {
  combat_estimates <- covbat_train$combat.out
  combat_estimates$beta.hat <- combat_estimates$B.hat
  combat_estimates$ref.batch <- NULL
  combat_estimates$var.pooled <- combat_estimates$stand.sd^2
  combat_train <- combat_from_train(dat = dat, batch = batch, 
                                    mod = mod, estimates = combat_estimates)
  comdata <- combat_train$dat.combat
  bayesdata <- comdata - combat_train$stand.mean
  
  x_pc <- covbat_train$x.pc
  npc <- covbat_train$npc
  pc_scores <- t((bayesdata - x_pc$center) / x_pc$scale) %*% x_pc$rotation
  scores <- pc_scores[, 1:npc]
  
  combat_score_estimates <- covbat_train$combat.scores
  combat_score_estimates$beta.hat <- combat_score_estimates$B.hat
  combat_score_estimates$ref.batch <- NULL
  combat_score_estimates$var.pooled <- combat_score_estimates$stand.sd^2
  combat_score_estimates$gamma.star <- combat_score_estimates$gamma.hat
  combat_score_estimates$delta.star <- combat_score_estimates$delta.hat
  combat_scores_train <- combat_from_train(t(scores), batch, estimates = combat_score_estimates)
  
  full_scores <- pc_scores
  full_scores[, 1:npc] <- t(combat_scores_train$dat.combat)
  x.covbat <- t(full_scores %*% t(x_pc$rotation)) * matrix(x_pc$scale, 
                                                           dim(bayesdata)[1], 
                                                           dim(bayesdata)[2]) + 
    matrix(x_pc$center, dim(bayesdata)[1], dim(bayesdata)[2]) + combat_train$stand.mean
  return(list(dat.covbat = x.covbat, 
              combat.out = list(dat.combat = comdata, 
                                s.data = s.data, 
                                gamma.hat = gamma.hat, delta.hat = delta.hat, 
                                gamma.star = gamma.star, delta.star = delta.star, 
                                gamma.bar = gamma.bar, 
                                t2 = t2, a.prior = a.prior, b.prior = b.prior, batch = batch, 
                                mod = mod, 
                                stand.mean = stand.mean, stand.sd = combat_estimates$stand.sd, 
                                B.hat = B.hat), 
              combat.scores = combat_scores_train, npc = npc, 
              x.pc = x_pc))
}