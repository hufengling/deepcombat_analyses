vanilla_encoder <- nn_module(
  "Encoder",
  initialize = function(vae_dim, n_batch, n_covariate) {
    self$n_layers = length(vae_dim)
    self$layers <- nn_module_list()
    self$layers[[1]] <- nn_linear(vae_dim[1] + n_batch + n_covariate, vae_dim[2])
    
    if (self$n_layers > 3) {
      for (i in 2:(self$n_layers - 2)) {
        self$layers[[i]] <- nn_linear(vae_dim[i], vae_dim[i + 1])
      }
    }
    
    self$e_mu <- nn_linear(vae_dim[self$n_layers - 1], vae_dim[self$n_layers])
    self$e_logvar <- nn_linear(vae_dim[self$n_layers - 1], vae_dim[self$n_layers])
  },
  
  forward = function(features, batch = NULL, covariates = NULL) {
    tmp <- torch_cat(list(features, batch, covariates), dim = -1)
    for (i in 1:(self$n_layers - 2)) {
      tmp <- tmp %>% 
        self$layers[[i]]() %>% 
        torch_tanh()
    }
    
    # encode hidden layer to mean and variance vectors
    mu <- self$e_mu(tmp)
    logvar <- self$e_logvar(tmp)
    return(list(mu, logvar))
  })

vanilla_decoder <- nn_module(
  "Decoder",
  initialize = function(vae_dim, n_batch, n_covariate, 
                        inject_decoder = FALSE, inject_last = FALSE, deep_inject = FALSE) {
    self$inject_decoder = inject_decoder
    self$inject_last = inject_last
    self$deep_inject = deep_inject
    self$n_layers = length(vae_dim)
    self$layers <- nn_module_list()
    
    if (self$inject_decoder) {
      self$layers[[1]] <- nn_linear(vae_dim[self$n_layers] + n_batch + n_covariate, 
                                    vae_dim[self$n_layers - 1])
    } else {
      self$layers[[1]] <- nn_linear(vae_dim[self$n_layers] + n_batch, 
                                    vae_dim[self$n_layers - 1])
    }
    
    if (self$deep_inject) {
      for (i in 2:(self$n_layers - 1)) {
        self$layers[[i]] <- nn_linear(vae_dim[self$n_layers + 1 - i] + n_batch, 
                                      vae_dim[self$n_layers - i])
      }
    } else {
      for (i in 2:(self$n_layers - 1)) {
        self$layers[[i]] <- nn_linear(vae_dim[self$n_layers + 1 - i], 
                                      vae_dim[self$n_layers - i])
      }
    }
    
    if (self$inject_last) {
      self$layers[[self$n_layers - 1]] <- nn_linear(vae_dim[2] + n_batch + n_covariate, 
                                                    vae_dim[1])
    }
  },
  forward = function(z, batch = NULL, decoder_covariates = NULL) {
    if (self$inject_decoder) {
      tmp <- torch_cat(list(z, batch, decoder_covariates), dim = -1)
    } else {
      tmp <- torch_cat(list(z, batch), dim = -1)
    }
    
    if (self$deep_inject) {
      for (i in 1:(self$n_layers - 2)) {
        tmp <- tmp %>% 
          self$layers[[i]]() %>% 
          torch_tanh()
        tmp <- torch_cat(list(tmp, batch), dim = -1)
      }
    } else {
      for (i in 1:(self$n_layers - 2)) {
        tmp <- tmp %>% 
          self$layers[[i]]() %>% 
          torch_tanh()
      }
    }
    
    if (self$inject_last) {
      tmp <- torch_cat(list(tmp, decoder_covariates), dim = -1)
    }
    
    output <- tmp %>% 
      self$layers[[self$n_layers - 1]]()
    
    return(output)
  })

vanilla_vae <- nn_module(
  "VAE", 
  initialize = function(feature_dim, latent_dim, 
                        n_hidden, n_batch, n_covariate, 
                        inject_decoder = FALSE, inject_last = FALSE, deep_inject = FALSE,
                        rescale = FALSE, rescale_n_batch = NULL) {
    self$latent_dim <- latent_dim
    self$n_batch <- n_batch
    self$n_covariate <- n_covariate
    self$inject_decoder <- inject_decoder
    self$inject_last <- inject_last
    self$deep_inject <- deep_inject
    self$dims <- calculate_vae_dim(feature_dim, latent_dim, n_hidden)
    self$rescale <- rescale
    
    self$encoder <- vanilla_encoder(self$dims, n_batch, n_covariate)
    self$decoder <- vanilla_decoder(self$dims, n_batch, n_covariate, 
                                    self$inject_decoder, self$inject_last, self$deep_inject)
    if (self$rescale) {
      self$rescaler <- nn_linear(rescale_n_batch, self$dims[1], bias = FALSE)
    }
  },
  forward = function(item_list) {
    covariates <- NULL
    decoder_covariates <- NULL
    if (self$n_covariate != 0) {
      covariates <- item_list[[2]]
    }
    if (self$inject_decoder | self$inject_last) {
      decoder_covariates = covariates
    }
    if (self$n_batch != 0) {
      batch <- item_list[[3]]
    } else {
      batch <- NULL
    }
    
    features <- item_list[[1]]
    
    # encode features to latent feature distributions
    latent_dist <- self$encoder(features = features, batch = batch, 
                                covariates = covariates)
    feat_mu <- latent_dist[[1]]
    feat_logvar <- latent_dist[[2]]
    
    # sample from latent distribution with re-parameterization trick
    feat_z <- feat_mu + torch_exp(feat_logvar * 0.5) * torch_randn_like(feat_mu)
    
    feat_reconstructed <- self$decoder(z = feat_z, batch = batch,
                                       decoder_covariates = decoder_covariates)
    if (self$rescale) {
      feat_reconstructed <- feat_reconstructed * self$rescaler(item_list[[3]] - 0.5)$exp()
    }
    
    return(list(feat_recon = feat_reconstructed, 
                feat_mu = feat_mu, 
                feat_logvar = feat_logvar, 
                feat_z = feat_z))
  },
  encode_decode = function(torch_ds, raw_means, raw_sds,
                           correct = c("combat", "combat"),
                           mean_only = c(FALSE, FALSE),
                           use_covariates = c(TRUE, TRUE),
                           ref_batch = NULL,
                           verbose = FALSE) {
    ## Check parameters
    correct[1] <- match.arg(correct[1], c("combat", "covbat"))
    correct[2] <- match.arg(correct[2], c("combat", "covbat", "scale", "none"))
    
    if (!is.null(ref_batch)) {
      if (!ref_batch %in% unique(as.matrix(torch_ds$batch))) {
        stop("ref_batch must be either NULL or one of the batches in the dataset")
      }
    }
    
    ## Setup
    covariates <- NULL
    decoder_covariates <- NULL
    raw_means <- torch_tensor(raw_means)
    raw_sds <- torch_tensor(raw_sds)
    if (self$n_covariate != 0) {
      covariates <- torch_ds$covariates
    }
    if (self$inject_decoder | self$inject_last) {
      decoder_covariates <- covariates
    }
    if (self$n_batch != 0) {
      batch <- torch_ds$batch
      if (is.null(ref_batch)) {
        new_batch <- torch_ds$new_batch
      } else {
        new_batch <- ref_batch * torch_ones_like(torch_ds$new_batch)
      }
    } else {
      batch <- NULL
      new_batch <- NULL
    }
    
    if (verbose) {
      if (is.null(ref_batch)) {
        cat("Harmonizing to intermediate space\n")
      } else {
        cat(paste0("Harmonizing to reference batch ", ref_batch, "\n"))
      }
    }
    
    ## Get latent space and correct
    latent_dist <- self$encoder(features = torch_ds$data, batch = batch,
                                covariates = covariates)
    feat_mu <- latent_dist[[1]]
    feat_logvar <- latent_dist[[2]]
    
    if (verbose) {
      cat("Harmonizing latent space\n")
    }
    if (correct[1] == "combat") {
      if (use_covariates[1]) {
        corrected_mu <- neuroCombat(dat = t(as.matrix(feat_mu)),
                                    batch = as.matrix(torch_ds$batch),
                                    mod = as.matrix(torch_ds$covariates),
                                    ref.batch = ref_batch,
                                    mean.only = mean_only[1],
                                    verbose = verbose)$dat.combat %>% 
          t() %>% torch_tensor()
      } else {
        corrected_mu <- neuroCombat(dat = t(as.matrix(feat_mu)),
                                    batch = as.matrix(torch_ds$batch),
                                    ref.batch = ref_batch,
                                    mean.only = mean_only[1],
                                    verbose = verbose)$dat.combat %>% 
          t() %>% torch_tensor()
      }
    }
    # if (correct[1] == "covbat") {
    #   cat("Harmonizing latent space with CovBat\n")
    #   if (!is.null(ref_batch)) {
    #     stop("Cannot yet run CovBat with reference batch setting.")
    #   }
    #   if (use_covariates[1]) {
    #     corrected_mu <- covbat(dat = t(as.matrix(feat_mu)),
    #                            bat = as.matrix(torch_ds$batch),
    #                            mod = as.matrix(torch_ds$covariates),
    #                            mean.only = mean_only[1],
    #                            verbose = verbose)$dat.covbat %>% 
    #       t() %>% torch_tensor()
    #   } else {
    #     corrected_mu <- covbat(dat = t(as.matrix(feat_mu)),
    #                            bat = as.matrix(torch_ds$batch),
    #                            mean.only = mean_only[1],
    #                            verbose = verbose)$dat.covbat %>% 
    #       t() %>% torch_tensor()
    #   }
    # }
    
    ## Decode latent space
    if (verbose) {
      cat("Decoding harmonized latent space\n")
    }
    # if (self$rescale) {
    #   feat_reconstructed <- self$decoder(z = feat_mu, batch = batch,
    #                                      decoder_covariates = decoder_covariates) * self$rescaler(torch_ds$batch - 0.5)$exp() * raw_sds + raw_means
    #   feat_restyled <- self$decoder(z = feat_mu, batch = new_batch,
    #                                 decoder_covariates = decoder_covariates) * self$rescaler(torch_ds$new_batch)$exp() * raw_sds + raw_means
    #   feat_combat_restyled <- self$decoder(z = corrected_mu, batch = new_batch,
    #                                        decoder_covariates = decoder_covariates) * self$rescaler(torch_ds$new_batch)$exp() * raw_sds + raw_means
    # } else {
      feat_reconstructed <- self$decoder(z = feat_mu, batch = batch,
                                         decoder_covariates = decoder_covariates) * raw_sds + raw_means
      feat_restyled <- self$decoder(z = feat_mu, batch = new_batch,
                                    decoder_covariates = decoder_covariates) * raw_sds + raw_means
      feat_combat_restyled <- self$decoder(z = corrected_mu, batch = new_batch, 
                                           decoder_covariates = decoder_covariates) * raw_sds + raw_means
    # }
    
    ## Correct residuals from autoencoder
    feat_resids <- torch_ds$data * raw_sds + raw_means - feat_reconstructed
    if (correct[2] == "combat") {
      if (verbose) {
        cat("Harmonizing autoencoder residuals using ComBat\n")
      }
      if (use_covariates[2]) {
        corrected_resids <- neuroCombat(dat = t(as.matrix(feat_resids)),
                                        batch = as.matrix(torch_ds$batch),
                                        mod = as.matrix(torch_ds$covariates),
                                        ref.batch = ref_batch,
                                        mean.only = mean_only[2],
                                        verbose = verbose)$dat.combat %>% 
          t() %>% torch_tensor()
      } else {
        corrected_resids <- neuroCombat(dat = t(as.matrix(feat_resids)),
                                        batch = as.matrix(torch_ds$batch),
                                        ref.batch = ref_batch,
                                        mean.only = mean_only[2],
                                        verbose = verbose)$dat.combat %>% 
          t() %>% torch_tensor()
      }
    }
    # if (correct[2] == "covbat") {
    #   if (verbose) {
    #     cat("Harmonizing autoencoder residuals using CovBat\n")
    #   }
    #   if (use_covariates[2]) {
    #     corrected_resids <- covbat(dat = t(as.matrix(feat_resids)),
    #                                bat = as.matrix(torch_ds$batch),
    #                                mod = as.matrix(torch_ds$covariates),
    #                                mean.only = mean_only[1],
    #                                verbose = verbose)$dat.covbat %>% 
    #       t() %>% torch_tensor()
    #   } else {
    #     corrected_resids <- covbat(dat = t(as.matrix(feat_resids)),
    #                                bat = as.matrix(torch_ds$batch),
    #                                mean.only = mean_only[1],
    #                                verbose = verbose)$dat.covbat %>% 
    #       t() %>% torch_tensor()
    #   }
    # }
    # if (correct[2] == "scale") {
    #   if (verbose) {
    #     cat("Harmonizing autoencoder residuals using scale only\n")
    #   }
    #   feat_resids_mat <- as.matrix(feat_resids)
    #   batch_mat <- as.matrix(torch_ds$batch)
    #   unique_batches <- unique(batch_mat)
    #   
    #   if (is.null(ref_batch)) {
    #     reference_sds <- colSds(feat_resids_mat)
    #   }
    #   else {
    #     in_batch <- batch_mat == ref_batch
    #     reference_sds <- colSds(feat_resids_mat[in_batch, ])
    #   }
    #   
    #   for (i in 1:length(unique_batches)) {
    #     in_batch <- batch_mat == unique_batches[i]
    #     n_in_batch <- sum(in_batch)
    #     
    #     tmp_sds <- colSds(feat_resids_mat[in_batch, ])
    #     rescale_factor <- matrix(reference_sds / tmp_sds, 
    #                              nrow = n_in_batch, ncol = length(tmp_sds), 
    #                              byrow = TRUE)
    #     
    #     feat_resids_mat[in_batch, ] <- feat_resids_mat[in_batch, ] * rescale_factor
    #   }
    #   
    #   corrected_resids <- torch_tensor(feat_resids_mat)
    # }
    # if (correct[2] == "none") {
    #   if (verbose) {
    #     cat("Not harmonizing autoencoder residuals\n")
    #   }
    #   corrected_resids <- feat_resids
    # }
    
    return(list(recon = feat_reconstructed,
                resids = corrected_resids,
                restyle = feat_restyled, 
                resids_restyle = feat_restyled + corrected_resids,
                combat = feat_combat_restyled,
                combat_restyle = feat_combat_restyled + corrected_resids,
                combat_mu = corrected_mu,
                latent_mu = feat_mu,
                latent_logvar = feat_logvar))
  },
  get_vae_transforms = function(output) {
    recon <- output$feat_recon
    mu <- output$feat_mu
    mu_2 <- mu$pow(2)
    logvar <- output$feat_logvar
    var <- logvar$exp()
    inv_var <- (-logvar)$exp()
    inv_var_t <- torch_transpose(inv_var, 1, 2)
    n_latent <- dim(mu)[2]
    n_minibatch <- dim(mu)[1]
    
    return(list(recon = recon,
                mu = mu,
                mu_2 = mu_2,
                logvar = logvar,
                var = var,
                inv_var = inv_var,
                inv_var_t = inv_var_t,
                n_latent = n_latent,
                n_minibatch = n_minibatch))
  },
  # get_covariate_similarity = function(item, similarity_type = c("cosine", "correlation", "ones")) {
  #   if (similarity_type == "cosine") {
  #     scaled_covariate <- item[[2]] / torch_sum(item[[2]]^2, 2, keepdim = TRUE)^0.5
  #     covariate_similarity <- torch_matmul(scaled_covariate, torch_transpose(scaled_covariate, 1, 2))
  #   }
  #   if (similarity_type == "correlation") {
  #     covariate_similarity <- torch_tensor((cor(t(as.matrix(item[[2]]))))) * 0.5 + 0.5
  #   }
  #   if (similarity_type == "ones") {
  #     covariate_similarity <- torch_ones(c(dim(item[[2]])[1], dim(item[[2]])[1]))
  #   }
  #   
  #   return(covariate_similarity)
  # },
  get_recon_loss = function(transform_list, target,
                            recon_weights,
                            recon_type = c("mse", "nll")) {
    if (recon_type == "mse")
      loss_recon <- nn_mse_loss(reduction = "sum")(transform_list$recon, target)
    # if (recon_type == "nll") {
    #   log_sigma <- torch_log(torch_std(transform_list$recon - target))
    #   nll_tensor <- torch_pow((transform_list$recon - target) / exp(log_sigma), 2) + log_sigma# + 0.5 * log(2 * pi)
    #   loss_recon <- torch_sum(recon_weights * nll_tensor) # reweight so recon of both batches is equally prioritized
    # }
    
    loss_recon
  },
  get_prior_loss = function(transform_list,
                            recon_weights) {
    return(0.5 * torch_sum(recon_weights * (transform_list$mu_2 + transform_list$var - 1 - transform_list$logvar)))
  })#,
  # get_pairwise_loss = function(transform_list, pairwise_type = c("KL", "MMD", "none")) {
  #   ## Pairwise KL divergence between subjects from different batch
  #   # https://stats.stackexchange.com/questions/462331/efficiently-computing-pairwise-kl-divergence-between-multiple-diagonal-covarianc
  #   
  #   if (pairwise_type == "KL") {
  #     xi_varj_xi <- torch_matmul(transform_list$mu_2, transform_list$inv_var_t)
  #     # xi_vari_xi <- torch_diag(xi_varj_xi) * ones
  #     # xi_varj_xj <- torch_matmul(mu * inv_var, torch_transpose(mu, 1, 2))
  #     #trace_cov_prod <- torch_matmul(var, inv_var_t)
  #     pairwise_mat <- 0.5 * (xi_varj_xi -
  #                              2 * torch_matmul(transform_list$mu, 
  #                                               transform_list$inv_var_t * 
  #                                                 torch_transpose(transform_list$mu, 1, 2)) + #xi_varj_xj
  #                              (torch_diag(xi_varj_xi)) * torch_ones_like(xi_varj_xi) + #xi_vari_xi
  #                              (torch_matmul(transform_list$inv_var, 
  #                                            torch_transpose(transform_list$var, 1, 2)) - transform_list$n_latent)) #trace_cov_prod - k
  #   }
  #   if (pairwise_type == "MMD") {
  #     xi_xi <- torch_matmul(transform_list$mu_2, torch_ones_like(transform_list$inv_var_t))
  #     pairwise_mat <- 0.5 * (xi_xi - 2 * torch_matmul(transform_list$mu, 
  #                                                     torch_transpose(transform_list$mu, 1, 2)) + torch_transpose(xi_xi, 1, 2))
  #   }
  #   
  #   if (pairwise_type == "none") {
  #     pairwise_mat <- torch_zeros(c(transform_list$n_minibatch, 
  #                                   transform_list$n_minibatch))
  #   }
  #   return(pairwise_mat)
  # })