vanilla_encoder <- nn_module(
  "Encoder",
  initialize = function(vae_dim) {
    self$n_layers = length(vae_dim)
    self$layers <- nn_module_list()
    self$layers[[1]] <- nn_linear(vae_dim[1], vae_dim[2])
    
    for (i in 2:(self$n_layers - 2)) {
      self$layers[[i]] <- nn_linear(vae_dim[i], vae_dim[i + 1])
    }
    
    self$e_mu <- nn_linear(vae_dim[self$n_layers - 1], vae_dim[self$n_layers])
    self$e_logvar <- nn_linear(vae_dim[self$n_layers - 1], vae_dim[self$n_layers])
  },
  
  forward = function(features) {
    tmp <- features
    for (i in 1:(self$n_layers - 2)) {
      tmp <- tmp %>% 
        self$layers[[i]]() %>% 
        nnf_relu()
    }
    
    # encode hidden layer to mean and variance vectors
    mu <- self$e_mu(tmp)
    logvar <- self$e_logvar(tmp)
    return(list(mu, logvar))
  })

vanilla_decoder <- nn_module(
  "Decoder",
  initialize = function(vae_dim) {
    self$inject_decoder = inject_decoder
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
    
  },
  forward = function(z, batch = NULL, decoder_covariates = NULL) {
    tmp <- torch_cat(list(z, batch, decoder_covariates), dim = -1)
    
    if (self$deep_inject) {
      for (i in 1:(self$n_layers - 2)) {
        tmp <- tmp %>% 
          self$layers[[i]]() %>% 
          nnf_relu()
        tmp <- torch_cat(list(tmp, batch), dim = -1)
      }
    } else {
      for (i in 1:(self$n_layers - 2)) {
        tmp <- tmp %>% 
          self$layers[[i]]() %>% 
          nnf_relu()
      }
    }
    
    
    output <- tmp %>% 
      self$layers[[self$n_layers - 1]]()
    
    return(output)
  })

vanilla_vae <- nn_module(
  "VAE", 
  initialize = function(feature_dim, latent_dim, 
                        n_hidden, n_batch, n_covariate, 
                        inject_decoder = FALSE, deep_inject = FALSE) {
    self$latent_dim <- latent_dim
    self$n_covariate <- n_covariate
    self$inject_decoder <- inject_decoder
    self$deep_inject <- deep_inject
    self$dims <- calculate_vae_dim(feature_dim, latent_dim, n_hidden)
    
    self$encoder <- vanilla_encoder(self$dims, n_batch, n_covariate)
    self$decoder <- vanilla_decoder(self$dims, n_batch, n_covariate, 
                                    self$inject_decoder, self$deep_inject)
  },
  forward = function(item_list) {
    covariates <- NULL
    decoder_covariates <- NULL
    if (self$n_covariate != 0) {
      covariates <- item_list[[2]]
    }
    if (self$inject_decoder) {
      decoder_covariates = covariates
    }
    
    features <- item_list[[1]]
    batch <- item_list[[3]]
    
    # encode features to latent feature distributions
    latent_dist <- self$encoder(features = features, batch = batch, 
                                covariates = covariates)
    feat_mu <- latent_dist[[1]]
    feat_logvar <- latent_dist[[2]]
    
    # sample from latent distribution with re-parameterization trick
    feat_z <- feat_mu + torch_exp(feat_logvar$mul(0.5)) * torch_randn(self$latent_dim)
    
    feat_reconstructed <- self$decoder(z = feat_z, batch = batch,
                                       decoder_covariates = decoder_covariates)
    
    return(list(feat_recon = feat_reconstructed, 
                feat_mu = feat_mu, 
                feat_logvar = feat_logvar, 
                feat_z = feat_z))
  },
  encode_decode = function(torch_ds, raw_means, raw_sds) {
    covariates <- NULL
    decoder_covariates <- NULL
    if (self$n_covariate != 0) {
      covariates <- torch_ds$covariates
    }
    if (self$inject_decoder) {
      decoder_covariates <- covariates
    }
    
    raw_means <- torch_tensor(raw_means)
    raw_sds <- torch_tensor(raw_sds)
    
    latent_dist <- self$encoder(features = torch_ds$data_raw, batch = torch_ds$batch,
                                covariates = covariates)
    feat_mu <- latent_dist[[1]]
    feat_logvar <- latent_dist[[2]]
    
    feat_reconstructed <- self$decoder(z = feat_mu, batch = torch_ds$batch,
                                       decoder_covariates = decoder_covariates) * raw_sds + raw_means
    feat_restyled <- self$decoder(z = feat_mu, batch = torch_ds$new_batch,
                                  decoder_covariates = decoder_covariates) * raw_sds + raw_means
    feat_resids <- torch_ds$data_raw * raw_sds + raw_means - feat_reconstructed
    
    combat_mu <- neuroCombat(dat = t(as.matrix(feat_resids)),
                             batch = as.matrix(torch_ds$batch),
                             mod = as.matrix(torch_ds$covariates),
                             ref.batch = as.numeric(torch_ds$new_batch[1]),
                             verbose = FALSE)$dat.combat %>% 
      t() %>% torch_tensor()
    feat_restyled_combat <- neuroCombat(dat = t(as.matrix(feat_resids)),
                                        batch = as.matrix(torch_ds$batch),
                                        mod = as.matrix(torch_ds$covariates),
                                        ref.batch = as.numeric(torch_ds$new_batch[1]),
                                        verbose = FALSE)$dat.combat %>% 
      t() %>% torch_tensor()
    
    return(list(recon = feat_reconstructed,
                restyle = feat_restyled, 
                resids = feat_resids,
                resids_restyle = feat_restyled + feat_resids,
                combat = feat_combat,
                combat_restyle = feat_restyled + feat_combat,
                latent_mu = feat_mu,
                latent_logvar = feat_logvar))
  })