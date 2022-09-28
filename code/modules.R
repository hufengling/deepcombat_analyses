ae_encoder <- nn_module(
  "Encoder",
  initialize = function(vae_dim, n_batch, n_covariate) {
    self$n_layers = length(vae_dim)
    self$layers <- nn_module_list()
    self$layers[[1]] <- nn_linear(vae_dim[1] + n_batch + n_covariate, vae_dim[2])
    
    if (self$n_layers > 3) {
      for (i in 2:(self$n_layers - 1)) {
        self$layers[[i]] <- nn_linear(vae_dim[i], vae_dim[i + 1])
      }
    }
  },
  
  forward = function(features, batch = NULL, covariates = NULL) {
    tmp <- torch_cat(list(features, batch, covariates), dim = -1)
    for (i in 1:(self$n_layers - 2)) {
      tmp <- tmp %>% 
        self$layers[[i]]() %>% 
        torch_tanh()
    }
    
    # encode hidden layer to mean and variance vectors
    z <- self$layers[[self$n_layers - 1]](tmp)
    return(z)
  })

ae_decoder <- nn_module(
  "Decoder",
  initialize = function(vae_dim) {
    self$n_layers = length(vae_dim)
    self$layers <- nn_module_list()
    
    for (i in 1:(self$n_layers - 1)) {
      self$layers[[i]] <- nn_linear(vae_dim[self$n_layers + 1 - i], 
                                    vae_dim[self$n_layers - i])
    }
  },
  forward = function(z) {
    tmp <- z
    for (i in 1:(self$n_layers - 2)) {
      tmp <- tmp %>% 
        self$layers[[i]]() %>% 
        torch_tanh()
    }
    
    output <- tmp %>% 
      self$layers[[self$n_layers - 1]]()
    
    return(output)
  })

ae <- nn_module(
  "AE", 
  initialize = function(feature_dim, latent_dim, 
                        n_hidden, n_batch, n_covariate) {
    self$latent_dim <- latent_dim
    self$n_batch <- n_batch
    self$n_covariate <- n_covariate
    self$dims <- calculate_vae_dim(feature_dim, latent_dim, n_hidden)
    
    self$encoder <- ae_encoder(self$dims, self$n_batch, n_covariate)
    self$decoder <- ae_decoder(self$dims)
  },
  forward = function(item_list) {
    covariates <- NULL
    if (self$n_covariate != 0) {
      covariates <- item_list[[2]]
    }
    if (self$n_batch != 0) {
      batch <- item_list[[3]]
    } else {
      batch <- NULL
    }
    
    features <- item_list[[1]]
    
    # encode features to latent feature distributions
    feat_z <- self$encoder(features = features, batch = batch, 
                           covariates = covariates)
    feat_reconstructed <- self$decoder(z = feat_z)
    
    return(list(feat_recon = feat_reconstructed,
                feat_z = feat_z))
  },
  encode_decode = function(torch_ds, raw_means, raw_sds) {
    covariates <- NULL
    if (self$n_covariate != 0) {
      covariates <- torch_ds$covariates
    }
    if (self$n_batch != 0) {
      batch <- torch_ds$batch
      new_batch <- torch_ds$new_batch
    } else {
      batch <- NULL
      new_batch <- NULL
    }
    
    raw_means <- torch_tensor(raw_means)
    raw_sds <- torch_tensor(raw_sds)
    
    feat_z <- self$encoder(features = torch_ds$data_raw, batch = batch,
                           covariates = covariates)
    
    combat_z <- neuroCombat(dat = t(as.matrix(feat_z)),
                             batch = as.matrix(torch_ds$batch),
                             mod = as.matrix(torch_ds$covariates),
                             ref.batch = as.numeric(torch_ds$new_batch[1]),
                             verbose = FALSE)$dat.combat %>% 
      t() %>% torch_tensor()
    
    feat_reconstructed <- self$decoder(z = feat_z) * raw_sds + raw_means
    feat_restyled <- self$decoder(z = combat_z) * raw_sds + raw_means
    feat_resids <- torch_ds$data_raw * raw_sds + raw_means - feat_reconstructed
    
    return(list(recon = feat_reconstructed,
                restyle = feat_restyled, 
                restyle_resids = feat_restyled + feat_resids,
                latent = feat_z))
  })