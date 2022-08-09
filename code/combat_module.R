vanilla_encoder <- nn_module(
  "Encoder",
  initialize = function(vae_dim, n_batch, n_covariate) {
    self$n_layers = length(vae_dim)
    self$layers <- nn_module_list()
    self$layers[[1]] <- nn_linear(vae_dim[1] + n_batch + n_covariate, vae_dim[2])
    
    for (i in 2:(self$n_layers - 2)) {
      self$layers[[i]] <- nn_linear(vae_dim[i], vae_dim[i + 1])
    }
    
    self$e_mu <- nn_linear(vae_dim[self$n_layers - 1], vae_dim[self$n_layers])
    self$e_logvar <- nn_linear(vae_dim[self$n_layers - 1], vae_dim[self$n_layers])
  },
  
  forward = function(features, batch = NULL, covariates = NULL) {
    tmp <- torch_cat(list(features, batch, covariates), dim = -1)
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
  initialize = function(vae_dim, n_batch, n_covariate, inject_decoder = FALSE) {
    self$n_layers = length(vae_dim)
    self$layers <- nn_module_list()
    if (inject_decoder) {
      self$layers[[1]] <- nn_linear(vae_dim[self$n_layers] + n_batch + n_covariate, 
                                    vae_dim[self$n_layers - 1])
    } else {
      self$layers[[1]] <- nn_linear(vae_dim[self$n_layers] + n_batch, 
                                  vae_dim[self$n_layers - 1])
    }
    
    for (i in 2:(self$n_layers - 1)) {
      self$layers[[i]] <- nn_linear(vae_dim[self$n_layers + 1 - i], vae_dim[self$n_layers - i])
    }
  },
  
  forward = function(z, batch = NULL, decoder_covariates = NULL) {
    tmp <- torch_cat(list(z, batch, decoder_covariates), dim = -1)
    
    for (i in 1:(self$n_layers - 2)) {
      tmp <- tmp %>% 
        self$layers[[i]]() %>% 
        nnf_relu()
    }
    
    output <- tmp %>% 
      self$layers[[self$n_layers - 1]]()
    
    return(output)
  })

vanilla_vae <- nn_module(
  "VAE", 
  initialize = function(feature_dim, latent_dim, n_hidden, n_batch, n_covariate, inject_decoder = FALSE) {
    self$latent_dim <- latent_dim
    self$n_covariate <- n_covariate
    self$inject_decoder <- inject_decoder
    self$dims <- calculate_vae_dim(feature_dim, latent_dim, n_hidden)
    
    self$encoder <- vanilla_encoder(self$dims, n_batch, n_covariate)
    self$decoder <- vanilla_decoder(self$dims, n_batch, n_covariate, self$inject_decoder)
  },
  forward = function(item_list) {
    covariates <- NULL
    decoder_covariates <- NULL
    if (self$n_covariate != 0) {
      covariates <- item_list[[3]]
    }
    if (self$inject_decoder) {
      decoder_covariates = covariates
    }
    
    features <- item_list[[1]]
    batch <- item_list[[4]]
    
    latent_dist <- self$encoder(features = features, batch = batch, 
                                covariates = covariates) # encode features to latent feature distributions
    feat_mu <- latent_dist[[1]]
    feat_logvar <- latent_dist[[2]]
    feat_z <- feat_mu + torch_exp(feat_logvar$mul(0.5)) * torch_randn(self$latent_dim) # reparameterization trick
    
    feat_reconstructed <- self$decoder(z = feat_z, batch = batch,
                                       decoder_covariates = decoder_covariates)
    
    return(list(feat_recon = feat_reconstructed, feat_mu = feat_mu, feat_logvar = feat_logvar))
  },
  
  sample_from_latent = function(torch_ds, decode_from_new = FALSE, n_samples) {
    covariates <- NULL
    decoder_covariates <- NULL
    
    if (self$n_covariate != 0) {
      covariates <- torch_ds$covariates
    }
    if (self$inject_decoder) {
      decoder_covariates <- covariates
    }

    latent_dist <- self$encoder(features = torch_ds$data_raw, batch = torch_ds$batch,
                                covariates = covariates)
    feat_mu <- latent_dist[[1]]
    feat_logvar <- latent_dist[[2]]
    
    sum_sample <- torch_zeros_like(torch_ds$data_raw)
    for (i in 1:n_samples) {
      feat_z <- feat_mu + torch_exp(feat_logvar$mul(0.5)) * torch_randn(self$latent_dim)
      
      if (decode_from_new) {
        sum_sample <- sum_sample + self$decoder(z = feat_mu, batch = torch_ds$new_batch,
                                                decoder_covariates = decoder_covariates)
      }
      else {
        sum_sample <- sum_sample + self$decoder(z = feat_mu, batch = torch_ds$batch,
                                                decoder_covariates = decoder_covariates)
      }
    }
    mean_sample <- sum_sample / n_samples
    
    return(list(feat_recon = mean_sample, feat_mu = feat_mu))
  },
  encode_decode = function(torch_ds, decode_from_new = FALSE) {
    covariates <- NULL
    decoder_covariates <- NULL
    if (self$n_covariate != 0) {
      covariates <- torch_ds$covariates
    }
    if (self$inject_decoder) {
      decoder_covariates <- covariates
    }
    
    latent_dist <- self$encoder(features = torch_ds$data_raw, batch = torch_ds$batch,
                                covariates = covariates)
    feat_mu <- latent_dist[[1]]
    
    if (decode_from_new) {
      feat_reconstructed <- self$decoder(z = feat_mu, batch = torch_ds$new_batch,
                                         decoder_covariates = decoder_covariates)
    }
    else {
      feat_reconstructed <- self$decoder(z = feat_mu, batch = torch_ds$batch,
                                         decoder_covariates = decoder_covariates)
    }
    
    return(list(feat_recon = feat_reconstructed, feat_mu = feat_mu))
  },
  
  decode_from_latent = function(latent_tensor, batch) {
    self$decoder(z = latent_tensor, batch = batch)
  }
)
