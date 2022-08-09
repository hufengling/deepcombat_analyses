encoder <- nn_module(
  "Encoder",
  initialize = function(vae_dim, inject_batch = FALSE, n_batch) {
    self$inject_batch = inject_batch
    if (inject_batch & n_batch == 0) {
      stop("Number of batches must be greater than 0")
    }
    if (!inject_batch) {
      n_batch = 0
    }
    
    self$e1 <- nn_linear(vae_dim[1] + n_batch, vae_dim[2])
    self$e2 <- nn_linear(vae_dim[2] + n_batch, vae_dim[3])
    self$e_mu <- nn_linear(vae_dim[3] + n_batch, vae_dim[4])
    self$e_logvar <- nn_linear(vae_dim[3] + n_batch, vae_dim[4])
  },
  forward = function(x, batch = NULL) {
    if (self$inject_batch) {
      hidden1 <- torch_cat(list(x, batch), dim = -1) %>% 
        self$e1() %>% 
        nnf_relu()
      hidden2 <- torch_cat(list(hidden1, batch), dim = -1) %>% 
        self$e2() %>% 
        nnf_relu()
      hidden3 <- torch_cat(list(hidden2, batch), dim = -1) # calculate hidden layer
      
    } else {
      hidden3 <- x %>% 
        self$e1() %>% 
        nnf_relu() %>% 
        self$e2() %>% 
        nnf_relu()
    }
    
    # encode hidden layer to mean and variance vectors
    mu <- self$e_mu(hidden3)
    logvar <- self$e_logvar(hidden3)
    list(mu, logvar)
  })

decoder <- nn_module(
  "Decoder",
  initialize = function(vae_dim, inject_batch = FALSE, n_batch) {
    self$inject_batch = inject_batch
    if (inject_batch & n_batch == 0) {
      stop("Number of batches must be greater than 0")
    }
    if (!inject_batch) {
      n_batch = 0
    }
    
    self$d1 <- nn_linear(vae_dim[4] + n_batch, vae_dim[3])
    self$d2 <- nn_linear(vae_dim[3] + n_batch, vae_dim[2])
    self$d3 <- nn_linear(vae_dim[2] + n_batch, vae_dim[1])
  },
  forward = function(z, batch = NULL) {
    if (self$inject_batch) {
      hidden1 <- torch_cat(list(z, batch), dim = -1) %>% 
        self$d1() %>% 
        nnf_relu()
      hidden2 <- torch_cat(list(hidden1, batch), dim = -1) %>% 
        self$d2() %>% 
        nnf_relu()
      x_hat <- torch_cat(list(hidden2, batch), dim = -1) %>% 
        self$d3()
    } else{
      x_hat <- z %>% 
        self$d1() %>% 
        nnf_relu() %>% 
        self$d2() %>% 
        nnf_relu() %>% 
        self$d3()
    }
    
    x_hat
  })

vae_module <- nn_module(
  "VAE", 
  initialize = function(feature_dim, latent_feature_dim, covariate_dim, latent_overall_dim, n_batch) {
    if (latent_overall_dim >= latent_feature_dim + covariate_dim) {
      warning("latent_overall_dim is recommended to be smaller than latent_feature_dim + covariate_dim")
    }
    if (latent_overall_dim < min(latent_feature_dim, covariate_dim)) {
      warning("latent_overall_dim is recommended to be bigger than either latent_feature_dim or covariate_dim")
    }
    
    self$latent_feature_dim <- latent_feature_dim
    self$encoder_dims <- calculate_vae_dim(feature_dim, latent_feature_dim)
    self$decoder_dims <- calculate_vae_dim(feature_dim, latent_overall_dim)
    
    self$encoder <- encoder(self$encoder_dims, inject_batch = TRUE, n_batch)
    self$covariate_injector <- nn_linear(latent_feature_dim + covariate_dim, latent_overall_dim)
    self$decoder <- decoder(self$decoder_dims, inject_batch = TRUE, n_batch)
  },
  restyle = function(item_list) {
    features <- item_list[[1]]
    residuals <- item_list[[2]]
    covariates <- item_list[[3]]
    batch <- item_list[[4]]
    new_batch <- item_list[[5]]
    
    latent_feature_dist <- self$encoder(x = residuals, batch = batch) # encode features to latent feature distributions using original batch
    feat_z <- latent_feature_dist[[1]] # dont need to map to distribution anymore when restyling
    
    latent_overall <- self$covariate_injector(torch_cat(list(feat_z, covariates), dim = -1))
    
    feat_reconstructed <- self$decoder(z = latent_overall, batch = new_batch) # decode using "restyle" batch
    
    list(feat_recon = feat_reconstructed, feat_mu = feat_z, feat_logvar = NULL, feat_latent = latent_overall)
  },
  forward = function(item_list) {
    features <- item_list[[1]]
    residuals <- item_list[[2]]
    covariates <- item_list[[3]]
    batch <- item_list[[4]]
    
    latent_feature_dist <- self$encoder(x = residuals, batch = batch) # encode features to latent feature distributions
    feat_mu <- latent_feature_dist[[1]]
    feat_logvar <- latent_feature_dist[[2]]
    feat_z <- feat_mu + torch_exp(feat_logvar$mul(0.5)) * torch_randn(self$latent_feature_dim) # reparameterization trick
    
    latent_overall <- self$covariate_injector(torch_cat(list(feat_z, covariates), dim = -1))
    
    feat_reconstructed <- self$decoder(z = latent_overall, batch = batch)
    
    return(list(feat_recon = feat_reconstructed, feat_mu = feat_mu, feat_logvar = feat_logvar, feat_latent = latent_overall))
  }
)