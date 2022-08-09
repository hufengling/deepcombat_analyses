encoder <- nn_module(
  "Encoder",
  initialize = function(vae_dim, n_batch = 0, n_covariates = 0, 
                        inject_batch = FALSE, inject_covariates = FALSE) {
    self$inject_batch = inject_batch
    self$inject_covariates = inject_covariates
    if (inject_batch & n_batch == 0) {
      stop("Number of batches must be greater than 0")
    }
    if (inject_covariates & n_covariates == 0) {
      stop("Number of covariates must be greater than 0")
    }
    
    self$e1 <- nn_linear(vae_dim[1] + n_batch, vae_dim[2])
    self$e2 <- nn_linear(vae_dim[2] + n_batch, vae_dim[3])
    
    if (inject_covariates) {
      self$cov1 <- nn_linear(n_covariates, n_covariates)
    }
    
    self$e_mu <- nn_linear(vae_dim[3] + n_batch + n_covariates, vae_dim[4])
    self$e_logvar <- nn_linear(vae_dim[3] + n_batch + n_covariates, vae_dim[4])
  },
  forward = function(x, batch = NULL, covariates = NULL) {
    if (self$inject_batch) {
      hidden1 <- torch_cat(list(x, batch), dim = -1) %>% 
        self$e1() %>% 
        nnf_relu()
      hidden2 <- torch_cat(list(hidden1, batch), dim = -1) %>% 
        self$e2() %>% 
        nnf_relu()
      if (self$inject_covariates) {
        cov_hidden <- covariates %>% 
          self$cov1() %>% 
          nnf_relu()
        hidden3 <- torch_cat(list(hidden2, batch, cov_hidden), dim = -1) # calculate hidden layer
      } else {
        hidden3 <- torch_cat(list(hidden2, batch), dim = -1) # calculate hidden layer
      }
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
  initialize = function(vae_dim, n_batch, inject_batch = FALSE) {
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
    self$d4 <- nn_linear(vae_dim[1], vae_dim[1])
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
        self$d3() %>% 
        nnf_relu() %>% 
        self$d4()
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
  initialize = function(feature_dim, latent_dim, covariate_dim, n_batch) {
    self$latent_dim <- latent_dim
    self$covariate_dim <- covariate_dim
    self$dims <- calculate_vae_dim(feature_dim, latent_dim)
    
    self$encoder <- encoder(self$dims, n_batch, 
                            inject_batch = TRUE, inject_covariates = FALSE)
    self$decoder <- decoder(self$dims, n_batch, inject_batch = TRUE)
  },
  restyle = function(item_list) {
    features <- item_list[[1]]
    residuals <- item_list[[2]]
    covariates <- item_list[[3]]
    batch <- item_list[[4]]
    new_batch <- item_list[[5]]
    
    latent_feature_dist <- self$encoder(x = residuals, batch = batch, covariates = covariates) # encode features to latent feature distributions using original batch
    feat_z <- latent_feature_dist[[1]] # dont need to map to distribution anymore when restyling
    feat_z <- feat_z * torch_cat(batch, covariates, torch_ones(self$latent_dim - self$covariate_dim))
    
    feat_reconstructed <- self$decoder(z = feat_z, batch = new_batch) # decode using "restyle" batch
    
    list(feat_recon = feat_reconstructed, feat_mu = feat_z)
  },
  forward = function(item_list) {
    features <- item_list[[1]]
    residuals <- item_list[[2]]
    covariates <- item_list[[3]]
    batch <- item_list[[4]]
    
    latent_feature_dist <- self$encoder(x = residuals, batch = batch, covariates = covariates) # encode features to latent feature distributions
    feat_mu <- latent_feature_dist[[1]]
    feat_logvar <- latent_feature_dist[[2]]
    feat_z <- feat_mu + torch_exp(feat_logvar$mul(0.5)) * torch_randn(self$latent_dim) # reparameterization trick
    feat_z <- feat_z * torch_cat(list(batch, covariates, 
                                      torch_ones(c(dim(feat_z)[1], 
                                                   self$latent_dim - self$covariate_dim - 1))),
                                 dim = 2)# trying to learn coefficients
    
    feat_reconstructed <- self$decoder(z = feat_z, batch = batch)
    
    return(list(feat_recon = feat_reconstructed, feat_mu = feat_mu, feat_logvar = feat_logvar))
  }
)

# vae_adversary <- nn_module(
#   "VAE adversary",
#   initialize = function(latent_dim, n_batch) {
#     
#   }
# )
