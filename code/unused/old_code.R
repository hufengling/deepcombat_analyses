
# 
# residualize_data <- function(std_list, is_split = TRUE) {
#   if (all(c("data_raw", "cov", "batch") %in% names(std_list))) {
#     y <- std_list$data_raw # outcome tensor for training data
#     X <- cbind(std_list$cov, std_list$batch) # data matrix for training data
#     beta_hat <- Ginv(t(X) %*% X) %*% t(X) %*% y # beta_hat based on training data only
#     
#     std_list$data_residuals <- y - X %*% beta_hat
#     std_list$beta_hat <- beta_hat
#     
#     return(std_list)
#   }
#   
#   y <- std_list$train$data_raw # outcome tensor for training data
#   X <- cbind(std_list$train$cov, std_list$train$batch) # data matrix for training data
#   beta_hat <- Ginv(t(X) %*% X) %*% t(X) %*% y # beta_hat based on training data only
#   
#   std_list$train$data_residuals <- y - X %*% beta_hat
#   std_list$beta_hat <- beta_hat
#   
#   if (is_split) {
#     test_y <- std_list$test[[1]]
#     test_X <- cbind(std_list$test[[2]], std_list$test[[3]])
#     std_list$test$data_residuals <- test_y - test_X %*% beta_hat
#   }
#   
#   return(std_list)
# } 
# combat_vae <- nn_module(
#   "VAE",
#   initialize = function(feature_dim, latent_dim,
#                         n_hidden, n_batch, n_covariate,
#                         inject_decoder = FALSE, deep_inject = FALSE) {
#     self$latent_dim <- latent_dim
#     self$n_batch <- n_batch
#     self$n_covariate <- n_covariate
#     self$inject_decoder <- inject_decoder
#     self$deep_inject <- deep_inject
#     self$dims <- calculate_vae_dim(feature_dim, self$latent_dim, n_hidden)
# 
#     self$encoder <- vanilla_encoder(self$dims, self$n_batch, self$n_covariate)
#     self$decoder <- vanilla_decoder(self$dims, self$n_batch, self$n_covariate,
#                                     self$inject_decoder, self$deep_inject)
#   },
#   forward = function(item_list) {
#     covariates <- NULL
#     decoder_covariates <- NULL
#     if (self$n_covariate != 0) {
#       covariates <- item_list[[2]]
#     }
#     if (self$inject_decoder) {
#       decoder_covariates = covariates
#     }
#     if (self$n_batch != 0) {
#       batch <- item_list[[3]]
#     } else {
#       batch <- NULL
#     }
# 
#     features <- item_list[[1]]
#     new_batch <- item_list[[4]]
# 
#     # encode features to latent feature distributions
#     latent_dist <- self$encoder(features = features, batch = batch,
#                                 covariates = covariates)
#     feat_mu <- latent_dist[[1]]
#     feat_logvar <- latent_dist[[2]]
# 
#     design_mat <- torch_cat(list(batch, covariates), dim = -1)
#     design_mat_t <- torch_transpose(design_mat, 1, 2)
#     beta_mat <- torch_inverse(torch_matmul(design_mat_t, design_mat)) %>% torch_matmul(design_mat_t) %>% torch_matmul(feat_mu)
#     corrected_feat_mu <- feat_mu - torch_matmul(batch, torch_unsqueeze(beta_mat[1, ], 1))
# 
#     # sample from latent distribution with re-parameterization trick
#     feat_z <- corrected_feat_mu + torch_exp(feat_logvar$mul(0.5)) * torch_randn_like(feat_mu)
# 
#     feat_reconstructed <- self$decoder(z = feat_z, batch = batch,
#                                        decoder_covariates = decoder_covariates)
# 
#     return(list(feat_recon = feat_reconstructed,
#                 feat_mu = feat_mu,
#                 feat_logvar = feat_logvar,
#                 feat_z = feat_z))
#   },
#   encode_decode = function(torch_ds, raw_means, raw_sds) {
#     covariates <- NULL
#     decoder_covariates <- NULL
#     if (self$n_covariate != 0) {
#       covariates <- torch_ds$covariates
#     }
#     if (self$inject_decoder) {
#       decoder_covariates <- covariates
#     }
# 
#     if (self$n_batch != 0) {
#       batch <- torch_ds$batch
#       new_batch <- torch_ds$new_batch
#     } else {
#       batch <- NULL
#       new_batch <- NULL
#     }
# 
#     raw_means <- torch_tensor(raw_means)
#     raw_sds <- torch_tensor(raw_sds)
# 
#     latent_dist <- self$encoder(features = torch_ds$data, batch = batch,
#                                 covariates = covariates)
#     feat_mu <- latent_dist[[1]]
#     feat_logvar <- latent_dist[[2]]
# 
#     design_mat <- torch_cat(list(batch, covariates), dim = -1)
#     design_mat_t <- torch_transpose(design_mat, 1, 2)
#     beta_mat <- torch_inverse(torch_matmul(design_mat_t, design_mat)) %>% torch_matmul(design_mat_t) %>% torch_matmul(feat_mu)
#     corrected_feat_mu <- feat_mu - torch_matmul(batch, torch_unsqueeze(beta_mat[1, ], 1))
# 
#     feat_reconstructed <- self$decoder(z = corrected_feat_mu, batch = batch,
#                                        decoder_covariates = decoder_covariates) * raw_sds + raw_means
#     feat_restyled <- self$decoder(z = corrected_feat_mu, batch = new_batch,
#                                   decoder_covariates = decoder_covariates) * raw_sds + raw_means
#     feat_resids <- torch_ds$data * raw_sds + raw_means - feat_reconstructed
# 
#     combat_mu <- neuroCombat(dat = t(as.matrix(feat_mu)),
#                              batch = as.matrix(torch_ds$batch),
#                              mod = as.matrix(torch_ds$covariates),
#                              ref.batch = as.numeric(torch_ds$new_batch[1]),
#                              verbose = FALSE)$dat.combat %>%
#       t() %>% torch_tensor()
# 
#     feat_combat_resids <- (torch_ds$data - self$decoder(z = combat_mu, batch = batch,
#                                                             decoder_covariates = decoder_covariates)) * raw_sds
#     feat_combat_restyled <- self$decoder(z = combat_mu, batch = new_batch,
#                                          decoder_covariates = decoder_covariates) * raw_sds + raw_means
# 
#     return(list(recon = feat_reconstructed,
#                 restyle = feat_restyled,
#                 resids = feat_resids,
#                 resids_restyle = feat_restyled + feat_resids,
#                 combat = combat_mu,
#                 combat_restyle = feat_combat_restyled + feat_resids,
#                 combat_restyle2 = feat_combat_restyled + feat_combat_resids,
#                 latent_mu = feat_mu,
#                 latent_logvar = feat_logvar))
#   })
# 
# latent_adversary <- nn_module(
#   "latent_adversary",
#   initialize = function(n_features, n_batch, n_hidden = 2) {
#     self$layer_dim <- calculate_vae_dim(n_features, n_batch, n_hidden = n_hidden)
#     self$n_layers <- length(self$layer_dim)
#     self$layers <- nn_module_list()
# 
#     for (i in 1:(self$n_layers - 1)) {
#       self$layers[[i]] <- nn_linear(self$layer_dim[i], self$layer_dim[i + 1])
#     }
#   },
# 
#   forward = function(item, recon) {
#     tmp <- recon - item[[1]]
#     for (i in 1:(self$n_layers - 2)) {
#       tmp <- tmp %>%
#         self$layers[[i]]() %>%
#         nnf_relu()
#     }
#     output <- tmp %>%
#       self$layers[[self$n_layers - 1]]() %>%
#       nnf_sigmoid()
# 
#     return(output)
#   })