predictor_network <- nn_module(
  "prediction",
  initialize = function(n_feature, n_batch, n_covariate, n_hidden, dropout_rate = 0.8) {
    self$dropout_rate <- dropout_rate
    self$layer_dim <- calculate_vae_dim(n_feature, n_covariate + n_batch, n_hidden = n_hidden)
    self$n_layers <- length(self$layer_dim)
    self$layers <- nn_module_list()
    
    for (i in 1:(self$n_layers - 1)) {
      self$layers[[i]] <- nn_linear(self$layer_dim[self$n_layers + 1 - i],# + n_batch, 
                                    self$layer_dim[self$n_layers - i])
    }
  },
  
  forward = function(item_list) {
    covariates <- item_list[[2]]
    batch <- item_list[[3]]
    
    tmp <- torch_cat(list(batch, covariates), dim = -1) 
    
    if (self$n_layers == 2) {
      output <- tmp %>% self$layers[[1]]()
      return(output)
    } else {
      for (i in 1:(self$n_layers - 2)) {
        tmp <- tmp %>% 
          self$layers[[i]]() %>% 
          nnf_dropout(p = self$dropout_rate) %>% 
          nnf_relu()
        #tmp <- torch_cat(list(batch, tmp), dim = -1)
      }
      output <- tmp %>% self$layers[[self$n_layers - 1]]()
      return(output)
    }
  },
  
  predict = function(torch_ds, raw_means, raw_sds) {
    covariates <- torch_ds$covariates
    batch <- torch_ds$batch
    new_batch <- torch_ds$new_batch
    tmp_recon <- torch_cat(list(batch, covariates), dim = 2)
    tmp_restyle <- torch_cat(list(new_batch, covariates), dim = 2)
    
    if (self$n_layers == 2) {
      recon <- tmp_recon %>% self$layers[[1]]()
      restyle <- tmp_restyle %>% self$layers[[1]]()
    } else {
      for (i in 1:(self$n_layers - 2)) {
        tmp_recon <- tmp_recon %>% 
          self$layers[[i]]() %>% 
          nnf_relu()
        #tmp_recon <- torch_cat(list(batch, tmp_recon), dim = -1)
        tmp_restyle <- tmp_restyle %>% 
          self$layers[[i]]() %>% 
          nnf_relu()
        #tmp_restyle <- torch_cat(list(new_batch, tmp_restyle), dim = -1)
      }
      recon <- tmp_recon %>% self$layers[[self$n_layers - 1]]()
      restyle <- tmp_restyle %>% self$layers[[self$n_layers - 1]]()
    }
    
    raw_means <- torch_tensor(raw_means)
    raw_sds <- torch_tensor(raw_sds)
    
    recon <- recon * raw_sds + raw_means 
    restyle <- restyle * raw_sds + raw_means
    resids <- (torch_ds$data * raw_sds + raw_means) - recon
    
    # combat <- neuroCombat(dat = t(as.matrix(resids)),
    #                       batch = as.matrix(torch_ds$batch),
    #                       mod = as.matrix(covariates),
    #                       ref.batch = as.numeric(torch_ds$new_batch[1]))$dat.combat %>% 
    #   t() %>% torch_tensor()
    
    return(list(recon = recon, 
                restyle = restyle, 
                resids = resids,
                resids_restyle = restyle + resids))
                #combat = combat,
                #combat_restyle = restyle + combat))
  })
# 
# linear_predictor_network <- nn_module(
#   "linear_prediction",
#   initialize = function(n_feature, n_batch, n_covariate) {
#     self$p1 <- nn_linear(n_batch + n_covariate, n_feature)
#   },
#   
#   forward = function(item_list) {
#     batch <- item_list[[4]]
#     covariates <- item_list[[3]]
#     
#     torch_cat(list(batch, covariates), dim = -1) %>% 
#       self$p1()
#   }
# )