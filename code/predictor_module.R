predictor_network <- nn_module(
  "prediction",
  initialize = function(n_feature, n_batch, n_covariate, n_hidden, dropout_rate = 0.8) {
    self$dropout_rate <- dropout_rate
    self$layer_dim <- calculate_vae_dim(n_feature, n_covariate + n_batch, n_hidden = n_hidden)
    self$n_layers <- length(self$layer_dim)
    self$layers <- nn_module_list()
    
    for (i in 1:(self$n_layers - 1)) {
      self$layers[[i]] <- nn_linear(self$layer_dim[self$n_layers + 1 - i], self$layer_dim[self$n_layers - i])
    }
  },
  
  forward = function(item_list) {
    batch <- item_list[[4]]
    covariates <- item_list[[3]]
    
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
      }
      output <- tmp %>% self$layers[[self$n_layers - 1]]()
      return(output)
    }
  },
  
  predict = function(torch_ds, predict_from_new = FALSE) {
    if (predict_from_new) {
      batch <- torch_ds$new_batch
    } else {
      batch <- torch_ds$batch
    }
    covariates <- torch_ds$covariates
    
    tmp <- torch_cat(list(batch, covariates), dim = 2) 
    
    if (self$n_layers == 2) {
      output <- tmp %>% self$layers[[1]]()
      return(output)
    } else {
      for (i in 1:(self$n_layers - 2)) {
        tmp <- tmp %>% 
          self$layers[[i]]() %>% 
          nnf_dropout(p = self$dropout_rate) %>% 
          nnf_relu()
      }
      output <- tmp %>% self$layers[[self$n_layers - 1]]()
      return(output)
    }
  })

linear_predictor_network <- nn_module(
  "linear_prediction",
  initialize = function(n_feature, n_batch, n_covariate) {
    self$p1 <- nn_linear(n_batch + n_covariate, n_feature)
  },
  
  forward = function(item_list) {
    batch <- item_list[[4]]
    covariates <- item_list[[3]]
    
    torch_cat(list(batch, covariates), dim = -1) %>% 
      self$p1()
  }
)