split_train_test <- function(data_list, train_prop = 0.9) {
  if (train_prop == 1) {
    return(list(train = data_list, test = NULL))
  }
  
  n <- nrow(data_list[[1]])
  train_index <- sample(1:n, floor(n * train_prop))
  test_index <- (1:n)[-train_index]
  
  train_list <- lapply(data_list, function(df) {
    df_train <- df[train_index, ]
  })
  
  test_list <- lapply(data_list, function(df) {
    df_test <- df[test_index, ]
  })
  
  return(list(train = train_list, test = test_list))
}
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

calculate_vae_dim <- function(input_dim, latent_dim, n_hidden) {
  if (input_dim <= latent_dim) {
    stop("Latent dimension must be smaller than input dimension")
  }
  if (n_hidden == 0) {
    return(c(input_dim, latent_dim))
  }
  hidden_dim <- rep(0, n_hidden)
  range <- input_dim - latent_dim
  for (i in 1:n_hidden) {
    hidden_dim[i] <- latent_dim + floor(range * (n_hidden - i + 1) / (n_hidden + 1))
  }
  
  return(c(input_dim, hidden_dim, latent_dim))
}

run_through_vae <- function(torch_dataset, torch_model, restyle) {
  torch_model$eval()
  dataset_length <- torch_dataset$.length()
  
  vae_recon <- vector("list", length = dataset_length)
  vae_latent_mu <- vector("list", length = dataset_length)
  vae_latent_overall <- vector("list", length = dataset_length)
  
  for (i in 1:dataset_length) {
    if (restyle) {
      tmp_output <- torch_model$restyle(torch_dataset[i])
    } else {
      tmp_output <- torch_model(torch_dataset[i])
    }
    
    vae_recon[[i]] <- tmp_output$feat_recon %>% as.numeric()
    vae_latent_mu[[i]] <- tmp_output$feat_mu %>% as.numeric()
    vae_latent_overall[[i]] <- tmp_output$feat_latent %>% as.numeric()
  }
  
  vae_recon <- vae_recon %>% 
    unlist() %>% 
    matrix(nrow = dataset_length, byrow = T) %>% 
    as.data.frame()
  vae_latent_mu <- vae_latent_mu %>% 
    unlist() %>% 
    matrix(nrow = dataset_length, byrow = T) %>% 
    as.data.frame()
  vae_latent_overall <- vae_latent_overall %>% 
    unlist() %>% 
    matrix(nrow = dataset_length, byrow = T) %>% 
    as.data.frame()
  
  return(list(vae_recon = vae_recon,
              vae_latent_mu = vae_latent_mu, 
              vae_latent_overall = vae_latent_overall))
}

train_nn <- function(torch_dl, torch_model, torch_optim, 
                     n_epochs_total, 
                     beta_mse = 1, beta_l1 = 1, beta_prior = 1, beta_ind = 1) {
  torch_model$train()
  for (epoch in 1:n_epochs_total) {
    loss_recorder <- c(0, 0, 0, 0)
    
    coro::loop(for (item in torch_dl) {
      torch_optim$zero_grad()
      output <- torch_model(item)
      recon <- output$feat_recon
      mu <- output$feat_mu
      mu_2 <- mu$pow(2)
      logvar <- output$feat_logvar
      var <- logvar$exp()
      inv_var <- (-logvar)$exp()
      inv_var_t <- torch_transpose(inv_var, 1, 2)
      
      batch_size <- dim(mu)[1]
      n_latent_dim <- dim(mu)[2]
      is_different_batch <- torch_cdist(item[[3]], item[[3]]) != 0
      is_same_batch <- is_different_batch == 0
      
      n_different <- torch_sum(is_different_batch)
      
      # Loss
      ## Pairwise KL divergence between subjects from different batch
      ones <- torch_ones_like(is_different_batch)
      trace_cov_prod <- torch_matmul(var, inv_var_t)
      xi_varj_xi <- torch_matmul(mu_2, inv_var_t)
      xi_vari_xi <- torch_diag(xi_varj_xi) * ones
      xi_varj_xj <- torch_matmul(mu * inv_var, torch_transpose(mu, 1, 2))
      pairwise_kl_mat <- 0.5 * (xi_varj_xi - 2 * xi_varj_xj + xi_vari_xi + trace_cov_prod - n_latent_dim * ones)
      
      loss_between_batch <- torch_sum(pairwise_kl_mat * is_different_batch) / n_different
      loss_within_batch <- torch_sum(pairwise_kl_mat * is_same_batch) / (batch_size^2 - n_different)
      loss_ind <- (loss_between_batch - loss_within_batch)^2
      
      ## KL divergence between subjects and prior
      loss_prior <- 0.5 * torch_sum(mu_2 + var - 1 - logvar) / batch_size
      
      ## MSE and L1 loss
      mse_loss <- nn_mse_loss(reduction = "mean")
      l1_loss <- nn_l1_loss(reduction = "mean")
      loss_recon <- mse_loss(recon, item[[1]]) 
      loss_recon_l1 <- l1_loss(recon, item[[1]])
      
      ## Full loss
      full_loss <- beta_mse * loss_recon + beta_l1 * loss_recon_l1 + beta_prior * loss_prior + beta_ind * loss_ind
      full_loss$backward()
      torch_optim$step()
      
      loss_recorder <- loss_recorder + c(as.numeric(beta_mse * loss_recon), 
                                         as.numeric(beta_l1 * loss_recon_l1), 
                                         as.numeric(beta_prior * loss_prior), 
                                         as.numeric(beta_ind * loss_ind))
    })
    
    print(paste0("VAE loss at epoch ", epoch, ": ", 
                 round(sum(loss_recorder), 3), 
                 " (MSE, L1, KLD, Ind): (", 
                 paste(round(loss_recorder, 3), collapse = ", "), ")"))
  }
  
  return(list(model = torch_model, optim = torch_optim))
}

train_predictor <- function(torch_dl, torch_model, torch_optim, n_epochs_total) {
  torch_model$train()
  for (epoch in 1:n_epochs_total) {
    loss_recorder <- 0
    
    coro::loop(for (item in torch_dl) {
      torch_optim$zero_grad()
      output <- torch_model(item)
      
      # Loss
      mse_loss <- nn_mse_loss(reduction = "mean")
      
      loss_recon <- mse_loss(output, item[[1]])
      loss_recon$backward()
      torch_optim$step()
      
      loss_recorder <- loss_recorder + as.numeric(loss_recon)
    })
    
    print(paste0("Prediction loss at epoch ", epoch, ": ", 
                 round(loss_recorder)))
  }
  
  return(list(model = torch_model, optim = torch_optim))
}

train_adversary <- function(torch_dl, torch_model, torch_optim, 
                            adv_model, adv_optim,
                            n_epochs_total, n_vae_per_cycle, n_adv_per_cycle, n_initial_adv,
                            beta_mse = 1, beta_l1 = 1, beta_kl = 1, beta_adv = 1) {
  epoch_counter <- 0
  for (adv_epoch in 1:n_initial_adv) {
    adv_model$train()
    torch_model$eval()
    
    loss_recorder_adv <- 0
    epoch_counter <- epoch_counter + 1
    
    coro::loop(for (item in torch_dl) {
      adv_optim$zero_grad()
      output <- torch_model(item)
      adv_output <- adv_model(output$feat_mu)
      target <- torch_flatten(item[[3]] + torch_ones_like(item[[3]]))$to(dtype = torch_long())
      
      ce_loss <- nn_cross_entropy_loss(reduction = "mean")
      
      adv_train_loss <- ce_loss(adv_output, target)
      adv_train_loss$backward()
      adv_optim$step()
      
      loss_recorder_adv <- loss_recorder_adv + as.numeric(adv_train_loss)
    })
    
    print(paste0("Adversarial loss at epoch ", epoch_counter, ": ", round(loss_recorder_adv, 3)))
  }
  
  
  while (epoch_counter <= n_epochs_total) {
    for (epoch in 1:n_vae_per_cycle) {
      adv_model$eval()
      torch_model$train()
      
      loss_recorder <- c(0, 0, 0, 0)
      epoch_counter <- epoch_counter + 1
      coro::loop(for (item in torch_dl) {
        torch_optim$zero_grad()
        output <- torch_model(item)
        recon <- output$feat_recon
        mu <- output$feat_mu
        logvar <- output$feat_logvar
        
        adv_output <- adv_model(mu)
        target <- torch_flatten(item[[3]] + torch_ones_like(item[[3]]))$to(dtype = torch_long())
        
        # Loss
        mse_loss <- nn_mse_loss(reduction = "mean")
        l1_loss <- nn_l1_loss(reduction = "mean")
        adv_loss <- nn_cross_entropy_loss(reduction = "mean")
        
        kl_div <- 0.5 * torch_mean((mu)$pow(2) + logvar$exp() - 1 - logvar) * dim(mu)[2]
        loss_recon <- mse_loss(recon, item[[1]]) 
        loss_recon_l1 <- l1_loss(recon, item[[1]])
        loss_adv <- adv_loss(adv_output, target)
        
        full_loss <- beta_mse * loss_recon + beta_l1 * loss_recon_l1 + beta_kl * kl_div - beta_adv * loss_adv
        full_loss$backward()
        torch_optim$step()
        
        loss_recorder <- loss_recorder + c(as.numeric(beta_mse * loss_recon), 
                                           as.numeric(beta_l1 * loss_recon_l1),
                                           as.numeric(beta_kl * kl_div), 
                                           -as.numeric(beta_adv * loss_adv))
      })
      
      print(paste0("VAE loss at epoch ", epoch_counter, ": ", 
                   round(sum(loss_recorder), 3), 
                   " (MSE, L1, KLD, Adv): (", paste(round(loss_recorder, 3), collapse = ", "), ")"))
    }
    
    for (adv_epoch in 1:n_adv_per_cycle) {
      adv_model$train()
      torch_model$eval()
      
      loss_recorder_adv <- 0
      epoch_counter <- epoch_counter + 1
      
      coro::loop(for (item in torch_dl) {
        adv_optim$zero_grad()
        output <- torch_model(item)
        adv_output <- adv_model(output$feat_mu)
        target <- torch_flatten(item[[3]] + torch_ones_like(item[[3]]))$to(dtype = torch_long())
        
        ce_loss <- nn_cross_entropy_loss(reduction = "mean")
        
        adv_train_loss <- ce_loss(adv_output, target)
        adv_train_loss$backward()
        adv_optim$step()
        
        loss_recorder_adv <- loss_recorder_adv + as.numeric(adv_train_loss)
      })
      
      print(paste0("Adversarial loss at epoch ", epoch_counter, ": ", round(loss_recorder_adv, 3)))
    }
  }
  
  return(list(model = torch_model, optim = torch_optim, 
              adv_model = adv_model, adv_optim = adv_optim))
}

get_mse_mae <- function(tensor_1, tensor_2) {
  if (typeof(tensor_1) != "list") {
    tensor_1 <- list(tensor_1)
  } 
  for (i in 1:length(tensor_1)) {
    tensor <- tensor_1[[i]]
    diff <- as.matrix(tensor - tensor_2)
    print(paste("Tensor", i))
    print(paste("MSE:", mean(diff^2)))
    print(paste("MAE:", mean(abs(diff))))
  }
  
  return(NULL)
}

# combat_resids <- function (pred_tensor, true_tensor, batch, mod, ref.batch = 0) {
#   if (typeof(pred_tensor) != "list") {
#     pred_tensor <- list(pred_tensor)
#   }
#   
#   resid_list <- lapply(pred_tensor, function (tensor) {
#     return(as.matrix(true_tensor - tensor))
#   })
#   
#   combat_list <- lapply(resid_list, function (resid_mat) {
#     combat_output <- neuroCombat(t(resid_mat), batch = batch, mod = mod, ref.batch = ref.batch)
#     return(t(combat_output$dat.combat))
#   })
#   
#   recon_list <- list()
#   for (i in 1:length(pred_tensor)) {
#     recon_list[[i]] <- as.matrix(pred_tensor[[i]]) + combat_list[[i]]
#   }
#   
#   return(list(resid_list = resid_list, combat_list = combat_list, recon_list = recon_list))
# }
