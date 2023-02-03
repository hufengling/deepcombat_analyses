train_nn_annealer <- function(train_epochs, anneal_rate, beta_weights, ...) {
  trained <- train_nn(n_epochs = train_epochs[1],
                      beta_mse = 1, beta_prior = 0,
                      beta_ind = 0, beta_cov = 0,
                      anneal_rate = 0, ...)
  pretrained_model <- trained$model
  pretrained_optim <- trained$optim
  
  beta_prior <- beta_weights[1]
  beta_ind <- beta_weights[2]
  beta_cov <- beta_weights[3]
  
  trained <- train_nn(n_epochs = train_epochs[2],
                      beta_mse = 1, beta_prior = beta_prior,
                      beta_ind = beta_ind, beta_cov = beta_cov,
                      anneal_rate = anneal_rate, ...)
  model <- trained$model
  optim <- trained$optim
  
  trained <- train_nn(n_epochs = train_epochs[3],
                      beta_mse = 1, beta_prior = beta_prior,
                      beta_ind = beta_ind, beta_cov = beta_cov,
                      anneal_rate = 0, ...)
  model <- trained$model
  optim <- trained$optim
  final_recon_loss <- trained$final_recon_loss
  
  return(list(model = model, optim = optim, final_recon_loss = final_recon_loss))
}

train_nn <- function(torch_dl, torch_model, torch_optim, 
                     #adv_model, adv_optim,
                     n_epochs_total, 
                     beta_mse = 1, beta_prior = 1, 
                     beta_ind = 1, beta_cov = 1,
                     anneal_rate = 0, 
                     similarity_type = "ones", pairwise_type = "KL", recon_type = "mse",
                     batch_weights = c(1, 1), verbose = FALSE) {
  
  torch_model$train()
  anneal_item_rate <- torch_dl$.length() * anneal_rate
  anneal_recorder <- 0
  anneal_pause_rate <- floor(anneal_item_rate / 4)
  pause_recorder <- anneal_pause_rate
  if (anneal_rate <= 0) {
    beta_scale_factor <- 1
  }
  
  for (epoch in 1:n_epochs_total) {
    loss_recorder <- rep(0, 4)
    
    coro::loop(for (item in torch_dl) {
      torch_optim$zero_grad()
      output <- torch_model(item)
      vae_transform_list <- torch_model$get_vae_transforms(output)
      
      recon_weights <- item[[3]] * batch_weights[2] + (!item[[3]]) * batch_weights[1]
      covariate_similarity <- torch_model$get_covariate_similarity(item, similarity_type = similarity_type)
      pairwise_mat <- torch_model$get_pairwise_loss(vae_transform_list, pairwise_type = pairwise_type)
      
      ## Pairwise loss
      is_different_batch <- torch_cdist(item[[3]], item[[3]]) != 0
      
      loss_ind <- torch_sum(pairwise_mat * covariate_similarity * is_different_batch) / vae_transform_list$n_minibatch# / (torch_sum(covariate_similarity * is_different_batch))
      loss_cov <- torch_sum(pairwise_mat * covariate_similarity) / vae_transform_list$n_minibatch# / ((torch_sum(covariate_similarity) - dim(item[[1]])[1]))
      
      ## KL divergence between subjects and prior
      loss_prior <- torch_model$get_prior_loss(vae_transform_list, recon_weights) / vae_transform_list$n_minibatch
      
      ## MSE and L1 loss
      loss_recon <- torch_model$get_recon_loss(vae_transform_list, item[[1]], 
                                               recon_weights = recon_weights, 
                                               recon_type = recon_type) / vae_transform_list$n_minibatch#nn_mse_loss(reduction = "sum")(vae_transform_list$recon, item[[1]]) / vae_transform_list$n_batch
      #loss_recon_l1 <- nn_l1_loss(reduction = "mean")(recon, item[[1]])
      
      ## Adversary on residuals
      #resid <- recon - item[[1]]
      
      # adv_model$train()
      # adv_output <- adv_model(item, torch_tensor(recon, requires_grad = FALSE))
      # loss_adv <- nn_bce_loss()(adv_output, item[[3]])
      # loss_adv$backward()
      # adv_optim$step()
      # adv_optim$zero_grad()
      # 
      # adv_model$eval()
      # adv_output <- adv_model(item, recon)
      # loss_resid <- nn_bce_loss()(adv_output, item[[3]])
      # #print(loss_resid)
      
      ## Frobenius norm between covariance matrices of residuals across batch
      # resid_0 <- resid * !item[[3]]
      # variance_mat_0 <- torch_eye(batch_size) - torch_ones(c(batch_size, batch_size)) / torch_sum(!item[[3]])
      # cov_mat_0 <- torch_matmul(torch_transpose(resid_0, 1, 2), variance_mat_0) %>%
      #   torch_matmul(resid_0) / (torch_sum(!item[[3]]) - 1)
      # 
      # resid_1 <- resid * item[[3]]
      # variance_mat_1 <- torch_eye(batch_size) - torch_ones(c(batch_size, batch_size)) / torch_sum(item[[3]])
      # cov_mat_1 <- torch_matmul(torch_transpose(resid_1, 1, 2), variance_mat_1) %>%
      #   torch_matmul(resid_1) / (torch_sum(item[[3]]) - 1)
      # 
      # loss_resid <- nn_mse_loss()(cov_mat_1, cov_mat_1 * torch_eye(n_features)) + 
      #   nn_mse_loss()(cov_mat_0, cov_mat_0 * torch_eye(n_features))
      
      ## Full loss
      if (anneal_rate > 0) {
        beta_scale_factor <- (anneal_recorder %% (anneal_item_rate + 1)) / anneal_item_rate
        if (beta_scale_factor == 1 & pause_recorder != 0) {
          pause_recorder <- pause_recorder - 1
        } else {
          pause_recorder <- anneal_pause_rate
          anneal_recorder <- anneal_recorder + 1
        }
      }
      
      full_loss <- beta_mse * loss_recon + 
        beta_scale_factor * (beta_prior * loss_prior + beta_ind * loss_ind + beta_cov * loss_cov)
      
      full_loss$backward()
      torch_optim$step()
      
      loss_recorder <- loss_recorder + c(as.numeric(beta_mse * loss_recon),
                                         as.numeric(beta_scale_factor * beta_prior * loss_prior), 
                                         as.numeric(beta_scale_factor * beta_ind * loss_ind),
                                         as.numeric(beta_scale_factor * beta_cov * loss_cov))
    })
    gc()
    if (verbose) {
      print(paste0("Loss at epoch ", epoch, ": ", 
                   round(sum(loss_recorder), 3), 
                   " (MSE, KLD, Ind, Cov): (", 
                   paste(round(loss_recorder, 3), collapse = ", "), ")"))
    }
  }
  return(list(model = torch_model, 
              optim = torch_optim,
              final_recon_loss = round(loss_recorder[1], 3)))
}