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

get_mse_mae <- function(tensor_1, tensor_2) {
  if (typeof(tensor_1) != "list") {
    tensor_1 <- list(tensor_1)
  } 
  if (typeof(tensor_2) == "externalptr")
    tensor_2 <- as.matrix(tensor_2)
  
  for (i in 1:length(tensor_1)) {
    tensor <- tensor_1[[i]]
    if (typeof(tensor) == "externalptr")
      tensor <- as.matrix(tensor)
    if (all(dim(tensor) == dim(tensor_2))) {
      diff <- as.matrix(tensor - tensor_2)
      print(names(tensor_1)[i])
      print(paste("MSE:", mean(diff^2)))
      print(paste("MAE:", mean(abs(diff))))
    }
  }
  
  return(NULL)
}

transform_resids <- function(resid_output, vae_resids) {
  vae_min <- torch_min(vae_resids, dim = 1, keepdim = TRUE)[[1]]
  #vae_range <- torch_max(vae_resids, dim = 1, keepdim = TRUE)[[1]] - vae_min
  #resids_sign <- (vae_resids > 0) * 2 - 1
  
  transformed_resid_output <- lapply(resid_output, function(output_item) {
    if (dim(output_item)[2] == dim(vae_min)[2]) {
      #(exp(output_item) - 0.5) * vae_range + vae_min
      exp(output_item) + vae_min - 1
    } else {
      output_item
    }
  })
  
  transformed_resid_output
}

range01 <- function (df) {
  apply(df, 2, function (col) {
    (col - min(col)) / (max(col) - min(col))
  })
}

make_input_list <- function(raw, covariates, 
                            data_opts = c("raw", "residuals", "scaled_residuals"), 
                            get_ps = FALSE, get_combat_covbat = FALSE, ...) {
  cov <- model.matrix(~ SEX + DIAGNOSIS + AGE, covariates)[, -1] # remove intercept
  batch <- as.matrix(model.matrix(~ manufac, covariates)[, -1]) # remove intercept
  
  if (get_ps) {
    ps_mod <- predict(glm(manufac ~ SEX + DIAGNOSIS + AGE, covariates, family = binomial))
    cov_norm <- as.matrix(exp(ps_mod) / (1 + exp(ps_mod)))
  }
  else {
    cov_norm <- range01(cov)
  }
  
  covbat_output <- covbat_fh(dat = t(raw), bat = batch, mod = cov_norm, ...)
  if (get_combat_covbat) {
    return(list(combat = covbat_output$combat.out,
                covbat = covbat_output))
  }
  
  if (data_opts == "raw") {
    data <- raw
    mean <- 0
    data_norm <- scale(raw)
  } 
  else {
    mean <- t(covbat_output$combat.out$stand.mean)
    if (data_opts == "residuals") {
      data <- raw - mean
    }
    if (data_opts == "scaled_residuals") {
      data <- t(covbat_output$bayesdata)
    }
    data_norm <- scale(data)
  }
  
  return(list(input_list = list(data = data_norm, cov = cov_norm, batch = batch),
              data = data, mean = mean, covbat_output = covbat_output))
}

plot_latent_by_batch <- function(latent_tensor, batch) {
  wide_df <- as.data.frame(cbind(as.matrix(batch), as.matrix(latent_tensor)))
  tall_df <- pivot_longer(wide_df, -one_of("V1"))
  tall_df$V1 <- as.factor(tall_df$V1)
  summary_mat <- tall_df %>% 
    group_by(V1) %>% 
    summarise(mean = mean(value), sd = sd(value))
  print(summary_mat)
  
  ggplot(tall_df) + geom_density(aes(value, fill = V1), alpha = 0.5)
}

multiply_by_row <- function(mat, vec) {
  mat * rep(vec, each = nrow(mat))
}
