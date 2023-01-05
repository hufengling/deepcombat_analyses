train_ae <- function(torch_dl, torch_model, torch_optim, 
                     #adv_model, adv_optim,
                     n_epochs_total) {
  
  torch_model$train()
  
  for (epoch in 1:n_epochs_total) {
    loss_recorder <- 0
    
    coro::loop(for (item in torch_dl) {
      torch_optim$zero_grad()
      
      output <- torch_model(item)
      ## MSE
      full_loss <- nn_mse_loss(reduction = "mean")(output$feat_recon, item[[1]])
      full_loss$backward()
      torch_optim$step()
      
      loss_recorder <- loss_recorder + as.numeric(full_loss)
    })
    gc()
    print(paste0("Loss at epoch ", epoch, ": ", 
                 round(loss_recorder, 3)))
  }
  return(list(model = torch_model, optim = torch_optim))
}