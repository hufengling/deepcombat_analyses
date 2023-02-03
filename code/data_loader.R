adni_ct_dataset <- dataset(
  name = "adni_ct_dataset",
  
  initialize = function(input_list, 
                        data_type = c("train", "test", "all"), 
                        insert_new_batch = FALSE, 
                        new_batch = NULL) {
    self$insert_new_batch = insert_new_batch
    if (!data_type %in% c("train", "test", "all")) {
      stop("data_type must be one of 'train', 'test', or 'all'")
    }
    if (insert_new_batch) {
      if (is.null(new_batch)) {
        stop("If insert_new_batch == TRUE, must provide new_batch")
      }
      if (!is.matrix(new_batch)) {
        if (is.data.frame(new_batch)) {
          new_batch = as.matrix(new_batch)
        } else {
          stop("new_batch must be a matrix or a dataframe")
        }
      }
      if (data_type != "all") {
        warning("data_type was not set as 'all'. Changing data_type to 'all'")
        data_type = "all"
      }
    }
    
    if (data_type == "train") {
      self$data <- torch_tensor(input_list$train$data)
      #self$data_residuals <- torch_tensor(input_list$train$data_residuals)
      self$covariates <- torch_tensor(input_list$train$cov)
      self$batch <- torch_tensor(input_list$train$batch)
    } else if (data_type == "test") {
      self$data <- torch_tensor(input_list$test$data)
      #self$data_residuals <- torch_tensor(input_list$test$data_residuals)
      self$covariates <- torch_tensor(input_list$test$cov)
      self$batch <- torch_tensor(input_list$test$batch)
    } else if (data_type == "all") {
      if (!all(c("data", "cov", "batch") %in% names(input_list))) {
        stop("List must have elements 'data', 'cov', 'batch' - not 'train' and 'test'")
      }
      self$data <- torch_tensor(input_list$data)
      #self$data_residuals <- torch_tensor(input_list$data_residuals)
      self$covariates <- torch_tensor(input_list$cov)
      self$batch <- torch_tensor(input_list$batch)
      
      if (insert_new_batch) {
        self$new_batch <- torch_tensor(new_batch)
      } else {
        self$new_batch <- torch_tensor(batch)
      }
    }
  },
  
  .getitem = function(index) {
    data <- self$data[index, ]
    #data_residuals <- self$data_residuals[index, ]
    covariates <- self$covariates[index, ]
    batch <- self$batch[index]
    new_batch <- self$new_batch[index]
    return(list(data, 
                #data_residuals, 
                covariates, batch, new_batch))
  },
  
  .length = function() {
    self$data$size()[[1]]
  }
)

# easy_dataset <- dataset(
#   name = "easy_dataset",
#   
#   initialize = function(input_list, 
#                         insert_new_batch = FALSE, 
#                         new_batch = NULL) {
#     self$insert_new_batch = insert_new_batch
#     if (insert_new_batch) {
#       if (is.null(new_batch)) {
#         stop("If insert_new_batch == TRUE, must provide new_batch")
#       }
#       if (!is.matrix(new_batch)) {
#         if (is.data.frame(new_batch)) {
#           new_batch = as.matrix(new_batch)
#         } else {
#           stop("new_batch must be a matrix or a dataframe")
#         }
#       }
#     }
#     self$data <- torch_tensor(input_list$data)
#     self$covariates <- torch_tensor(input_list$cov)
#     self$batch <- torch_tensor(input_list$batch)
#     
#     if (insert_new_batch) {
#       self$new_batch <- torch_tensor(new_batch)
#     } else {
#       self$new_batch <- torch_tensor(batch)
#     }
#   },
#   
#   .getitem = function(index) {
#     data <- self$data[index, ]
#     #data_residuals <- self$data_residuals[index, ]
#     covariates <- self$covariates[index, ]
#     batch <- self$batch[index]
#     new_batch <- self$new_batch[index]
#     return(list(data,
#                 covariates, batch, new_batch))
#   },
#   
#   .length = function() {
#     self$data$size()[[1]]
#   }
# )
