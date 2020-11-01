library(dplyr)
library(caret)
library(xgboost)
library(pROC)
library(ROCR)

# XGBoost tager 2 slags input:
# Dense Matrix: R’s dense matrix, i.e. matrix
# Sparse Matrix: R’s sparse matrix, i.e. Matrix::dgCMatrix ;

source("./functions.R")

# henter data ----
load("./data/adult.rda")

data <- rename(adult,
               "response" = "<=50K",
               "state_gov" = "State-gov",
               "bachelors" = "Bachelors",
               "never_married" = "Never-married",
               "Adm_clerical" = "Adm-clerical",
               "family" = "Not-in-family",
               "country" = "United-States")

response <- ifelse(data$response == "<=50K", 0, 1)

# udtager kun kategoriske features
data <- dplyr::select(data, "state_gov", "bachelors", "never_married",
                      "Adm_clerical", "family", "White", "Male",
                      "country")

# one hot encoder data
dummy <- dummyVars(" ~ .", data = data)
data_one_hot <- data.frame(predict(dummy, newdata = data))

# sætter parametre
cross_val <- 5
n <- 50

sparse_vs_dense_random_na_result <- list()
sparse_vs_dense_random_na_result$tune_lenght <- n
sparse_vs_dense_random_na_result$cv <- cross_val

search_grid <- define_search_grid(n)
sparse_vs_dense_random_na_result$tune_params <- names(search_grid[[1]])

# Konstruerer split index og splitter response ----
index <- createDataPartition(response, p = 0.8, list = FALSE)

response_train <-  response[index]
response_test <- response[-index]

# Sætter random parametre ----
na_impute_pct <- c(0.1, 0.2, 0.3, 0.4)

sparse_vs_dense_random_na_result$missing_list <- na_impute_pct

result <- list()

for (i in 1:length(na_impute_pct)) {

  data_random_na <- as.data.frame(
    lapply(data_one_hot,
           function(x) x[sample(c(TRUE, NA),
                                prob = c(1 - na_impute_pct[i], na_impute_pct[i]),
                                size = length(x),
                                replace = TRUE)]))

  # splitter data
  data_random_na_train <- data_random_na[index, ]
  data_random_na_test <- data_random_na[-index, ]

  # sætter base score og definerer search grid ----
  base_score <- round(sum(response_train) / length(response_train), 2)

  # sparse data with random NA model ----
  sparse_random_na_train <- as(data_random_na_train, "Matrix")
  sparse_random_na_test <- as(data_random_na_test, "Matrix")

  models_sparse_random_na_tune <-
    lapply(search_grid,
           function(x) fit_xgb_cv(x,
                                  sparse_random_na_train,
                                  response_train,
                                  cross_val,
                                  base_score)
    )

  cv_results_sparse <- models_sparse_random_na_tune %>%
    bind_rows() %>%
    filter(test_auc == max(test_auc))

  best_iter <- cv_results_sparse$nrounds
  cv_best_params <- as.list(dplyr::select(cv_results_sparse, -test_auc, -nrounds))

  model_sparse_random_na <- xgboost(data = sparse_random_na_train,
                                    label = response_train,
                                    params = cv_best_params,
                                    nrounds = best_iter,
                                    eval_metric = "auc",
                                    verbose = 1,
                                    print_every_n = 20)

  # dense data with random NA model ----
  dense_random_na_train <- as.matrix(data_random_na_train)
  dense_random_na_test <- as.matrix(data_random_na_test)

  models_dense_random_na_tune <-
    lapply(search_grid,
           function(x) fit_xgb_cv(x,
                                  dense_random_na_train,
                                  response_train,
                                  cross_val,
                                  base_score)
    )

  cv_results_dense <- models_dense_random_na_tune %>%
    bind_rows() %>%
    filter(test_auc == max(test_auc))

  best_iter <- cv_results_dense$nrounds
  cv_best_params <- as.list(dplyr::select(cv_results_dense, -test_auc, -nrounds))

  model_dense_random_na <- xgboost(data = dense_random_na_train,
                                   label = response_train,
                                   params = cv_best_params,
                                   nrounds = best_iter,
                                   eval_metric = "auc",
                                   verbose = 1,
                                   print_every_n = 20)

  # evaluerer modellen på testsæt ----
  pred_sparse_random_na <- predict(model_sparse_random_na, sparse_random_na_test)
  pred_dense_random_na <- predict(model_dense_random_na, dense_random_na_test)

  # udregner roc
  auc_sparse_random_na <- round(auc(roc(response_test, pred_sparse_random_na)), 4)
  auc_dense_random_na <- round(auc(roc(response_test, pred_dense_random_na)), 4)

  pred_list <- list(pred_sparse_random_na, pred_dense_random_na)
  m <- length(pred_list)
  actuals_list <- rep(list(response_test), m)

  pred <- prediction(pred_list, actuals_list)
  rocs <- performance(pred, "tpr", "fpr")

  percent <- paste0(na_impute_pct[i] * 100, "_percent")

  jpeg(paste0("./results/sparse_vs_dense_random_na_roc_plot_", percent,".jpg"))

  plot(rocs, col = as.list(1:m), main = "ROC Curves")

  legend(x = "bottomright",
         legend = c(paste("Sparse model random na auc:", auc_sparse_random_na),
                    paste("Dense model random na auc:", auc_dense_random_na)),
         title = paste0(na_impute_pct[i] * 100, "% missing values"),
         fill = 1:m, cex = 1.2)

  dev.off()

  # udregner accuracy
  pred_sparse_class <- ifelse(pred_sparse_random_na >= 0.5, 1, 0)
  pred_dense_class <- ifelse(pred_dense_random_na >= 0.5, 1, 0)

  acc_sparse_model <- sum(pred_sparse_class == response_test) / length(response_test)
  acc_dense_model <- sum(pred_dense_class == response_test) / length(response_test)

  cat("Accuracy for sparse model er:", acc_sparse_model, "\n")
  cat("Accuracy for dense model er:", acc_dense_model, "\n")

  sparse_vs_dense_random_na_result[[paste0("acc_sparse_model_", percent)]] <-
    acc_sparse_model
  sparse_vs_dense_random_na_result[[paste0("acc_dense_model_", percent)]] <-
    acc_dense_model
}

save(sparse_vs_dense_random_na_result,
     file = "./results/sparse_vs_dense_random_na_result.rda")
