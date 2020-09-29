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
index <- createDataPartition(response, p = 0.8, list = FALSE)

response_train <-  response[index]
response_test <- response[-index]

base_score <- round(sum(response_train) / length(response_train), 2)
cross_val <- 5

sparse_vs_dense_result <- list()
sparse_vs_dense_result$cv <- cross_val

# one hot encoder data
dummy <- dummyVars(" ~ .", data = data)
data_one_hot <- data.frame(predict(dummy, newdata = data))

data_one_hot_train <- data_one_hot[index, ]
data_one_hot_test <- data_one_hot[-index, ]

# sætter model parametre ----
n <- 50
search_grid <- define_search_grid(n)

sparse_vs_dense_result$tune_lenght <- n
sparse_vs_dense_result$tune_params <- names(search_grid[[1]])

# sparse data model ----
sparse_train <- as(data_one_hot_train, "Matrix")
sparse_test <- as(data_one_hot_test, "Matrix")

tictoc::tic()
models_sparse_tune <- lapply(search_grid,
                             function(x) fit_xgb_cv(x,
                                                    sparse_train,
                                                    response_train,
                                                    cross_val,
                                                    base_score)
                             )
tid <- tictoc::toc()
sparse_vs_dense_result$time_sparse_model <- tid$toc - tid$tic

cv_results_sparse <- models_sparse_tune %>%
                     bind_rows() %>%
                    filter(test_auc == max(test_auc))
best_iter <- cv_results_sparse$nrounds
cv_best_params <- as.list(dplyr::select(cv_results_sparse, -test_auc, -nrounds))

model_sparse <- xgboost(data = sparse_train, label = response_train,
                        params = cv_best_params, nrounds = best_iter,
                        eval_metric = "auc",
                        verbose = 1,
                        print_every_n = 20)

# dense data model ----
dense_train <- as.matrix(data_one_hot_train)
dense_test <- as.matrix(data_one_hot_test)

tictoc::tic()
models_dense_tune <- lapply(search_grid,
                             function(x) fit_xgb_cv(x,
                                                    dense_train,
                                                    response_train,
                                                    cross_val,
                                                    base_score)
                             )
tid <- tictoc::toc()
sparse_vs_dense_result$time_dense_model <- tid$toc - tid$tic

cv_results_dense <- models_dense_tune %>%
                    bind_rows() %>%
                    filter(test_auc == max(test_auc))
best_iter <- cv_results_dense$nrounds
cv_best_params <- as.list(dplyr::select(cv_results_dense, -test_auc, -nrounds))

model_dense <- xgboost(data = dense_train, label = response_train,
                       params = cv_best_params, nrounds = best_iter,
                       eval_metric = "auc",
                       verbose = 1,
                       print_every_n = 20)

# evaluerer modellen på testsæt ----
pred_sparse <- predict(model_sparse, sparse_test)
pred_dense <- predict(model_dense, dense_test)

##### side analyse #####
pred_sparse_tmp <- predict(model_sparse, dense_test)
identical(pred_sparse_tmp, pred_sparse)
summary(pred_sparse_tmp)
summary(pred_sparse)

pred_dense_tmp <- predict(model_dense, sparse_test)
identical(pred_dense_tmp, pred_dense)
summary(pred_dense_tmp)
summary(pred_dense)

# NB! Hvis modellen er trænet på sparse data, så er
# det vigtigt at den også scorer på sparse data.
# Ellers giver scoringer ingen mening. Problemet
# gælder dog ikke den anden vej rundt, hvor en model
# træner på dense data, godt kan score på sparse
# data med samme resultater!
########################

# udregner roc
auc_sparse <- round(auc(roc(response_test, pred_sparse)), 4)
auc_dense <- round(auc(roc(response_test, pred_dense)), 4)

pred_list <- list(pred_sparse, pred_dense)
m <- length(pred_list)
actuals_list <- rep(list(response_test), m)

pred <- prediction(pred_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")

jpeg("./results/sparse_vs_dense_roc_plot.jpg")

plot(rocs, col = as.list(1:m), main = "ROC Curves")

legend(x = "bottomright",
       legend = c(paste("Sparse model auc:", auc_sparse),
                  paste("Dense model auc:", auc_dense)),
       fill = 1:m, cex = 1)

dev.off()

# udregner accuracy
pred_sparse_class <- ifelse(pred_sparse >= 0.5, 1, 0)
pred_dense_class <- ifelse(pred_dense >= 0.5, 1, 0)

acc_sparse_model <- sum(pred_sparse_class == response_test) / length(response_test)
acc_dense_model <- sum(pred_dense_class == response_test) / length(response_test)

cat("Accuracy for sparse model er:", acc_sparse_model, "\n")
cat("Accuracy for dense model er:", acc_dense_model, "\n")

sparse_vs_dense_result$acc_sparse_model <- acc_sparse_model
sparse_vs_dense_result$acc_dense_model <- acc_dense_model

save(sparse_vs_dense_result, file = "./results/sparse_vs_dense_result.rda")


