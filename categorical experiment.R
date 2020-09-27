library(readr)
library(dplyr)
library(caret)
library(xgboost)
library(pROC)
library(ROCR)

source("./functions.R")

# henter data ----
data <- read_csv("C:/Users/alexg/Downloads/adult.data")

data <- rename(data,
               "response" = "<=50K",
               "state_gov" = "State-gov",
               "bachelors" = "Bachelors",
               "never_married" = "Never-married",
               "Adm_clerical" = "Adm-clerical",
               "family" = "Not-in-family",
               "country" = "United-States")

response <- ifelse(data$response == "<=50K", 0, 1)
data <- select(data, "state_gov", "bachelors", "never_married",
               "Adm_clerical", "family", "White", "Male", "country")

apply(data, 2, function(x) length(unique(x)))

index <- caret::createDataPartition(response, p = 0.8, list = FALSE)

response_train <-  response[index]
response_test <- response[-index]

base_score <- round(sum(response_train) / length(response_train), 2)
cross_val <- 5

# data hvor faktor er konverteret til numeriske features ----
data_num <- data
data_num <- apply(data_num, 2, function(x) as.numeric(as.factor(x)))

train_num <- data_num[index, ]
test_num <- data_num[-index, ]

train_num_sparse <- as(train_num, "dgCMatrix")
test_num_sparse <- as(test_num, "dgCMatrix")

# data hvor faktorer er one-hot-encoded ----
dummy <- dummyVars(" ~ .", data = data)
data_one_hot <- data.frame(predict(dummy, newdata = data))

train_one_hot <- data_one_hot[index, ]
test_one_hot <- data_one_hot[-index, ]

train_one_hot_sparse <- as(train_one_hot, "Matrix")
test_one_hot_sparse <- as(test_one_hot, "Matrix")

# data hvor weigth of evidence er brugt ----

# sætter model parametre ----
search_grid <- define_search_grid(n = 50)

# model på numeriske features ----
models_num_tune <- lapply(search_grid,
                          function(x) fit_xgb_cv(x,
                                                 train_num_sparse,
                                                 response_train,
                                                 cross_val,
                                                 base_score)
                          )

cv_results <- models_num_tune %>%
              bind_rows() %>%
              filter(test_auc == max(test_auc))
best_iter <- cv_best_result$nrounds
cv_best_params <- as.list(select(cv_results, -test_auc, -nrounds))

model_num <- xgboost(data = train_num_sparse, label = response_train,
                     params = cv_best_params, nrounds = best_iter,
                     eval_metric = "auc",
                     verbose = 1,
                     print_every_n = 20)

# model på one hot features ----
models_one_hot_tune <- lapply(search_grid,
                              function(x) fit_xgb_cv(x,
                                                     train_one_hot_sparse,
                                                     response_train,
                                                     cross_val,
                                                     base_score)
                              )

cv_results_one_hot <- models_one_hot_tune %>%
                      bind_rows() %>%
                      filter(test_auc == max(test_auc))
best_iter <- cv_results_one_hot$nrounds
cv_best_params <- as.list(select(cv_results_one_hot, -test_auc, -nrounds))

model_one_hot <- xgboost(data = train_one_hot_sparse, label = response_train,
                         params = cv_best_params, nrounds = best_iter,
                         eval_metric = "auc",
                         verbose = 1,
                         print_every_n = 20)


# evaluerer modellen på testsæt ----
pred_num <- predict(model_num, test_num_sparse)
pred_one_hot <- predict(model_one_hot, test_one_hot_sparse)

# udregner roc
auc_num <- round(auc(roc(response_test, pred_num)), 4)
auc_one_hot <- round(auc(roc(response_test, pred_one_hot)), 4)

pred_list <- list(pred_num, pred_one_hot)
m <- length(pred_list)
actuals_list <- rep(list(response_test), m)

pred <- prediction(pred_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")

legend(x = "bottomright",
       legend = c(paste("Num model auc:", auc_num),
                  paste("One hot auc:", auc_one_hot)),
       fill = 1:m)
