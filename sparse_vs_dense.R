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

save(sparse_vs_dense_result, file = "./results/sparse_vs_dense_result.rda")
