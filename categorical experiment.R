library(readr)
library(caret)
library(klaR)
library(dplyr)
library(xgboost)
library(pROC)
library(ROCR)

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

# levels
#apply(data, 2, function(x) length(unique(x)))

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

class(train_one_hot_sparse) == class(train_num_sparse)

# data hvor weight of evidence er brugt ----
data_woe <- data
data_woe[sapply(data_woe, is.character)] <-
  lapply(data_woe[sapply(data_woe, is.character)], as.factor)

data_woe <- as.data.frame(data_woe)

train_woe <- data_woe[index, ]
test_woe <- data_woe[-index, ]

woemodel <- woe(train_woe, factor(response_train), zeroadj=0.001)

train_woe <- predict(woemodel, train_woe, replace = TRUE)
test_woe <- predict(woemodel, test_woe, replace = TRUE)

train_woe_sparse <- as(train_woe, "Matrix")
test_woe_sparse <- as(test_woe, "Matrix")

train_woe_sparse <- as(train_woe_sparse, "dgCMatrix")
test_woe_sparse <- as(test_woe_sparse, "dgCMatrix")

class(train_woe_sparse) == class(train_num_sparse)

# sætter model parametre ----
search_grid <- define_search_grid(n = 100)

# model på numeriske features ----
cat("tuner numerisk model\n")
tictoc::tic()
models_num_tune <- lapply(search_grid,
                          function(x) fit_xgb_cv(x,
                                                 train_num_sparse,
                                                 response_train,
                                                 cross_val,
                                                 base_score)
                          )
tictoc::toc()

cv_results <- models_num_tune %>%
              bind_rows() %>%
              filter(test_auc == max(test_auc))
best_iter <- cv_results$nrounds
cv_best_params <- as.list(dplyr::select(cv_results, -test_auc, -nrounds))

model_num <- xgboost(data = train_num_sparse, label = response_train,
                     params = cv_best_params, nrounds = best_iter,
                     eval_metric = "auc",
                     verbose = 1,
                     print_every_n = 20)

# model på one hot features ----
cat("tuner one hot model\n")
tictoc::tic()
models_one_hot_tune <- lapply(search_grid,
                              function(x) fit_xgb_cv(x,
                                                     train_one_hot_sparse,
                                                     response_train,
                                                     cross_val,
                                                     base_score)
                              )
tictoc::toc()

cv_results_one_hot <- models_one_hot_tune %>%
                      bind_rows() %>%
                      filter(test_auc == max(test_auc))
best_iter <- cv_results_one_hot$nrounds
cv_best_params <- as.list(dplyr::select(cv_results_one_hot, -test_auc, -nrounds))

model_one_hot <- xgboost(data = train_one_hot_sparse, label = response_train,
                         params = cv_best_params, nrounds = best_iter,
                         eval_metric = "auc",
                         verbose = 1,
                         print_every_n = 20)



# model på woe features ----
cat("tuner woe model\n")
tictoc::tic()
models_woe_tune <- lapply(search_grid,
                          function(x) fit_xgb_cv(x,
                                                 train_woe_sparse,
                                                 response_train,
                                                 cross_val,
                                                 base_score)
)
tictoc::toc()

cv_results <- models_woe_tune %>%
              bind_rows() %>%
              filter(test_auc == max(test_auc))
best_iter <- cv_results$nrounds
cv_best_params <- as.list(dplyr::select(cv_results, -test_auc, -nrounds))

model_woe <- xgboost(data = train_woe_sparse, label = response_train,
                     params = cv_best_params, nrounds = best_iter,
                     eval_metric = "auc",
                     verbose = 1,
                     print_every_n = 20)

# evaluerer modellen på testsæt ----
pred_num <- predict(model_num, test_num_sparse)
pred_one_hot <- predict(model_one_hot, test_one_hot_sparse)
pred_woe <- predict(model_woe, test_woe_sparse)

# udregner roc
auc_num <- round(auc(roc(response_test, pred_num)), 4)
auc_one_hot <- round(auc(roc(response_test, pred_one_hot)), 4)
auc_woe <- round(auc(roc(response_test, pred_woe)), 4)

pred_list <- list(pred_num, pred_one_hot, pred_woe)
m <- length(pred_list)
actuals_list <- rep(list(response_test), m)

pred <- prediction(pred_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")

legend(x = "bottomright",
       legend = c(paste("Num model auc:", auc_num),
                  paste("One hot auc:", auc_one_hot),
                  paste("Woe auc:", auc_woe)),
       fill = 1:m, cex = 0.75)

# udregner accuracy
pred_num_class <- ifelse(pred_num >= 0.5, 1, 0)
pred_one_hot_class <- ifelse(pred_one_hot >= 0.5, 1, 0)
pred_woe_class <- ifelse(pred_woe >= 0.5, 1, 0)

cat("Accuracy for numerisk model er:",
    sum(pred_num_class == response_test) / length(response_test))

cat("Accuracy for one hot encoding model er:",
    sum(pred_one_hot_class == response_test) / length(response_test))

cat("Accuracy for weight of evidence model er:",
    sum(pred_woe_class == response_test) / length(response_test))



# konklusion ----
# Her er blevet trænet 2 xgboost modeller med binær target på
# features der alle er kategoriske. Forskellen på modellerne
# er hvordan de kategoriste features håndteret. Der er gjort
# på følgende måde:
# 1: De er blot taget med som numeriske features
# 2: De er one-hot encoded
# 3: Weight of evidens er benyttet
# Modellerne performer stort set ens når der måles på roc auc
# og accuracy. Dog er der stor forskel i træningstiden på
# modellerne. One-hot modellen tager omtrent 25 % længere tid
# at tune end modellen med de numeriske features.
