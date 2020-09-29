
# fit_distr ###################################################################
#' @title Fit distribution
#'
#' @description En wrapper omkring en brugervalgt funktion, som generer en
#'              vektor med tal. Hvis vektoren indeholder tal større end
#'              \code{upper}, så sættes alle værdier over denne grænse, til
#'              grænsen. Det modsatta gør sig gældende med \code{lower}.
#'
#' @param FUN Den valgte funktion.
#' @param lower Øvre grænse
#' @param upper Nedre grænse
#' @param ... Parametre til den valgte funktion
#'
#' @return En numerisk vektor med tal.
#'
fit_distr <- function(FUN, lower, upper, ...) {

  x <- FUN(...)

  x <- ifelse(x < lower, lower,
              ifelse(x > upper, upper, x))

  return(x)
}



# define_search_grid ----------------------------------------------------------
#' @title define_search_grid
#'
#' @description Konstruere random search grid
#'
#' @param n tune length
#'
#' @return Random search grid
#'
define_search_grid <- function(n) {

  set.seed(4252)
  search_grid <- data.frame(
    eta = fit_distr(stats::runif, 0.001, 0.2, n = n, min = 0.001, max = 0.3) %>% round(3),
    max_depth = fit_distr(stats::rpois, 1, 12, n = n, lambda = 5),
    gamma = fit_distr(stats::rpois, 0, 10, n = n, lambda = 3),
    min_child_weight = fit_distr(sample, 0, 50, x = 0:50, size = n, replace = TRUE),
    subsample = fit_distr(stats::runif, 0.5, 1, n = n, min = 0.5, max = 1),
    colsample_bytree = fit_distr(stats::runif, 0.5, 1, n = n, min = 0.5, max = 1),
    lambda = fit_distr(stats::runif, 0, 50, n = n, min = 0, max = 50) %>% round()
  )

  search_grid <- split(search_grid, seq(nrow(search_grid)))

  return(search_grid)
}


# fit_xgb_cv ------------------------------------------------------------------
#' @title Fit xgboost model med cross-validation
#'
#' @description Modellen fitter en xgboost model med cross-validation fra
#'              pakken \code{\link[xgboost]{xgb.cv}}.
#'
#' @param x Hyperparametre til modellen samt antal iterationer (nrounds)
#' @param xgb_matrix Træningsdata i xgb.DMatrix format
#' @param cv_folds Antal af folds i cross-validation
#' @param ... Andre parametre til \code{\link[xgboost]{xgb.cv}}
#'
#'
#' @return En vektor med de brugte hyperparametre samt den gennemsnitlige
#'         root mean square error på holdout sættet.
#'
fit_xgb_cv <- function(x, xgb_matrix, label, cv_folds = 3, ...) {

  model <- xgboost::xgb.cv(params = search_grid[1],
                           "objective" = "binary:logistic",
                           booster = 'gbtree',
                           data = xgb_matrix,
                           label = label,
                           nfold = cv_folds,
                           metrics = "auc",
                           nrounds = 1000,
                           early_stopping_rounds = 25,
                           maximize = TRUE,
                           seed = 874,
                           verbose = 0,
                           ...
                           )

  output <- c(unlist(model$params[[1]]),
              nrounds = model$best_iteration,
              test_auc = model$evaluation_log$test_auc_mean[model$best_iteration])

  return(output)
}
