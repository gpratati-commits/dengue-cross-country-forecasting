# ============================================================
# Negative-binomial dengue forecasting benchmark
#
# Models:
# 1. Epidemiology + seasonality
# 2. Epidemiology + seasonality + climate
#
# Forecast horizons:
# 1, 2 and 4 weeks
#
# Final available source year for each setting is held out
# as the test period.
# ============================================================

library(MASS)


# ============================================================
# PATHS
# ============================================================

INPUT_FILE <- "data/processed/dengue_model_features.csv"

OUTPUT_DIR <- "outputs/tables"

dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)
PREDICTION_DIR <- "outputs/predictions"

dir.create(
  PREDICTION_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

# ============================================================
# READ DATA
# ============================================================

df <- read.csv(
  INPUT_FILE,
  stringsAsFactors = FALSE
)

cat(
  "Rows loaded:",
  nrow(df),
  "\n"
)


# ============================================================
# METRICS
# ============================================================

mae <- function(actual, predicted) {

  mean(
    abs(actual - predicted)
  )
}


rmse <- function(actual, predicted) {

  sqrt(
    mean(
      (actual - predicted)^2
    )
  )
}


# ============================================================
# MASE SCALE
# ============================================================

mase_scale <- function(training) {

  training <- training[
    order(training$analysis_week),
  ]

  previous_cases <- c(
    NA,
    head(
      training$cases,
      -1
    )
  )

  previous_week <- c(
    NA,
    head(
      training$analysis_week,
      -1
    )
  )

  consecutive <- (
    training$analysis_week
    - previous_week
    == 1
  )

  valid <- (
    !is.na(training$cases)
    &
    !is.na(previous_cases)
    &
    consecutive
  )

  if (!any(valid)) {

    return(NA_real_)
  }

  mean(
    abs(
      training$cases[valid]
      - previous_cases[valid]
    )
  )
}


# ============================================================
# MODEL SETTINGS
# ============================================================

settings <- unique(
  df$setting
)

horizons <- c(
  1,
  2,
  4
)

results <- list()
prediction_rows <- list()
prediction_index <- 1

result_index <- 1


# ============================================================
# FIT MODELS
# ============================================================

for (setting_name in settings) {

  setting_data <- df[
    df$setting == setting_name,
  ]

  setting_data <- setting_data[
    order(setting_data$analysis_week),
  ]


  test_year <- max(
    setting_data$source_year,
    na.rm = TRUE
  )


  training <- setting_data[
    setting_data$source_year < test_year,
  ]


  test <- setting_data[
    setting_data$source_year == test_year,
  ]


  scale <- mase_scale(
    training
  )


  for (horizon in horizons) {

    response <- paste0(
      "target_cases_h",
      horizon
    )


    required_columns <- c(
      response,
      "log_cases",
      "season_sin",
      "season_cos",
      "log_rain_era5_lag_4",
      "temp_era5_lag_4",
      "humidity_era5_lag_4"
    )


    train_complete <- training[
      complete.cases(
        training[, required_columns]
      ),
    ]


    test_complete <- test[
      complete.cases(
        test[, required_columns]
      ),
    ]


    if (
      nrow(train_complete) < 50
      ||
      nrow(test_complete) == 0
    ) {

      next
    }


    # --------------------------------------------------------
    # Epidemiology-only model
    # --------------------------------------------------------

    formula_epi <- as.formula(
      paste(
        response,
        "~ log_cases + season_sin + season_cos"
      )
    )


    # --------------------------------------------------------
    # Climate-informed model
    # --------------------------------------------------------

    formula_climate <- as.formula(
      paste(
        response,
        "~ log_cases + season_sin + season_cos",
        "+ log_rain_era5_lag_4",
        "+ temp_era5_lag_4",
        "+ humidity_era5_lag_4"
      )
    )


    models <- list(
      epidemiology_only =
        formula_epi,

      climate_informed =
        formula_climate
    )


    for (model_name in names(models)) {

      fit <- tryCatch(

        glm.nb(
          models[[model_name]],
          data = train_complete
        ),

        error = function(e) {

          message(
            "Model failed: ",
            setting_name,
            " horizon=",
            horizon,
            " model=",
            model_name,
            " | ",
            e$message
          )

          return(NULL)
        }
      )


      if (is.null(fit)) {

        next
      }


      # ----------------------------------------------------------
# Generate predictions on held-out test data
# ----------------------------------------------------------

predictions <- predict(
  fit,
  newdata = test_complete,
  type = "response"
)


# ----------------------------------------------------------
# Actual observed dengue cases for the forecast horizon
# ----------------------------------------------------------

actual <- test_complete[[response]]


# ----------------------------------------------------------
# Safety checks
# ----------------------------------------------------------

if (length(actual) != length(predictions)) {

  stop(
    paste(
      "Prediction length mismatch:",
      setting_name,
      "horizon",
      horizon,
      model_name
    )
  )
}


valid_prediction <- (
  !is.na(actual)
  &
  !is.na(predictions)
  &
  is.finite(actual)
  &
  is.finite(predictions)
)


actual_eval <- actual[
  valid_prediction
]

prediction_eval <- predictions[
  valid_prediction
]


if (length(actual_eval) == 0) {

  next
}
prediction_detail <- test_complete[
  valid_prediction,
  c(
    "source_year",
    "source_week",
    "analysis_week",
    "week_label"
  )
]

prediction_detail$setting <- setting_name
prediction_detail$test_year <- test_year
prediction_detail$horizon_weeks <- horizon
prediction_detail$model <- model_name

prediction_detail$target_analysis_week <- (
  prediction_detail$analysis_week
  + horizon
)

prediction_detail$actual <- actual_eval
prediction_detail$predicted <- prediction_eval

prediction_rows[[prediction_index]] <- prediction_detail[
  ,
  c(
    "setting",
    "test_year",
    "source_year",
    "source_week",
    "week_label",
    "analysis_week",
    "target_analysis_week",
    "horizon_weeks",
    "model",
    "actual",
    "predicted"
  )
]

prediction_index <- prediction_index + 1


# ----------------------------------------------------------
# Forecast metrics
# ----------------------------------------------------------

model_mae <- mae(
  actual_eval,
  prediction_eval
)


model_rmse <- rmse(
  actual_eval,
  prediction_eval
)


if (
  !is.na(scale)
  &&
  scale > 0
) {

  model_mase <- (
    model_mae
    /
    scale
  )

} else {

  model_mase <- NA_real_
}

      

      results[[result_index]] <- data.frame(

        setting =
          setting_name,

        test_year =
          test_year,

        horizon_weeks =
          horizon,

        model =
          model_name,

        n_predictions =
  length(actual_eval),

        mae =
          model_mae,

        rmse =
          model_rmse,

        mase =
          model_mase,

        stringsAsFactors = FALSE
      )


      result_index <- (
        result_index + 1
      )
    }
  }
}


# ============================================================
# COMBINE RESULTS
# ============================================================

results_df <- do.call(
  rbind,
  results
)


if (
  is.null(results_df)
  ||
  nrow(results_df) == 0
) {

  stop(
    "No negative-binomial models were successfully fitted."
  )
}

# ============================================================
# SAVE PREDICTION-LEVEL RESULTS
# ============================================================

if (length(prediction_rows) == 0) {
  stop(
    "No prediction-level rows were generated."
  )
}

predictions_df <- do.call(
  rbind,
  prediction_rows
)

PREDICTION_FILE <- file.path(
  PREDICTION_DIR,
  "negative_binomial_predictions.csv"
)

write.csv(
  predictions_df,
  PREDICTION_FILE,
  row.names = FALSE
)

cat(
  "\nSaved prediction-level data:",
  PREDICTION_FILE,
  "\n"
)
# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

RESULTS_FILE <- file.path(
  OUTPUT_DIR,
  "negative_binomial_results.csv"
)

write.csv(
  results_df,
  RESULTS_FILE,
  row.names = FALSE
)


cat(
  "\nNEGATIVE-BINOMIAL RESULTS\n"
)

print(
  results_df
)


# ============================================================
# SUMMARY ACROSS SETTINGS
# ============================================================

summary_df <- aggregate(

  cbind(
    mae,
    rmse,
    mase
  )
  ~ horizon_weeks + model,

  data = results_df,

  FUN = median,

  na.rm = TRUE
)


names(summary_df)[
  names(summary_df) == "mae"
] <- "median_mae"

names(summary_df)[
  names(summary_df) == "rmse"
] <- "median_rmse"

names(summary_df)[
  names(summary_df) == "mase"
] <- "median_mase"


SUMMARY_FILE <- file.path(
  OUTPUT_DIR,
  "negative_binomial_summary.csv"
)


write.csv(
  summary_df,
  SUMMARY_FILE,
  row.names = FALSE
)


cat(
  "\nMEDIAN PERFORMANCE ACROSS SETTINGS\n"
)

print(
  summary_df
)


cat(
  "\nNEGATIVE-BINOMIAL FORECASTING COMPLETE\n"
)

cat(
  "Saved:",
  RESULTS_FILE,
  "\n"
)

cat(
  "Saved:",
  SUMMARY_FILE,
  "\n"
)