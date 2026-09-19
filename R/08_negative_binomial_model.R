# Leakage-safe negative-binomial dengue forecasting
#
# Models:
#   1. epidemiology_only
#   2. climate_informed
#
# Forecast horizons:
#   1, 2 and 4 weeks
#
# IMPORTANT:
# Train/test membership is based on TARGET YEAR,
# not forecast-origin/source year.
#
# Outputs:
#   outputs/tables/negative_binomial_results.csv
#   outputs/tables/negative_binomial_summary.csv
#   outputs/predictions/negative_binomial_predictions.csv


# ============================================================
# PACKAGE
# ============================================================

suppressPackageStartupMessages(
  library(MASS)
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE <- "data/processed/dengue_model_features.csv"

TABLE_DIR <- "outputs/tables"

PREDICTION_DIR <- "outputs/predictions"

HORIZONS <- c(
  1L,
  2L,
  4L
)

MIN_TRAIN_ROWS <- 52L

MIN_TEST_ROWS <- 20L


dir.create(
  TABLE_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  PREDICTION_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

first_existing <- function(
  columns,
  candidates
) {

  hit <- candidates[
    candidates %in% columns
  ]

  if (length(hit) == 0L) {
    return(NULL)
  }

  hit[[1L]]
}


# ============================================================
# EXACT WEEK LOOKUP
#
# Returns the value exactly "offset" analysis-weeks away.
# This avoids accidentally treating gaps as consecutive weeks.
# ============================================================

week_lookup <- function(
  values,
  weeks,
  offset
) {

  idx <- match(
    weeks + offset,
    weeks
  )

  out <- rep(
    NA_real_,
    length(values)
  )

  ok <- !is.na(idx)

  if (any(ok)) {

    out[ok] <- as.numeric(
      values[idx[ok]]
    )

  }

  out
}


# ============================================================
# ROLLING MEAN OF PAST OBSERVATIONS
# ============================================================

rolling_past_mean <- function(
  values,
  weeks,
  window
) {

  mat <- sapply(
    seq_len(window),
    function(k) {

      week_lookup(
        values,
        weeks,
        -k
      )

    }
  )

  rowMeans(
    mat,
    na.rm = FALSE
  )
}


# ============================================================
# METRICS
# ============================================================

mae <- function(
  actual,
  predicted
) {

  mean(
    abs(
      actual - predicted
    )
  )
}


rmse <- function(
  actual,
  predicted
) {

  sqrt(
    mean(
      (
        actual - predicted
      )^2
    )
  )
}


mase_scale <- function(
  df
) {

  df <- df[
    order(
      df$analysis_week
    ),
    ,
    drop = FALSE
  ]

  previous_cases <- week_lookup(
    df$cases,
    df$analysis_week,
    -1L
  )

  ok <- (
    is.finite(
      df$cases
    )
    &
    is.finite(
      previous_cases
    )
  )

  if (!any(ok)) {
    return(NA_real_)
  }

  value <- mean(
    abs(
      df$cases[ok]
      -
      previous_cases[ok]
    )
  )

  if (
    !is.finite(value)
    ||
    value <= 0
  ) {

    return(NA_real_)

  }

  value
}


# ============================================================
# STANDARDISE USING TRAINING DATA ONLY
# ============================================================

scale_train_test <- function(
  train_data,
  test_data,
  features
) {

  train_scaled <- train_data

  test_scaled <- test_data


  for (feature in features) {

    center <- mean(
      train_data[[feature]],
      na.rm = TRUE
    )

    spread <- stats::sd(
      train_data[[feature]],
      na.rm = TRUE
    )


    if (!is.finite(center)) {
      return(NULL)
    }


    if (
      !is.finite(spread)
      ||
      spread == 0
    ) {

      train_scaled[[feature]] <- (
        train_data[[feature]]
        -
        center
      )

      test_scaled[[feature]] <- (
        test_data[[feature]]
        -
        center
      )

    } else {

      train_scaled[[feature]] <- (
        train_data[[feature]]
        -
        center
      ) / spread

      test_scaled[[feature]] <- (
        test_data[[feature]]
        -
        center
      ) / spread

    }

  }


  list(
    train = train_scaled,
    test = test_scaled
  )
}


# ============================================================
# FIT NEGATIVE-BINOMIAL MODEL SAFELY
# ============================================================

fit_nb_safe <- function(
  train_data,
  test_data,
  features,
  target_column
) {

  variable_features <- features[
    vapply(
      features,
      function(feature) {

        values <- train_data[[feature]]

        values <- values[
          is.finite(values)
        ]

        length(
          unique(values)
        ) > 1L

      },
      logical(1)
    )
  ]


  if (length(variable_features) == 0L) {

    message(
      "NB FIT ERROR: no usable variable predictors remain."
    )

    return(NULL)
  }


  scaled <- scale_train_test(
    train_data = train_data,
    test_data = test_data,
    features = variable_features
  )


  if (is.null(scaled)) {

    message(
      "NB FIT ERROR: predictor scaling failed."
    )

    return(NULL)
  }


  model_formula <- stats::as.formula(
    paste(
      target_column,
      "~",
      paste(
        variable_features,
        collapse = " + "
      )
    )
  )


  fitted_model <- tryCatch(
    {
      MASS::glm.nb(
        formula = model_formula,
        data = scaled$train,
        control = stats::glm.control(
          maxit = 200
        )
      )
    },
    error = function(e) {

      message(
        "NB FIT ERROR: ",
        conditionMessage(e)
      )

      NULL
    }
  )


  if (is.null(fitted_model)) {

    return(NULL)
  }


  prediction <- tryCatch(
    {
      stats::predict(
        fitted_model,
        newdata = scaled$test,
        type = "response"
      )
    },
    error = function(e) {

      message(
        "NB PREDICTION ERROR: ",
        conditionMessage(e)
      )

      NULL
    }
  )


  if (is.null(prediction)) {

    return(NULL)
  }


  prediction <- as.numeric(
    prediction
  )


  prediction <- pmax(
    prediction,
    0
  )


  return(
    list(
      model = fitted_model,
      prediction = prediction,
      features = variable_features
    )
  )
}


  

  




# ============================================================
# READ DATA
# ============================================================

if (!file.exists(DATA_FILE)) {

  cat(
    "ERROR: Missing input file:",
    DATA_FILE,
    "\n"
  )

  quit(
    save = "no",
    status = 1
  )

}


data <- utils::read.csv(
  DATA_FILE,
  stringsAsFactors = FALSE,
  check.names = FALSE
)


cat(
  "Rows loaded:",
  nrow(data),
  "\n"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns <- c(
  "setting",
  "source_year",
  "source_week",
  "cases"
)


missing_required <- setdiff(
  required_columns,
  names(data)
)


if (length(missing_required) > 0L) {

  cat(
    "ERROR: Missing required columns:",
    paste(
      missing_required,
      collapse = ", "
    ),
    "\n"
  )

  quit(
    save = "no",
    status = 1
  )

}


# ============================================================
# CLEAN CORE COLUMNS
# ============================================================

data$setting <- as.character(
  data$setting
)


data$source_year <- suppressWarnings(
  as.integer(
    data$source_year
  )
)


data$source_week <- suppressWarnings(
  as.integer(
    data$source_week
  )
)


data$cases <- suppressWarnings(
  as.numeric(
    data$cases
  )
)


data <- data[
  (
    !is.na(data$setting)
    &
    !is.na(data$source_year)
    &
    !is.na(data$source_week)
  ),
  ,
  drop = FALSE
]


data <- data[
  order(
    data$setting,
    data$source_year,
    data$source_week
  ),
  ,
  drop = FALSE
]


rownames(data) <- NULL


# ============================================================
# WEEK LABEL
# ============================================================

if (!("week_label" %in% names(data))) {

  data$week_label <- paste0(
    data$source_year,
    "-W",
    sprintf(
      "%02d",
      data$source_week
    )
  )

}


# ============================================================
# ANALYSIS WEEK
# ============================================================

if (!("analysis_week" %in% names(data))) {

  data$analysis_week <- ave(
    seq_len(
      nrow(data)
    ),
    data$setting,
    FUN = seq_along
  )

}


data$analysis_week <- suppressWarnings(
  as.integer(
    data$analysis_week
  )
)


data <- data[
  !is.na(
    data$analysis_week
  ),
  ,
  drop = FALSE
]


# ============================================================
# DETECT CLIMATE DATA
#
# First preference:
# already-created 4-week climate-lag variables.
#
# Otherwise:
# detect raw climate variables and construct lag 4.
# ============================================================

precomputed_rain <- first_existing(
  names(data),
  c(
    "log_rain_era5_lag_4",
    "rain_era5_lag_4",
    "rainfall_lag_4"
  )
)


precomputed_temp <- first_existing(
  names(data),
  c(
    "temp_era5_lag_4",
    "temperature_era5_lag_4",
    "temperature_lag_4"
  )
)


precomputed_humidity <- first_existing(
  names(data),
  c(
    "humidity_era5_lag_4",
    "relative_humidity_era5_lag_4",
    "humidity_lag_4"
  )
)


use_precomputed_climate <- (
  !is.null(precomputed_rain)
  &&
  !is.null(precomputed_temp)
  &&
  !is.null(precomputed_humidity)
)


if (!use_precomputed_climate) {

  raw_rain <- first_existing(
    names(data),
    c(
      "log_rain_era5",
      "rain_era5",
      "rainfall_era5",
      "rainfall",
      "Rainfall",
      "precipitation"
    )
  )


  raw_temp <- first_existing(
    names(data),
    c(
      "temp_era5",
      "temperature_era5",
      "temperature",
      "Temperature",
      "TempMean_MERRA2"
    )
  )


  raw_humidity <- first_existing(
    names(data),
    c(
      "humidity_era5",
      "relative_humidity_era5",
      "humidity",
      "Humidity",
      "RH"
    )
  )


  climate_available <- (
    !is.null(raw_rain)
    &&
    !is.null(raw_temp)
    &&
    !is.null(raw_humidity)
  )

} else {

  raw_rain <- NULL

  raw_temp <- NULL

  raw_humidity <- NULL

  climate_available <- TRUE

}


if (!climate_available) {

  cat(
    paste0(
      "WARNING: Complete climate columns were not detected. ",
      "The epidemiology-only model will still run.\n"
    )
  )

}


# ============================================================
# BUILD FEATURES WITHIN EACH SETTING
# ============================================================

build_group <- function(
  group
) {

  group <- group[
    order(
      group$analysis_week
    ),
    ,
    drop = FALSE
  ]


  if (
    anyDuplicated(
      group$analysis_week
    ) > 0L
  ) {

    cat(
      "WARNING: Duplicate analysis_week values in",
      unique(group$setting)[1L],
      "\n"
    )

  }


  # ----------------------------------------------------------
  # Current observed dengue information
  # ----------------------------------------------------------

  group$log_cases <- log1p(
    pmax(
      group$cases,
      0
    )
  )


  # ----------------------------------------------------------
  # Seasonality
  # ----------------------------------------------------------

  group$season_sin <- sin(
    2
    *
    pi
    *
    group$source_week
    /
    52
  )


  group$season_cos <- cos(
    2
    *
    pi
    *
    group$source_week
    /
    52
  )


  # ----------------------------------------------------------
  # Dengue history
  # ----------------------------------------------------------

  for (
    lag_value in c(
      1L,
      2L,
      4L,
      8L
    )
  ) {

    feature_name <- paste0(
      "log_cases_lag_",
      lag_value
    )


    group[[feature_name]] <- week_lookup(
      group$log_cases,
      group$analysis_week,
      -lag_value
    )

  }


  group$log_cases_roll_mean_4 <- rolling_past_mean(
    group$log_cases,
    group$analysis_week,
    4L
  )


  group$log_cases_roll_mean_8 <- rolling_past_mean(
    group$log_cases,
    group$analysis_week,
    8L
  )


  # ----------------------------------------------------------
  # Climate predictors
  # ----------------------------------------------------------

  if (climate_available) {

    if (use_precomputed_climate) {

      group$climate_rainfall_lag_4 <- suppressWarnings(
        as.numeric(
          group[[precomputed_rain]]
        )
      )


      group$climate_temperature_lag_4 <- suppressWarnings(
        as.numeric(
          group[[precomputed_temp]]
        )
      )


      group$climate_humidity_lag_4 <- suppressWarnings(
        as.numeric(
          group[[precomputed_humidity]]
        )
      )

    } else {

      rain_values <- suppressWarnings(
        as.numeric(
          group[[raw_rain]]
        )
      )


      temperature_values <- suppressWarnings(
        as.numeric(
          group[[raw_temp]]
        )
      )


      humidity_values <- suppressWarnings(
        as.numeric(
          group[[raw_humidity]]
        )
      )


      group$climate_rainfall_lag_4 <- week_lookup(
        rain_values,
        group$analysis_week,
        -4L
      )


      group$climate_temperature_lag_4 <- week_lookup(
        temperature_values,
        group$analysis_week,
        -4L
      )


      group$climate_humidity_lag_4 <- week_lookup(
        humidity_values,
        group$analysis_week,
        -4L
      )

    }

  }


  # ----------------------------------------------------------
  # FUTURE TARGETS
  #
  # These are the variables used for leakage-safe splitting.
  # ----------------------------------------------------------

  for (horizon in HORIZONS) {

    target_cases_col <- paste0(
      "target_cases_h",
      horizon
    )


    target_year_col <- paste0(
      "target_year_h",
      horizon
    )


    target_week_col <- paste0(
      "target_week_h",
      horizon
    )


    target_analysis_week_col <- paste0(
      "target_analysis_week_h",
      horizon
    )


    group[[target_cases_col]] <- week_lookup(
      group$cases,
      group$analysis_week,
      horizon
    )


    group[[target_year_col]] <- week_lookup(
      group$source_year,
      group$analysis_week,
      horizon
    )


    group[[target_week_col]] <- week_lookup(
      group$source_week,
      group$analysis_week,
      horizon
    )


    group[[target_analysis_week_col]] <- week_lookup(
      group$analysis_week,
      group$analysis_week,
      horizon
    )

  }


  group
}


# ============================================================
# APPLY FEATURE ENGINEERING TO EACH SETTING
# ============================================================

data <- do.call(
  rbind,
  lapply(
    split(
      data,
      data$setting
    ),
    build_group
  )
)


rownames(data) <- NULL


data <- data[
  order(
    data$setting,
    data$analysis_week
  ),
  ,
  drop = FALSE
]


# ============================================================
# MODEL FEATURE SETS
# ============================================================

epidemiology_features <- c(
  "log_cases",
  "log_cases_lag_1",
  "log_cases_lag_2",
  "log_cases_lag_4",
  "log_cases_lag_8",
  "log_cases_roll_mean_4",
  "log_cases_roll_mean_8",
  "season_sin",
  "season_cos"
)


climate_features <- c(
  "climate_rainfall_lag_4",
  "climate_temperature_lag_4",
  "climate_humidity_lag_4"
)


# ============================================================
# RESULT STORAGE
# ============================================================

results <- list()

prediction_frames <- list()

result_index <- 1L

prediction_index <- 1L


# ============================================================
# TEMPORAL MODEL LOOP
# ============================================================

for (
  setting_name in sort(
    unique(
      data$setting
    )
  )
) {

  setting_data <- data[
    data$setting == setting_name,
    ,
    drop = FALSE
  ]


  setting_data <- setting_data[
    order(
      setting_data$analysis_week
    ),
    ,
    drop = FALSE
  ]


  observed_years <- sort(
    unique(
      setting_data$source_year[
        is.finite(
          setting_data$cases
        )
      ]
    )
  )


  if (length(observed_years) < 3L) {

    cat(
      "Skipping",
      setting_name,
      ": fewer than three observed years.\n"
    )

    next

  }


  validation_year <- observed_years[
    length(observed_years) - 1L
  ]


  test_year <- observed_years[
    length(observed_years)
  ]


  cat(
    "\n",
    setting_name,
    ": validation target year=",
    validation_year,
    ", test target year=",
    test_year,
    "\n",
    sep = ""
  )


  # ----------------------------------------------------------
  # MASE denominator
  # ----------------------------------------------------------

  scale_data <- setting_data[
    setting_data$source_year < test_year,
    ,
    drop = FALSE
  ]


  scale <- mase_scale(
    scale_data
  )


  # ----------------------------------------------------------
  # EACH FORECAST HORIZON
  # ----------------------------------------------------------

  for (horizon in HORIZONS) {

    target_cases_col <- paste0(
      "target_cases_h",
      horizon
    )


    target_year_col <- paste0(
      "target_year_h",
      horizon
    )


    target_week_col <- paste0(
      "target_week_h",
      horizon
    )


    target_analysis_week_col <- paste0(
      "target_analysis_week_h",
      horizon
    )


    # --------------------------------------------------------
    # MODEL SETS
    # --------------------------------------------------------

    model_sets <- list(
      epidemiology_only =
        epidemiology_features
    )


    if (climate_available) {

      model_sets$climate_informed <- c(
        epidemiology_features,
        climate_features
      )


      common_features <- c(
        epidemiology_features,
        climate_features
      )

    } else {

      common_features <-
        epidemiology_features

    }


    # --------------------------------------------------------
    # COMMON COMPLETE COHORT
    #
    # This keeps climate and epidemiology NB models comparable.
    # --------------------------------------------------------

    required_complete <- unique(
      c(
        common_features,
        target_cases_col,
        target_year_col,
        target_week_col,
        target_analysis_week_col
      )
    )


    complete_data <- setting_data[
      stats::complete.cases(
        setting_data[
          ,
          required_complete,
          drop = FALSE
        ]
      ),
      ,
      drop = FALSE
    ]


    if (nrow(complete_data) == 0L) {

      cat(
        "Skipping ",
        setting_name,
        ", horizon=",
        horizon,
        ": no complete data.\n",
        sep = ""
      )

      next

    }


    finite_matrix <- sapply(
      required_complete,
      function(column) {

        is.finite(
          suppressWarnings(
            as.numeric(
              complete_data[[column]]
            )
          )
        )

      }
    )


    finite_rows <- apply(
      finite_matrix,
      1L,
      all
    )


    complete_data <- complete_data[
      finite_rows,
      ,
      drop = FALSE
    ]


    complete_data <- complete_data[
      complete_data[[target_cases_col]] >= 0,
      ,
      drop = FALSE
    ]


    # ========================================================
    # LEAKAGE-SAFE TEMPORAL SPLIT
    #
    # CRITICAL:
    # membership is based on TARGET YEAR.
    # ========================================================

    prevalidation_data <- complete_data[
      complete_data[[target_year_col]]
      <
      validation_year,
      ,
      drop = FALSE
    ]


    validation_data <- complete_data[
      complete_data[[target_year_col]]
      ==
      validation_year,
      ,
      drop = FALSE
    ]


    final_training <- complete_data[
      complete_data[[target_year_col]]
      <
      test_year,
      ,
      drop = FALSE
    ]


    test_data <- complete_data[
      complete_data[[target_year_col]]
      ==
      test_year,
      ,
      drop = FALSE
    ]


    # --------------------------------------------------------
    # MINIMUM DATA REQUIREMENT
    # --------------------------------------------------------

    if (
      nrow(final_training) < MIN_TRAIN_ROWS
      ||
      nrow(test_data) < MIN_TEST_ROWS
    ) {

      cat(
        "Skipping ",
        setting_name,
        ", horizon=",
        horizon,
        ": insufficient leakage-safe data ",
        "(pre-validation=",
        nrow(prevalidation_data),
        ", validation=",
        nrow(validation_data),
        ", final-training=",
        nrow(final_training),
        ", test=",
        nrow(test_data),
        ").\n",
        sep = ""
      )

      next

    }


    # --------------------------------------------------------
    # EXPLICIT LEAKAGE CHECKS
    # --------------------------------------------------------

    if (
      any(
        final_training[[target_year_col]]
        >=
        test_year
      )
    ) {

      cat(
        "ERROR: Training leakage detected for ",
        setting_name,
        ", horizon=",
        horizon,
        ".\n",
        sep = ""
      )

      next

    }


    if (
      any(
        test_data[[target_year_col]]
        !=
        test_year
      )
    ) {

      cat(
        "ERROR: Test leakage detected for ",
        setting_name,
        ", horizon=",
        horizon,
        ".\n",
        sep = ""
      )

      next

    }


    # ========================================================
    # FIT EACH MODEL
    # ========================================================

    for (
      model_name in names(
        model_sets
      )
    ) {

      features <- model_sets[[model_name]]


      fit_result <- fit_nb_safe(
        train_data = final_training,
        test_data = test_data,
        features = features,
        target_column = target_cases_col
      )


      if (is.null(fit_result)) {

        cat(
          "Skipping ",
          setting_name,
          ", horizon=",
          horizon,
          ", model=",
          model_name,
          ": model fit or prediction failed.\n",
          sep = ""
        )

        next

      }


      actual <- as.numeric(
        test_data[[target_cases_col]]
      )


      predicted <- as.numeric(
        fit_result$prediction
      )


      valid_prediction <- (
        is.finite(actual)
        &
        is.finite(predicted)
      )


      if (!any(valid_prediction)) {

        cat(
          "Skipping ",
          setting_name,
          ", horizon=",
          horizon,
          ", model=",
          model_name,
          ": no finite predictions.\n",
          sep = ""
        )

        next

      }


      evaluation_data <- test_data[
        valid_prediction,
        ,
        drop = FALSE
      ]


      actual_eval <- actual[
        valid_prediction
      ]


      predicted_eval <- predicted[
        valid_prediction
      ]


      # ------------------------------------------------------
      # METRICS
      # ------------------------------------------------------

      model_mae <- mae(
        actual_eval,
        predicted_eval
      )


      model_rmse <- rmse(
        actual_eval,
        predicted_eval
      )


      model_mase <- if (
        is.finite(scale)
        &&
        scale > 0
      ) {

        model_mae / scale

      } else {

        NA_real_

      }


      # ------------------------------------------------------
      # MODEL-LEVEL RESULTS
      # ------------------------------------------------------

      results[[result_index]] <- data.frame(

        setting =
          setting_name,

        validation_year =
          validation_year,

        test_year =
          test_year,

        horizon_weeks =
          horizon,

        model =
          model_name,

        n_prevalidation =
          nrow(
            prevalidation_data
          ),

        n_validation =
          nrow(
            validation_data
          ),

        n_final_training =
          nrow(
            final_training
          ),

        n_test =
          nrow(
            evaluation_data
          ),

        mae =
          model_mae,

        rmse =
          model_rmse,

        mase =
          model_mase,

        stringsAsFactors =
          FALSE
      )


      result_index <- (
        result_index + 1L
      )


      # ------------------------------------------------------
      # PREDICTION-LEVEL RESULTS
      # ------------------------------------------------------

      prediction_frames[[prediction_index]] <- data.frame(

        setting =
          setting_name,

        validation_year =
          validation_year,

        test_year =
          test_year,

        source_year =
          as.integer(
            evaluation_data$source_year
          ),

        source_week =
          as.integer(
            evaluation_data$source_week
          ),

        week_label =
          as.character(
            evaluation_data$week_label
          ),

        analysis_week =
          as.integer(
            evaluation_data$analysis_week
          ),

        target_year =
          as.integer(
            evaluation_data[[target_year_col]]
          ),

        target_week =
          as.integer(
            evaluation_data[[target_week_col]]
          ),

        target_analysis_week =
          as.integer(
            evaluation_data[[target_analysis_week_col]]
          ),

        horizon_weeks =
          horizon,

        model =
          model_name,

        actual =
          actual_eval,

        predicted =
          predicted_eval,

        stringsAsFactors =
          FALSE
      )


      prediction_index <- (
        prediction_index + 1L
      )

    }

  }

}


# ============================================================
# COMBINE RESULTS
# ============================================================

if (
  length(results) == 0L
  ||
  length(prediction_frames) == 0L
) {

  cat(
    "ERROR: No negative-binomial forecasts were successfully generated.\n"
  )

  quit(
    save = "no",
    status = 1
  )

}


results_df <- do.call(
  rbind,
  results
)


predictions_df <- do.call(
  rbind,
  prediction_frames
)


results_df <- results_df[
  order(
    results_df$setting,
    results_df$horizon_weeks,
    results_df$model
  ),
  ,
  drop = FALSE
]


predictions_df <- predictions_df[
  order(
    predictions_df$setting,
    predictions_df$horizon_weeks,
    predictions_df$model,
    predictions_df$target_analysis_week
  ),
  ,
  drop = FALSE
]


rownames(results_df) <- NULL

rownames(predictions_df) <- NULL


# ============================================================
# DIAGNOSTIC CHECKS
# ============================================================

prediction_key <- paste(
  predictions_df$setting,
  predictions_df$analysis_week,
  predictions_df$target_analysis_week,
  predictions_df$horizon_weeks,
  predictions_df$model,
  sep = "|"
)


duplicate_count <- sum(
  duplicated(
    prediction_key
  )
)


missing_actual <- sum(
  is.na(
    predictions_df$actual
  )
)


missing_predicted <- sum(
  is.na(
    predictions_df$predicted
  )
)


wrong_target_year <- sum(
  predictions_df$target_year
  !=
  predictions_df$test_year
)


cross_year_count <- sum(
  predictions_df$source_year
  !=
  predictions_df$target_year
)


# ============================================================
# HARD SAFETY CHECK
# ============================================================

if (
  duplicate_count > 0L
  ||
  missing_actual > 0L
  ||
  missing_predicted > 0L
  ||
  wrong_target_year > 0L
) {

  cat(
    "\nNEGATIVE-BINOMIAL SAFETY CHECK FAILED\n"
  )

  cat(
    "Missing actual:",
    missing_actual,
    "\n"
  )

  cat(
    "Missing predicted:",
    missing_predicted,
    "\n"
  )

  cat(
    "Duplicate forecast rows:",
    duplicate_count,
    "\n"
  )

  cat(
    "Rows with target_year != test_year:",
    wrong_target_year,
    "\n"
  )

  quit(
    save = "no",
    status = 1
  )

}


# ============================================================
# SUMMARY ACROSS SETTINGS
# ============================================================

summary_metrics <- stats::aggregate(

  cbind(
    mae,
    rmse,
    mase
  )
  ~
  horizon_weeks
  +
  model,

  data =
    results_df,

  FUN =
    stats::median,

  na.rm =
    TRUE
)


names(
  summary_metrics
)[
  names(summary_metrics) == "mae"
] <- "median_mae"


names(
  summary_metrics
)[
  names(summary_metrics) == "rmse"
] <- "median_rmse"


names(
  summary_metrics
)[
  names(summary_metrics) == "mase"
] <- "median_mase"


settings_count <- stats::aggregate(

  setting
  ~
  horizon_weeks
  +
  model,

  data =
    results_df,

  FUN =
    function(x) {
      length(
        unique(x)
      )
    }
)


names(
  settings_count
)[
  names(settings_count) == "setting"
] <- "settings_evaluated"


summary_df <- merge(

  summary_metrics,

  settings_count,

  by = c(
    "horizon_weeks",
    "model"
  ),

  all = TRUE,

  sort = TRUE
)


# ============================================================
# OUTPUT FILES
# ============================================================

results_file <- file.path(
  TABLE_DIR,
  "negative_binomial_results.csv"
)


summary_file <- file.path(
  TABLE_DIR,
  "negative_binomial_summary.csv"
)


prediction_file <- file.path(
  PREDICTION_DIR,
  "negative_binomial_predictions.csv"
)


# ============================================================
# SAVE RESULTS
# ============================================================

utils::write.csv(
  results_df,
  results_file,
  row.names = FALSE
)


utils::write.csv(
  summary_df,
  summary_file,
  row.names = FALSE
)


utils::write.csv(
  predictions_df,
  prediction_file,
  row.names = FALSE
)


# ============================================================
# PRINT SUMMARY
# ============================================================

cat(
  "\nMEDIAN PERFORMANCE ACROSS SETTINGS\n"
)


print(
  summary_df,
  row.names = FALSE
)


# ============================================================
# FINAL DIAGNOSTICS
# ============================================================

cat(
  "\nNEGATIVE-BINOMIAL PREDICTION DIAGNOSTICS\n"
)


cat(
  "Rows:",
  nrow(predictions_df),
  "\n"
)


cat(
  "Settings:",
  paste(
    sort(
      unique(
        predictions_df$setting
      )
    ),
    collapse = ", "
  ),
  "\n"
)


cat(
  "Horizons:",
  paste(
    sort(
      unique(
        predictions_df$horizon_weeks
      )
    ),
    collapse = ", "
  ),
  "\n"
)


cat(
  "Models:",
  paste(
    sort(
      unique(
        predictions_df$model
      )
    ),
    collapse = ", "
  ),
  "\n"
)


cat(
  "Missing actual:",
  missing_actual,
  "\n"
)


cat(
  "Missing predicted:",
  missing_predicted,
  "\n"
)


cat(
  "Duplicate forecast rows:",
  duplicate_count,
  "\n"
)


cat(
  "Rows with target_year != test_year:",
  wrong_target_year,
  "\n"
)


cat(
  "Cross-year test forecasts retained:",
  cross_year_count,
  "\n"
)


# ============================================================
# COMPLETE
# ============================================================

cat(
  "\nNEGATIVE-BINOMIAL FORECASTING COMPLETE\n"
)


cat(
  "Saved:",
  results_file,
  "\n"
)


cat(
  "Saved:",
  summary_file,
  "\n"
)


cat(
  "Saved:",
  prediction_file,
  "\n"
)