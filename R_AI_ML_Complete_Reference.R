# R AI/MACHINE LEARNING - Comprehensive Reference - by Richard Rembert
# R for Statistics, Data Science, and Machine Learning with modern packages
# and best practices for reproducible research and production ML workflows

# ═══════════════════════════════════════════════════════════════════════════════
#                           1. SETUP AND PROJECT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

# R Installation and Setup:
# 1. Install R from CRAN (https://cran.r-project.org/)
# 2. Install RStudio IDE (https://rstudio.com/)
# 3. Configure R environment and package management
# 4. Set up version control with Git
# 5. Install essential packages for data science and ML

# Essential Package Installation
if (!require(pacman)) {
  install.packages("pacman")
  library(pacman)
}

# Core Data Science Packages
pacman::p_load(
  # Data manipulation and analysis
  dplyr, tidyr, purrr, readr, tibble, stringr, forcats, lubridate,
  
  # Visualization
  ggplot2, plotly, gridExtra, corrplot, VIM, lattice,
  
  # Statistical modeling
  stats, MASS, car, broom, modelr,
  
  # Machine Learning
  caret, randomForest, e1071, glmnet, xgboost, neuralnet,
  rpart, rpart.plot, cluster, factoextra, 
  
  # Deep Learning
  torch, tensorflow, keras, reticulate,
  
  # Time Series
  forecast, tseries, xts, zoo, prophet,
  
  # Text Mining and NLP
  tm, quanteda, tidytext, textdata, wordcloud,
  
  # Data import/export
  haven, readxl, jsonlite, httr, rvest,
  
  # Parallel processing
  parallel, doParallel, foreach,
  
  # Model evaluation and metrics
  MLmetrics, pROC, ROCR, ModelMetrics,
  
  # Reporting and documentation
  knitr, rmarkdown, DT, shiny,
  
  # Utilities
  here, janitor, skimr, DataExplorer
)

# Project Structure Setup
create_project_structure <- function(project_name) {
  # Create main project directories
  dirs <- c(
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",
    "scripts",
    "models",
    "reports",
    "plots",
    "utils",
    "tests"
  )
  
  for (dir in dirs) {
    dir.create(file.path(project_name, dir), recursive = TRUE)
  }
  
  # Create essential files
  file.create(file.path(project_name, "README.md"))
  file.create(file.path(project_name, ".gitignore"))
  file.create(file.path(project_name, "requirements.R"))
  
  cat("Project structure created for:", project_name, "\n")
}

# Set global options for reproducibility
set.seed(42)
options(scipen = 999)  # Disable scientific notation
options(stringsAsFactors = FALSE)  # Don't auto-convert strings to factors

# ═══════════════════════════════════════════════════════════════════════════════
#                           2. DATA MANIPULATION AND PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

# Comprehensive Data Preprocessing Pipeline
DataPreprocessor <- R6Class("DataPreprocessor",
  public = list(
    data = NULL,
    target_column = NULL,
    categorical_columns = NULL,
    numerical_columns = NULL,
    preprocessing_steps = list(),
    
    initialize = function(data, target_column = NULL) {
      self$data <- data
      self$target_column <- target_column
      self$identify_column_types()
      cat("DataPreprocessor initialized with", nrow(data), "rows and", ncol(data), "columns\n")
    },
    
    identify_column_types = function() {
      self$categorical_columns <- names(select_if(self$data, is.factor)) %>%
        c(names(select_if(self$data, is.character)))
      
      self$numerical_columns <- names(select_if(self$data, is.numeric))
      
      if (!is.null(self$target_column)) {
        self$categorical_columns <- setdiff(self$categorical_columns, self$target_column)
        self$numerical_columns <- setdiff(self$numerical_columns, self$target_column)
      }
    },
    
    handle_missing_values = function(method = "median", threshold = 0.5) {
      # Remove columns with too many missing values
      missing_prop <- self$data %>%
        summarise_all(~sum(is.na(.)) / length(.)) %>%
        gather(column, missing_prop)
      
      cols_to_remove <- missing_prop %>%
        filter(missing_prop > threshold) %>%
        pull(column)
      
      if (length(cols_to_remove) > 0) {
        self$data <- self$data %>% select(-one_of(cols_to_remove))
        cat("Removed", length(cols_to_remove), "columns with >", threshold * 100, "% missing values\n")
      }
      
      # Handle missing values in remaining columns
      if (method == "median") {
        self$data <- self$data %>%
          mutate_if(is.numeric, ~ifelse(is.na(.), median(., na.rm = TRUE), .))
      } else if (method == "mean") {
        self$data <- self$data %>%
          mutate_if(is.numeric, ~ifelse(is.na(.), mean(., na.rm = TRUE), .))
      } else if (method == "mode") {
        self$data <- self$data %>%
          mutate_if(is.character, ~ifelse(is.na(.), names(sort(table(.), decreasing = TRUE))[1], .))
      }
      
      self$preprocessing_steps <- append(self$preprocessing_steps, 
                                         paste("Missing values handled using", method))
      return(self)
    },
    
    handle_outliers = function(method = "iqr", threshold = 1.5) {
      if (method == "iqr") {
        for (col in self$numerical_columns) {
          if (col %in% names(self$data)) {
            Q1 <- quantile(self$data[[col]], 0.25, na.rm = TRUE)
            Q3 <- quantile(self$data[[col]], 0.75, na.rm = TRUE)
            IQR <- Q3 - Q1
            
            lower_bound <- Q1 - threshold * IQR
            upper_bound <- Q3 + threshold * IQR
            
            outliers_count <- sum(self$data[[col]] < lower_bound | self$data[[col]] > upper_bound, na.rm = TRUE)
            
            # Cap outliers instead of removing them
            self$data[[col]] <- pmax(pmin(self$data[[col]], upper_bound), lower_bound)
            
            if (outliers_count > 0) {
              cat("Capped", outliers_count, "outliers in column", col, "\n")
            }
          }
        }
      }
      
      self$preprocessing_steps <- append(self$preprocessing_steps, 
                                         paste("Outliers handled using", method, "method"))
      return(self)
    },
    
    encode_categorical_variables = function(method = "one_hot") {
      if (method == "one_hot") {
        # One-hot encoding for categorical variables
        for (col in self$categorical_columns) {
          if (col %in% names(self$data) && length(unique(self$data[[col]])) <= 10) {
            # Create dummy variables
            dummy_vars <- model.matrix(~ . - 1, data = self$data[col])
            colnames(dummy_vars) <- paste0(col, "_", colnames(dummy_vars))
            
            # Remove original column and add dummy variables
            self$data <- self$data %>%
              select(-!!col) %>%
              bind_cols(as_tibble(dummy_vars))
          }
        }
      } else if (method == "label") {
        # Label encoding for categorical variables
        for (col in self$categorical_columns) {
          if (col %in% names(self$data)) {
            self$data[[col]] <- as.numeric(as.factor(self$data[[col]]))
          }
        }
      }
      
      self$preprocessing_steps <- append(self$preprocessing_steps, 
                                         paste("Categorical variables encoded using", method))
      return(self)
    },
    
    scale_features = function(method = "standardize") {
      numeric_cols <- names(select_if(self$data, is.numeric))
      
      if (!is.null(self$target_column)) {
        numeric_cols <- setdiff(numeric_cols, self$target_column)
      }
      
      if (method == "standardize") {
        self$data <- self$data %>%
          mutate_at(vars(one_of(numeric_cols)), ~scale(.) %>% as.vector())
      } else if (method == "normalize") {
        self$data <- self$data %>%
          mutate_at(vars(one_of(numeric_cols)), ~(. - min(., na.rm = TRUE)) / 
                      (max(., na.rm = TRUE) - min(., na.rm = TRUE)))
      }
      
      self$preprocessing_steps <- append(self$preprocessing_steps, 
                                         paste("Features scaled using", method))
      return(self)
    },
    
    create_features = function() {
      # Automated feature engineering
      numeric_cols <- names(select_if(self$data, is.numeric))
      
      if (length(numeric_cols) >= 2) {
        # Create interaction features for top variables
        for (i in 1:(min(3, length(numeric_cols) - 1))) {
          for (j in (i + 1):min(3, length(numeric_cols))) {
            new_col_name <- paste0(numeric_cols[i], "_x_", numeric_cols[j])
            self$data[[new_col_name]] <- self$data[[numeric_cols[i]]] * self$data[[numeric_cols[j]]]
          }
        }
      }
      
      # Create polynomial features
      for (col in numeric_cols[1:min(3, length(numeric_cols))]) {
        self$data[[paste0(col, "_squared")]] <- self$data[[col]]^2
      }
      
      self$preprocessing_steps <- append(self$preprocessing_steps, "Feature engineering completed")
      return(self)
    },
    
    get_processed_data = function() {
      return(self$data)
    },
    
    get_preprocessing_summary = function() {
      cat("Preprocessing Steps Applied:\n")
      for (i in seq_along(self$preprocessing_steps)) {
        cat(i, ".", self$preprocessing_steps[[i]], "\n")
      }
      
      cat("\nFinal Dataset Info:\n")
      cat("Rows:", nrow(self$data), "\n")
      cat("Columns:", ncol(self$data), "\n")
      cat("Missing values:", sum(is.na(self$data)), "\n")
    }
  )
)

# ═══════════════════════════════════════════════════════════════════════════════
#                           3. EXPLORATORY DATA ANALYSIS (EDA)
# ═══════════════════════════════════════════════════════════════════════════════

# Comprehensive EDA Functions
perform_eda <- function(data, target_column = NULL) {
  cat("=== EXPLORATORY DATA ANALYSIS ===\n\n")
  
  # Basic dataset information
  cat("Dataset Overview:\n")
  cat("Dimensions:", dim(data), "\n")
  cat("Memory usage:", format(object.size(data), units = "MB"), "\n\n")
  
  # Data types and missing values
  data_summary <- data %>%
    summarise_all(list(
      type = ~class(.)[1],
      missing = ~sum(is.na(.)),
      missing_pct = ~round(sum(is.na(.)) / length(.) * 100, 2),
      unique_values = ~n_distinct(.)
    )) %>%
    gather(metric, value) %>%
    separate(metric, into = c("column", "metric"), sep = "_(?=[^_]+$)") %>%
    spread(metric, value)
  
  print(data_summary)
  
  # Statistical summary for numerical variables
  numeric_vars <- names(select_if(data, is.numeric))
  if (length(numeric_vars) > 0) {
    cat("\n=== NUMERICAL VARIABLES SUMMARY ===\n")
    print(summary(data[numeric_vars]))
  }
  
  # Categorical variables summary
  categorical_vars <- names(select_if(data, function(x) is.factor(x) || is.character(x)))
  if (length(categorical_vars) > 0) {
    cat("\n=== CATEGORICAL VARIABLES SUMMARY ===\n")
    for (var in categorical_vars[1:min(5, length(categorical_vars))]) {
      cat("\n", var, ":\n")
      print(table(data[[var]], useNA = "always"))
    }
  }
  
  return(data_summary)
}

# Advanced visualization functions
create_correlation_plot <- function(data, method = "pearson") {
  numeric_data <- select_if(data, is.numeric)
  
  if (ncol(numeric_data) > 1) {
    cor_matrix <- cor(numeric_data, use = "complete.obs", method = method)
    
    # Create correlation heatmap
    corrplot(cor_matrix, 
             method = "color",
             type = "upper",
             order = "hclust",
             tl.cex = 0.8,
             tl.col = "black",
             tl.srt = 45,
             title = paste("Correlation Matrix (", str_to_title(method), ")", sep = ""))
    
    return(cor_matrix)
  }
}

create_distribution_plots <- function(data, ncol = 3) {
  numeric_vars <- names(select_if(data, is.numeric))
  
  if (length(numeric_vars) > 0) {
    plot_list <- list()
    
    for (var in numeric_vars) {
      p <- ggplot(data, aes_string(x = var)) +
        geom_histogram(bins = 30, fill = "skyblue", alpha = 0.7, color = "black") +
        geom_density(aes(y = ..count..), color = "red", size = 1) +
        labs(title = paste("Distribution of", var),
             x = var, y = "Frequency") +
        theme_minimal()
      
      plot_list[[var]] <- p
    }
    
    # Arrange plots in grid
    n_plots <- length(plot_list)
    n_pages <- ceiling(n_plots / (ncol * 3))
    
    for (page in 1:n_pages) {
      start_idx <- (page - 1) * ncol * 3 + 1
      end_idx <- min(page * ncol * 3, n_plots)
      
      grid.arrange(grobs = plot_list[start_idx:end_idx], ncol = ncol)
    }
  }
}

# Automated feature selection
perform_feature_selection <- function(data, target_column, method = "correlation", threshold = 0.05) {
  selected_features <- c()
  
  if (method == "correlation") {
    # Correlation-based feature selection
    numeric_data <- select_if(data, is.numeric)
    
    if (target_column %in% names(numeric_data)) {
      correlations <- cor(numeric_data, use = "complete.obs")[, target_column]
      selected_features <- names(correlations[abs(correlations) > threshold & 
                                               names(correlations) != target_column])
    }
  } else if (method == "variance") {
    # Remove low variance features
    numeric_data <- select_if(data, is.numeric)
    variances <- apply(numeric_data, 2, var, na.rm = TRUE)
    selected_features <- names(variances[variances > threshold])
  } else if (method == "univariate") {
    # Univariate statistical tests
    features <- setdiff(names(data), target_column)
    p_values <- c()
    
    for (feature in features) {
      if (is.numeric(data[[feature]]) && is.numeric(data[[target_column]])) {
        # Correlation test for numeric variables
        test_result <- cor.test(data[[feature]], data[[target_column]])
        p_values[feature] <- test_result$p.value
      } else if (is.factor(data[[feature]]) || is.character(data[[feature]])) {
        # Chi-square test for categorical variables
        if (is.numeric(data[[target_column]])) {
          # Convert numeric target to categorical for chi-square test
          target_cat <- cut(data[[target_column]], breaks = 3, labels = c("Low", "Medium", "High"))
          test_result <- chisq.test(table(data[[feature]], target_cat))
          p_values[feature] <- test_result$p.value
        }
      }
    }
    
    selected_features <- names(p_values[p_values < threshold])
  }
  
  cat("Feature selection using", method, "method:\n")
  cat("Selected", length(selected_features), "features out of", ncol(data) - 1, "\n")
  cat("Selected features:", paste(selected_features, collapse = ", "), "\n")
  
  return(selected_features)
}

# ═══════════════════════════════════════════════════════════════════════════════
#                           4. MACHINE LEARNING MODELS
# ═══════════════════════════════════════════════════════════════════════════════

# Universal ML Model Manager
MLModelManager <- R6Class("MLModelManager",
  public = list(
    data = NULL,
    target_column = NULL,
    problem_type = NULL,
    train_data = NULL,
    test_data = NULL,
    models = list(),
    results = list(),
    
    initialize = function(data, target_column, test_size = 0.2) {
      self$data <- data
      self$target_column <- target_column
      
      # Determine problem type
      if (is.numeric(data[[target_column]])) {
        unique_values <- length(unique(data[[target_column]]))
        self$problem_type <- if (unique_values > 10) "regression" else "classification"
      } else {
        self$problem_type <- "classification"
      }
      
      # Split data
      self$split_data(test_size)
      
      cat("MLModelManager initialized for", self$problem_type, "problem\n")
      cat("Training set:", nrow(self$train_data), "samples\n")
      cat("Test set:", nrow(self$test_data), "samples\n")
    },
    
    split_data = function(test_size) {
      set.seed(42)
      train_indices <- createDataPartition(self$data[[self$target_column]], 
                                            p = 1 - test_size, list = FALSE)
      self$train_data <- self$data[train_indices, ]
      self$test_data <- self$data[-train_indices, ]
    },
    
    train_linear_model = function() {
      if (self$problem_type == "regression") {
        formula_str <- paste(self$target_column, "~ .")
        model <- lm(as.formula(formula_str), data = self$train_data)
      } else {
        formula_str <- paste(self$target_column, "~ .")
        model <- glm(as.formula(formula_str), data = self$train_data, family = binomial())
      }
      
      self$models[["linear"]] <- model
      cat("Linear model trained successfully\n")
      return(model)
    },
    
    train_random_forest = function(ntree = 500, mtry = NULL) {
      if (is.null(mtry)) {
        mtry <- if (self$problem_type == "regression") {
          max(floor(ncol(self$train_data) / 3), 1)
        } else {
          floor(sqrt(ncol(self$train_data)))
        }
      }
      
      formula_str <- paste(self$target_column, "~ .")
      
      if (self$problem_type == "classification") {
        self$train_data[[self$target_column]] <- as.factor(self$train_data[[self$target_column]])
      }
      
      model <- randomForest(as.formula(formula_str), 
                           data = self$train_data,
                           ntree = ntree,
                           mtry = mtry,
                           importance = TRUE)
      
      self$models[["random_forest"]] <- model
      cat("Random Forest model trained successfully\n")
      return(model)
    },
    
    train_svm = function(kernel = "radial", cost = 1, gamma = "scale") {
      if (gamma == "scale") {
        gamma <- 1 / (ncol(self$train_data) - 1)
      }
      
      formula_str <- paste(self$target_column, "~ .")
      
      if (self$problem_type == "classification") {
        self$train_data[[self$target_column]] <- as.factor(self$train_data[[self$target_column]])
        model <- svm(as.formula(formula_str), 
                    data = self$train_data,
                    kernel = kernel,
                    cost = cost,
                    gamma = gamma,
                    probability = TRUE)
      } else {
        model <- svm(as.formula(formula_str), 
                    data = self$train_data,
                    kernel = kernel,
                    cost = cost,
                    gamma = gamma)
      }
      
      self$models[["svm"]] <- model
      cat("SVM model trained successfully\n")
      return(model)
    },
    
    train_xgboost = function(nrounds = 100, max_depth = 6, eta = 0.3) {
      # Prepare data for XGBoost
      feature_cols <- setdiff(names(self$train_data), self$target_column)
      
      train_matrix <- xgb.DMatrix(
        data = as.matrix(self$train_data[feature_cols]),
        label = if (self$problem_type == "regression") {
          self$train_data[[self$target_column]]
        } else {
          as.numeric(as.factor(self$train_data[[self$target_column]])) - 1
        }
      )
      
      objective <- if (self$problem_type == "regression") "reg:squarederror" else "binary:logistic"
      
      model <- xgb.train(
        data = train_matrix,
        objective = objective,
        nrounds = nrounds,
        max_depth = max_depth,
        eta = eta,
        verbose = 0
      )
      
      self$models[["xgboost"]] <- model
      cat("XGBoost model trained successfully\n")
      return(model)
    },
    
    train_neural_network = function(hidden = c(5, 3), threshold = 0.01) {
      if (self$problem_type == "regression") {
        formula_str <- paste(self$target_column, "~ .")
        model <- neuralnet(as.formula(formula_str),
                          data = self$train_data,
                          hidden = hidden,
                          threshold = threshold,
                          linear.output = TRUE)
      } else {
        # For classification, ensure binary encoding
        self$train_data[[self$target_column]] <- as.numeric(as.factor(self$train_data[[self$target_column]])) - 1
        formula_str <- paste(self$target_column, "~ .")
        model <- neuralnet(as.formula(formula_str),
                          data = self$train_data,
                          hidden = hidden,
                          threshold = threshold,
                          linear.output = FALSE)
      }
      
      self$models[["neural_network"]] <- model
      cat("Neural Network model trained successfully\n")
      return(model)
    },
    
    evaluate_models = function() {
      results <- list()
      
      for (model_name in names(self$models)) {
        cat("Evaluating", model_name, "...\n")
        
        # Make predictions
        predictions <- self$predict_model(model_name, self$test_data)
        actual <- self$test_data[[self$target_column]]
        
        if (self$problem_type == "regression") {
          # Regression metrics
          mse <- mean((predictions - actual)^2)
          rmse <- sqrt(mse)
          mae <- mean(abs(predictions - actual))
          r_squared <- cor(predictions, actual)^2
          
          results[[model_name]] <- list(
            mse = mse,
            rmse = rmse,
            mae = mae,
            r_squared = r_squared
          )
        } else {
          # Classification metrics
          if (model_name == "linear") {
            predictions <- ifelse(predictions > 0.5, 1, 0)
          }
          
          actual_factor <- as.factor(actual)
          predictions_factor <- as.factor(predictions)
          
          # Ensure same levels
          levels(predictions_factor) <- levels(actual_factor)
          
          cm <- confusionMatrix(predictions_factor, actual_factor)
          
          results[[model_name]] <- list(
            accuracy = cm$overall["Accuracy"],
            precision = cm$byClass["Precision"],
            recall = cm$byClass["Recall"],
            f1_score = cm$byClass["F1"],
            confusion_matrix = cm$table
          )
        }
      }
      
      self$results <- results
      self$print_evaluation_results()
      return(results)
    },
    
    predict_model = function(model_name, new_data) {
      model <- self$models[[model_name]]
      
      if (model_name == "linear") {
        predictions <- predict(model, new_data)
        if (self$problem_type == "classification") {
          predictions <- plogis(predictions)  # Convert to probabilities
        }
      } else if (model_name == "random_forest") {
        if (self$problem_type == "classification") {
          predictions <- predict(model, new_data, type = "prob")[, 2]
        } else {
          predictions <- predict(model, new_data)
        }
      } else if (model_name == "svm") {
        predictions <- predict(model, new_data)
        if (self$problem_type == "classification") {
          predictions <- as.numeric(predictions) - 1
        }
      } else if (model_name == "xgboost") {
        feature_cols <- setdiff(names(new_data), self$target_column)
        test_matrix <- xgb.DMatrix(data = as.matrix(new_data[feature_cols]))
        predictions <- predict(model, test_matrix)
      } else if (model_name == "neural_network") {
        predictions <- predict(model, new_data)
        if (self$problem_type == "classification") {
          predictions <- as.numeric(predictions > 0.5)
        }
      }
      
      return(predictions)
    },
    
    print_evaluation_results = function() {
      cat("\n=== MODEL EVALUATION RESULTS ===\n\n")
      
      if (self$problem_type == "regression") {
        # Create comparison table for regression
        comparison_df <- data.frame(
          Model = names(self$results),
          RMSE = sapply(self$results, function(x) round(x$rmse, 4)),
          MAE = sapply(self$results, function(x) round(x$mae, 4)),
          R_Squared = sapply(self$results, function(x) round(x$r_squared, 4))
        )
        
        # Sort by R-squared (descending)
        comparison_df <- comparison_df[order(comparison_df$R_Squared, decreasing = TRUE), ]
        print(comparison_df)
        
      } else {
        # Create comparison table for classification
        comparison_df <- data.frame(
          Model = names(self$results),
          Accuracy = sapply(self$results, function(x) round(x$accuracy, 4)),
          Precision = sapply(self$results, function(x) round(x$precision, 4)),
          Recall = sapply(self$results, function(x) round(x$recall, 4)),
          F1_Score = sapply(self$results, function(x) round(x$f1_score, 4))
        )
        
        # Sort by F1 score (descending)
        comparison_df <- comparison_df[order(comparison_df$F1_Score, decreasing = TRUE), ]
        print(comparison_df)
      }
      
      # Identify best model
      if (self$problem_type == "regression") {
        best_model <- comparison_df$Model[1]
        best_metric <- comparison_df$R_Squared[1]
        cat("\nBest model:", best_model, "with R-squared:", best_metric, "\n")
      } else {
        best_model <- comparison_df$Model[1]
        best_metric <- comparison_df$F1_Score[1]
        cat("\nBest model:", best_model, "with F1-score:", best_metric, "\n")
      }
    },
    
    get_feature_importance = function(model_name = "random_forest") {
      if (model_name == "random_forest" && "random_forest" %in% names(self$models)) {
        importance_data <- importance(self$models[["random_forest"]])
        
        # Create importance plot
        varImpPlot(self$models[["random_forest"]], 
                   main = "Feature Importance (Random Forest)")
        
        return(importance_data)
      } else if (model_name == "xgboost" && "xgboost" %in% names(self$models)) {
        importance_data <- xgb.importance(model = self$models[["xgboost"]])
        
        # Create importance plot
        xgb.plot.importance(importance_data, top_n = 10)
        
        return(importance_data)
      }
    },
    
    save_models = function(directory = "models") {
      if (!dir.exists(directory)) {
        dir.create(directory, recursive = TRUE)
      }
      
      for (model_name in names(self$models)) {
        filename <- file.path(directory, paste0(model_name, "_model.rds"))
        saveRDS(self$models[[model_name]], filename)
        cat("Saved", model_name, "model to", filename, "\n")
      }
      
      # Save results
      results_filename <- file.path(directory, "model_results.rds")
      saveRDS(self$results, results_filename)
      cat("Saved evaluation results to", results_filename, "\n")
    },
    
    load_models = function(directory = "models") {
      model_files <- list.files(directory, pattern = "*_model.rds", full.names = TRUE)
      
      for (file in model_files) {
        model_name <- gsub("_model.rds", "", basename(file))
        self$models[[model_name]] <- readRDS(file)
        cat("Loaded", model_name, "model from", file, "\n")
      }
      
      # Load results if available
      results_file <- file.path(directory, "model_results.rds")
      if (file.exists(results_file)) {
        self$results <- readRDS(results_file)
        cat("Loaded evaluation results\n")
      }
    }
  )
)

# ═══════════════════════════════════════════════════════════════════════════════
#                           5. ADVANCED STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# Advanced Statistical Testing Framework
StatisticalAnalyzer <- R6Class("StatisticalAnalyzer",
  public = list(
    data = NULL,
    results = list(),
    
    initialize = function(data) {
      self$data <- data
      cat("StatisticalAnalyzer initialized with", nrow(data), "observations\n")
    },
    
    # Normality tests
    test_normality = function(columns = NULL) {
      if (is.null(columns)) {
        columns <- names(select_if(self$data, is.numeric))
      }
      
      normality_results <- list()
      
      for (col in columns) {
        if (col %in% names(self$data)) {
          # Shapiro-Wilk test
          shapiro_test <- tryCatch({
            shapiro.test(self$data[[col]])
          }, error = function(e) NULL)
          
          # Kolmogorov-Smirnov test
          ks_test <- tryCatch({
            ks.test(self$data[[col]], "pnorm", 
                   mean = mean(self$data[[col]], na.rm = TRUE),
                   sd = sd(self$data[[col]], na.rm = TRUE))
          }, error = function(e) NULL)
          
          # Anderson-Darling test
          ad_test <- tryCatch({
            ad.test(self$data[[col]])
          }, error = function(e) NULL)
          
          normality_results[[col]] <- list(
            shapiro = shapiro_test,
            kolmogorov_smirnov = ks_test,
            anderson_darling = ad_test
          )
        }
      }
      
      self$results[["normality"]] <- normality_results
      self$print_normality_results(normality_results)
      return(normality_results)
    },
    
    print_normality_results = function(results) {
      cat("\n=== NORMALITY TEST RESULTS ===\n\n")
      
      for (col in names(results)) {
        cat("Variable:", col, "\n")
        
        if (!is.null(results[[col]]$shapiro)) {
          cat("  Shapiro-Wilk: p =", round(results[[col]]$shapiro$p.value, 4),
              ifelse(results[[col]]$shapiro$p.value > 0.05, "(Normal)", "(Non-normal)"), "\n")
        }
        
        if (!is.null(results[[col]]$kolmogorov_smirnov)) {
          cat("  Kolmogorov-Smirnov: p =", round(results[[col]]$kolmogorov_smirnov$p.value, 4),
              ifelse(results[[col]]$kolmogorov_smirnov$p.value > 0.05, "(Normal)", "(Non-normal)"), "\n")
        }
        
        cat("\n")
      }
    },
    
    # Hypothesis testing
    perform_t_test = function(variable1, variable2 = NULL, group_variable = NULL, 
                             paired = FALSE, alternative = "two.sided") {
      
      if (is.null(variable2) && !is.null(group_variable)) {
        # Independent samples t-test
        groups <- unique(self$data[[group_variable]])
        if (length(groups) == 2) {
          group1_data <- self$data[self$data[[group_variable]] == groups[1], variable1]
          group2_data <- self$data[self$data[[group_variable]] == groups[2], variable1]
          
          test_result <- t.test(group1_data, group2_data, 
                               paired = paired, alternative = alternative)
        }
      } else if (!is.null(variable2)) {
        # Paired or independent t-test
        test_result <- t.test(self$data[[variable1]], self$data[[variable2]], 
                             paired = paired, alternative = alternative)
      } else {
        # One-sample t-test
        test_result <- t.test(self$data[[variable1]], alternative = alternative)
      }
      
      self$results[["t_test"]] <- test_result
      
      cat("\n=== T-TEST RESULTS ===\n")
      print(test_result)
      
      return(test_result)
    },
    
    perform_chi_square_test = function(variable1, variable2) {
      contingency_table <- table(self$data[[variable1]], self$data[[variable2]])
      
      chi_square_result <- chisq.test(contingency_table)
      cramers_v <- sqrt(chi_square_result$statistic / (sum(contingency_table) * (min(dim(contingency_table)) - 1)))
      
      self$results[["chi_square"]] <- list(
        test = chi_square_result,
        contingency_table = contingency_table,
        cramers_v = cramers_v
      )
      
      cat("\n=== CHI-SQUARE TEST RESULTS ===\n")
      cat("Contingency Table:\n")
      print(contingency_table)
      cat("\nChi-square test:\n")
      print(chi_square_result)
      cat("Cramer's V (effect size):", round(cramers_v, 4), "\n")
      
      return(chi_square_result)
    },
    
    perform_anova = function(dependent_var, independent_var) {
      formula_str <- paste(dependent_var, "~", independent_var)
      anova_result <- aov(as.formula(formula_str), data = self$data)
      anova_summary <- summary(anova_result)
      
      # Post-hoc tests if significant
      tukey_result <- NULL
      if (anova_summary[[1]][["Pr(>F)"]][1] < 0.05) {
        tukey_result <- TukeyHSD(anova_result)
      }
      
      self$results[["anova"]] <- list(
        anova = anova_result,
        summary = anova_summary,
        tukey = tukey_result
      )
      
      cat("\n=== ANOVA RESULTS ===\n")
      print(anova_summary)
      
      if (!is.null(tukey_result)) {
        cat("\nTukey HSD Post-hoc Tests:\n")
        print(tukey_result)
      }
      
      return(anova_result)
    },
    
    # Correlation analysis
    correlation_analysis = function(method = "pearson", use = "complete.obs") {
      numeric_data <- select_if(self$data, is.numeric)
      
      if (ncol(numeric_data) >= 2) {
        cor_matrix <- cor(numeric_data, method = method, use = use)
        
        # Correlation significance test
        cor_test_results <- list()
        combinations <- combn(names(numeric_data), 2, simplify = FALSE)
        
        for (combo in combinations) {
          var1 <- combo[1]
          var2 <- combo[2]
          
          test_result <- cor.test(numeric_data[[var1]], numeric_data[[var2]], method = method)
          cor_test_results[[paste(var1, var2, sep = "_")]] <- test_result
        }
        
        self$results[["correlation"]] <- list(
          matrix = cor_matrix,
          tests = cor_test_results
        )
        
        # Create correlation plot
        corrplot(cor_matrix, method = "color", type = "upper", 
                order = "hclust", tl.cex = 0.8, tl.col = "black")
        
        return(cor_matrix)
      }
    },
    
    # Principal Component Analysis
    perform_pca = function(scale = TRUE, center = TRUE) {
      numeric_data <- select_if(self$data, is.numeric)
      
      if (ncol(numeric_data) >= 2) {
        pca_result <- prcomp(numeric_data, scale = scale, center = center)
        
        # Calculate proportion of variance explained
        var_explained <- summary(pca_result)$importance
        
        self$results[["pca"]] <- list(
          pca = pca_result,
          variance_explained = var_explained
        )
        
        # Create scree plot
        plot(pca_result, type = "l", main = "PCA Scree Plot")
        
        # Biplot
        biplot(pca_result, main = "PCA Biplot")
        
        cat("\n=== PCA RESULTS ===\n")
        print(var_explained)
        
        return(pca_result)
      }
    },
    
    # Regression diagnostics
    regression_diagnostics = function(model) {
      # Create diagnostic plots
      par(mfrow = c(2, 2))
      plot(model, main = "Regression Diagnostics")
      par(mfrow = c(1, 1))
      
      # Durbin-Watson test for autocorrelation
      dw_test <- durbinWatsonTest(model)
      
      # Breusch-Pagan test for heteroscedasticity
      bp_test <- bptest(model)
      
      # Shapiro-Wilk test for normality of residuals
      shapiro_test <- shapiro.test(residuals(model))
      
      diagnostics <- list(
        durbin_watson = dw_test,
        breusch_pagan = bp_test,
        shapiro_wilk = shapiro_test
      )
      
      self$results[["regression_diagnostics"]] <- diagnostics
      
      cat("\n=== REGRESSION DIAGNOSTICS ===\n")
      cat("Durbin-Watson Test (Autocorrelation):\n")
      print(dw_test)
      cat("\nBreusch-Pagan Test (Heteroscedasticity):\n")
      print(bp_test)
      cat("\nShapiro-Wilk Test (Normality of Residuals):\n")
      print(shapiro_test)
      
      return(diagnostics)
    }
  )
)

# ═══════════════════════════════════════════════════════════════════════════════
#                           6. TIME SERIES ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# Comprehensive Time Series Analysis
TimeSeriesAnalyzer <- R6Class("TimeSeriesAnalyzer",
  public = list(
    data = NULL,
    ts_data = NULL,
    models = list(),
    forecasts = list(),
    
    initialize = function(data, date_column, value_column, frequency = 12) {
      self$data <- data
      
      # Convert to time series
      if (is.character(data[[date_column]])) {
        dates <- as.Date(data[[date_column]])
      } else {
        dates <- data[[date_column]]
      }
      
      # Create time series object
      start_year <- as.numeric(format(min(dates), "%Y"))
      start_period <- as.numeric(format(min(dates), "%m"))
      
      self$ts_data <- ts(data[[value_column]], 
                        start = c(start_year, start_period), 
                        frequency = frequency)
      
      cat("TimeSeriesAnalyzer initialized with", length(self$ts_data), "observations\n")
      cat("Time series frequency:", frequency, "\n")
    },
    
    decompose_series = function(type = "additive") {
      decomposition <- decompose(self$ts_data, type = type)
      
      # Plot decomposition
      plot(decomposition, main = paste("Time Series Decomposition (", type, ")", sep = ""))
      
      return(decomposition)
    },
    
    test_stationarity = function() {
      # Augmented Dickey-Fuller test
      adf_test <- adf.test(self$ts_data)
      
      # KPSS test
      kpss_test <- kpss.test(self$ts_data)
      
      # Phillips-Perron test
      pp_test <- pp.test(self$ts_data)
      
      stationarity_results <- list(
        adf = adf_test,
        kpss = kpss_test,
        phillips_perron = pp_test
      )
      
      cat("\n=== STATIONARITY TESTS ===\n")
      cat("ADF Test: p =", round(adf_test$p.value, 4),
          ifelse(adf_test$p.value < 0.05, "(Stationary)", "(Non-stationary)"), "\n")
      cat("KPSS Test: p =", round(kpss_test$p.value, 4),
          ifelse(kpss_test$p.value > 0.05, "(Stationary)", "(Non-stationary)"), "\n")
      cat("PP Test: p =", round(pp_test$p.value, 4),
          ifelse(pp_test$p.value < 0.05, "(Stationary)", "(Non-stationary)"), "\n")
      
      return(stationarity_results)
    },
    
    difference_series = function(differences = 1, seasonal_differences = 0) {
      diff_data <- self$ts_data
      
      # Apply regular differencing
      for (i in 1:differences) {
        diff_data <- diff(diff_data)
      }
      
      # Apply seasonal differencing
      if (seasonal_differences > 0) {
        for (i in 1:seasonal_differences) {
          diff_data <- diff(diff_data, lag = frequency(self$ts_data))
        }
      }
      
      return(diff_data)
    },
    
    fit_arima = function(order = c(1, 1, 1), seasonal = c(0, 0, 0), auto = TRUE) {
      if (auto) {
        # Automatic ARIMA model selection
        arima_model <- auto.arima(self$ts_data, 
                                 seasonal = TRUE,
                                 stepwise = FALSE,
                                 approximation = FALSE)
      } else {
        # Manual ARIMA specification
        arima_model <- arima(self$ts_data, order = order, seasonal = seasonal)
      }
      
      self$models[["arima"]] <- arima_model
      
      cat("\n=== ARIMA MODEL ===\n")
      print(arima_model)
      
      # Model diagnostics
      checkresiduals(arima_model)
      
      return(arima_model)
    },
    
    fit_exponential_smoothing = function(model = "ZZZ") {
      # Exponential smoothing model
      ets_model <- ets(self$ts_data, model = model)
      
      self$models[["ets"]] <- ets_model
      
      cat("\n=== EXPONENTIAL SMOOTHING MODEL ===\n")
      print(ets_model)
      
      # Plot model components
      plot(ets_model, main = "Exponential Smoothing Components")
      
      return(ets_model)
    },
    
    fit_prophet = function() {
      # Prepare data for Prophet
      df <- data.frame(
        ds = as.Date(time(self$ts_data)),
        y = as.numeric(self$ts_data)
      )
      
      # Fit Prophet model
      prophet_model <- prophet(df)
      
      self$models[["prophet"]] <- prophet_model
      
      cat("\n=== PROPHET MODEL ===\n")
      cat("Prophet model fitted successfully\n")
      
      return(prophet_model)
    },
    
    forecast_models = function(h = 12) {
      forecasts <- list()
      
      for (model_name in names(self$models)) {
        if (model_name == "arima" || model_name == "ets") {
          forecast_result <- forecast(self$models[[model_name]], h = h)
          forecasts[[model_name]] <- forecast_result
          
          # Plot forecast
          plot(forecast_result, main = paste("Forecast -", str_to_title(model_name)))
          
        } else if (model_name == "prophet") {
          # Prophet forecast
          future <- make_future_dataframe(self$models[[model_name]], periods = h, freq = "month")
          forecast_result <- predict(self$models[[model_name]], future)
          forecasts[[model_name]] <- forecast_result
          
          # Plot Prophet forecast
          plot(self$models[[model_name]], forecast_result)
        }
      }
      
      self$forecasts <- forecasts
      return(forecasts)
    },
    
    evaluate_forecasts = function(test_data = NULL, h = 12) {
      if (is.null(test_data)) {
        # Use last h observations as test set
        train_length <- length(self$ts_data) - h
        train_data <- window(self$ts_data, end = time(self$ts_data)[train_length])
        test_data <- window(self$ts_data, start = time(self$ts_data)[train_length + 1])
      }
      
      results <- list()
      
      for (model_name in names(self$models)) {
        if (model_name == "arima" || model_name == "ets") {
          # Refit model on training data
          if (model_name == "arima") {
            temp_model <- auto.arima(train_data)
          } else {
            temp_model <- ets(train_data)
          }
          
          forecast_result <- forecast(temp_model, h = length(test_data))
          
          # Calculate accuracy metrics
          accuracy_metrics <- accuracy(forecast_result, test_data)
          results[[model_name]] <- accuracy_metrics
        }
      }
      
      cat("\n=== FORECAST ACCURACY ===\n")
      for (model_name in names(results)) {
        cat("\n", str_to_title(model_name), "Model:\n")
        print(results[[model_name]])
      }
      
      return(results)
    },
    
    seasonal_analysis = function() {
      # Seasonal decomposition
      seasonal_decomp <- stl(self$ts_data, s.window = "periodic")
      
      # Plot seasonal decomposition
      plot(seasonal_decomp, main = "STL Decomposition")
      
      # Seasonal plots
      seasonplot(self$ts_data, main = "Seasonal Plot")
      
      # Monthly/quarterly plots
      monthplot(self$ts_data, main = "Month Plot")
      
      return(seasonal_decomp)
    }
  )
)

# ═══════════════════════════════════════════════════════════════════════════════
#                           7. TEXT MINING AND NLP
# ═══════════════════════════════════════════════════════════════════════════════

# Natural Language Processing Pipeline
TextAnalyzer <- R6Class("TextAnalyzer",
  public = list(
    text_data = NULL,
    corpus = NULL,
    dtm = NULL,
    processed_texts = NULL,
    models = list(),
    
    initialize = function(text_data) {
      if (is.data.frame(text_data)) {
        self$text_data <- text_data
      } else {
        self$text_data <- data.frame(text = text_data, stringsAsFactors = FALSE)
      }
      
      cat("TextAnalyzer initialized with", nrow(self$text_data), "documents\n")
    },
    
    preprocess_text = function(text_column = "text", 
                              remove_punctuation = TRUE,
                              remove_numbers = TRUE,
                              convert_to_lowercase = TRUE,
                              remove_stopwords = TRUE,
                              stem_words = TRUE) {
      
      texts <- self$text_data[[text_column]]
      
      # Create corpus
      self$corpus <- VCorpus(VectorSource(texts))
      
      # Text preprocessing
      if (convert_to_lowercase) {
        self$corpus <- tm_map(self$corpus, content_transformer(tolower))
      }
      
      if (remove_punctuation) {
        self$corpus <- tm_map(self$corpus, removePunctuation)
      }
      
      if (remove_numbers) {
        self$corpus <- tm_map(self$corpus, removeNumbers)
      }
      
      # Remove extra whitespace
      self$corpus <- tm_map(self$corpus, stripWhitespace)
      
      if (remove_stopwords) {
        self$corpus <- tm_map(self$corpus, removeWords, stopwords("english"))
      }
      
      if (stem_words) {
        self$corpus <- tm_map(self$corpus, stemDocument)
      }
      
      # Convert back to text
      self$processed_texts <- sapply(self$corpus, as.character)
      
      cat("Text preprocessing completed\n")
      return(self$processed_texts)
    },
    
    create_document_term_matrix = function(min_doc_freq = 0.01, max_doc_freq = 0.95) {
      # Create document-term matrix
      self$dtm <- DocumentTermMatrix(self$corpus)
      
      # Remove sparse terms
      self$dtm <- removeSparseTerms(self$dtm, sparse = max_doc_freq)
      
      # Remove terms that appear in too few documents
      term_freq <- colSums(as.matrix(self$dtm))
      min_freq <- ceiling(nrow(self$dtm) * min_doc_freq)
      self$dtm <- self$dtm[, term_freq >= min_freq]
      
      cat("Document-term matrix created with", nrow(self$dtm), "documents and", ncol(self$dtm), "terms\n")
      
      return(self$dtm)
    },
    
    word_frequency_analysis = function(top_n = 20) {
      if (is.null(self$dtm)) {
        self$create_document_term_matrix()
      }
      
      # Calculate word frequencies
      word_freq <- colSums(as.matrix(self$dtm))
      word_freq <- sort(word_freq, decreasing = TRUE)
      
      # Create frequency plot
      top_words <- head(word_freq, top_n)
      
      barplot(top_words, las = 2, main = paste("Top", top_n, "Most Frequent Words"),
              ylab = "Frequency", col = "skyblue")
      
      # Word cloud
      wordcloud(names(word_freq), word_freq, max.words = 100, 
                colors = brewer.pal(8, "Dark2"))
      
      return(word_freq)
    },
    
    sentiment_analysis = function(text_column = "text", method = "afinn") {
      # Get sentiment lexicon
      if (method == "afinn") {
        sentiment_lexicon <- get_sentiments("afinn")
      } else if (method == "bing") {
        sentiment_lexicon <- get_sentiments("bing")
      } else if (method == "nrc") {
        sentiment_lexicon <- get_sentiments("nrc")
      }
      
      # Tokenize and analyze sentiment
      text_df <- data.frame(
        doc_id = 1:nrow(self$text_data),
        text = self$text_data[[text_column]],
        stringsAsFactors = FALSE
      )
      
      sentiment_scores <- text_df %>%
        unnest_tokens(word, text) %>%
        inner_join(sentiment_lexicon, by = "word") %>%
        group_by(doc_id) %>%
        summarise(
          sentiment_score = if (method == "afinn") sum(value) else sum(sentiment == "positive") - sum(sentiment == "negative"),
          .groups = "drop"
        )
      
      # Merge back with original data
      self$text_data$sentiment_score <- 0
      self$text_data$sentiment_score[sentiment_scores$doc_id] <- sentiment_scores$sentiment_score
      
      # Classify sentiment
      self$text_data$sentiment_class <- ifelse(self$text_data$sentiment_score > 0, "positive",
                                              ifelse(self$text_data$sentiment_score < 0, "negative", "neutral"))
      
      # Plot sentiment distribution
      ggplot(self$text_data, aes(x = sentiment_score)) +
        geom_histogram(bins = 30, fill = "lightblue", alpha = 0.7) +
        geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
        labs(title = "Sentiment Score Distribution", x = "Sentiment Score", y = "Frequency") +
        theme_minimal()
      
      return(self$text_data)
    },
    
    topic_modeling_lda = function(num_topics = 5, alpha = 0.1, beta = 0.1) {
      if (is.null(self$dtm)) {
        self$create_document_term_matrix()
      }
      
      # Convert to format required by topicmodels
      dtm_matrix <- as.matrix(self$dtm)
      
      # Remove empty documents
      row_sums <- rowSums(dtm_matrix)
      dtm_matrix <- dtm_matrix[row_sums > 0, ]
      
      # Fit LDA model
      lda_model <- LDA(dtm_matrix, k = num_topics, 
                      control = list(alpha = alpha, delta = beta, seed = 42))
      
      self$models[["lda"]] <- lda_model
      
      # Extract topics
      topics <- tidy(lda_model, matrix = "beta")
      
      # Extract document-topic probabilities
      doc_topics <- tidy(lda_model, matrix = "gamma")
      
      # Plot top terms per topic
      top_terms <- topics %>%
        group_by(topic) %>%
        top_n(10, beta) %>%
        ungroup() %>%
        arrange(topic, -beta)
      
      top_terms %>%
        mutate(term = reorder_within(term, beta, topic)) %>%
        ggplot(aes(term, beta, fill = factor(topic))) +
        geom_col(show.legend = FALSE) +
        facet_wrap(~ topic, scales = "free") +
        coord_flip() +
        scale_x_reordered() +
        labs(title = "Top Terms per Topic", x = "Terms", y = "Beta (Term-Topic Probability)") +
        theme_minimal()
      
      cat("LDA topic modeling completed with", num_topics, "topics\n")
      
      return(list(model = lda_model, topics = topics, doc_topics = doc_topics))
    },
    
    text_similarity = function(method = "cosine") {
      if (is.null(self$dtm)) {
        self$create_document_term_matrix()
      }
      
      # Convert to matrix
      dtm_matrix <- as.matrix(self$dtm)
      
      if (method == "cosine") {
        # Cosine similarity
        similarity_matrix <- dtm_matrix %*% t(dtm_matrix) / 
          (sqrt(rowSums(dtm_matrix^2)) %*% t(sqrt(rowSums(dtm_matrix^2))))
      } else if (method == "jaccard") {
        # Jaccard similarity
        binary_matrix <- (dtm_matrix > 0) * 1
        intersection <- binary_matrix %*% t(binary_matrix)
        union <- rowSums(binary_matrix) + rep(rowSums(binary_matrix), each = nrow(binary_matrix)) - intersection
        similarity_matrix <- intersection / union
      }
      
      # Plot heatmap
      corrplot(similarity_matrix, method = "color", type = "upper",
               title = paste("Document Similarity (", str_to_title(method), ")", sep = ""))
      
      return(similarity_matrix)
    },
    
    named_entity_recognition = function(text_column = "text") {
      # Simple NER using regex patterns (basic implementation)
      texts <- self$text_data[[text_column]]
      
      entities <- list()
      
      # Email pattern
      email_pattern <- "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"
      entities[["emails"]] <- unlist(str_extract_all(texts, email_pattern))
      
      # Phone number pattern (US format)
      phone_pattern <- "\\b(?:\\d{3}-)?\\d{3}-\\d{4}\\b|\\b\\(\\d{3}\\)\\s?\\d{3}-\\d{4}\\b"
      entities[["phones"]] <- unlist(str_extract_all(texts, phone_pattern))
      
      # URL pattern
      url_pattern <- "https?://[^\\s]+"
      entities[["urls"]] <- unlist(str_extract_all(texts, url_pattern))
      
      # Capitalize words (potential names)
      name_pattern <- "\\b[A-Z][a-z]+\\s[A-Z][a-z]+\\b"
      entities[["potential_names"]] <- unlist(str_extract_all(texts, name_pattern))
      
      cat("Named Entity Recognition completed\n")
      for (entity_type in names(entities)) {
        cat(str_to_title(entity_type), "found:", length(entities[[entity_type]]), "\n")
      }
      
      return(entities)
    }
  )
)

# ═══════════════════════════════════════════════════════════════════════════════
#                           8. DEEP LEARNING WITH R
# ═══════════════════════════════════════════════════════════════════════════════

# Deep Learning Framework
DeepLearningModel <- R6Class("DeepLearningModel",
  public = list(
    model = NULL,
    history = NULL,
    data = NULL,
    
    initialize = function() {
      # Check if TensorFlow/Keras is available
      if (!reticulate::py_module_available("tensorflow")) {
        cat("Installing TensorFlow...\n")
        install_tensorflow()
      }
      
      cat("DeepLearningModel initialized\n")
    },
    
    prepare_data = function(X, y, validation_split = 0.2, test_split = 0.2) {
      # Split data into train, validation, and test sets
      n <- nrow(X)
      
      # Create indices
      test_indices <- sample(1:n, size = floor(n * test_split))
      remaining_indices <- setdiff(1:n, test_indices)
      
      val_indices <- sample(remaining_indices, size = floor(length(remaining_indices) * validation_split))
      train_indices <- setdiff(remaining_indices, val_indices)
      
      # Split the data
      X_train <- X[train_indices, ]
      y_train <- y[train_indices]
      X_val <- X[val_indices, ]
      y_val <- y[val_indices]
      X_test <- X[test_indices, ]
      y_test <- y[test_indices]
      
      # Scale features
      means <- apply(X_train, 2, mean)
      sds <- apply(X_train, 2, sd)
      
      X_train <- scale(X_train)
      X_val <- scale(X_val, center = means, scale = sds)
      X_test <- scale(X_test, center = means, scale = sds)
      
      self$data <- list(
        X_train = X_train, y_train = y_train,
        X_val = X_val, y_val = y_val,
        X_test = X_test, y_test = y_test,
        means = means, sds = sds
      )
      
      cat("Data prepared - Train:", length(train_indices), 
          "Val:", length(val_indices), "Test:", length(test_indices), "\n")
      
      return(self$data)
    },
    
    build_dense_model = function(input_dim, hidden_layers = c(64, 32), 
                                output_dim = 1, activation = "relu", 
                                output_activation = "linear", dropout_rate = 0.2) {
      
      self$model <- keras_model_sequential()
      
      # Input layer
      self$model %>%
        layer_dense(units = hidden_layers[1], activation = activation, input_shape = input_dim) %>%
        layer_dropout(rate = dropout_rate)
      
      # Hidden layers
      if (length(hidden_layers) > 1) {
        for (i in 2:length(hidden_layers)) {
          self$model %>%
            layer_dense(units = hidden_layers[i], activation = activation) %>%
            layer_dropout(rate = dropout_rate)
        }
      }
      
      # Output layer
      self$model %>%
        layer_dense(units = output_dim, activation = output_activation)
      
      cat("Dense neural network built with", length(hidden_layers), "hidden layers\n")
      return(self$model)
    },
    
    build_cnn_model = function(input_shape, num_classes = 1, 
                              conv_layers = list(c(32, 3), c(64, 3), c(128, 3)),
                              dense_layers = c(128, 64)) {
      
      self$model <- keras_model_sequential()
      
      # Convolutional layers
      for (i in seq_along(conv_layers)) {
        if (i == 1) {
          self$model %>%
            layer_conv_2d(filters = conv_layers[[i]][1], 
                         kernel_size = c(conv_layers[[i]][2], conv_layers[[i]][2]),
                         activation = "relu", input_shape = input_shape) %>%
            layer_max_pooling_2d(pool_size = c(2, 2))
        } else {
          self$model %>%
            layer_conv_2d(filters = conv_layers[[i]][1], 
                         kernel_size = c(conv_layers[[i]][2], conv_layers[[i]][2]),
                         activation = "relu") %>%
            layer_max_pooling_2d(pool_size = c(2, 2))
        }
      }
      
      # Flatten for dense layers
      self$model %>% layer_flatten()
      
      # Dense layers
      for (units in dense_layers) {
        self$model %>%
          layer_dense(units = units, activation = "relu") %>%
          layer_dropout(rate = 0.5)
      }
      
      # Output layer
      if (num_classes == 1) {
        self$model %>% layer_dense(units = 1, activation = "sigmoid")
      } else {
        self$model %>% layer_dense(units = num_classes, activation = "softmax")
      }
      
      cat("CNN model built with", length(conv_layers), "convolutional layers\n")
      return(self$model)
    },
    
    compile_model = function(optimizer = "adam", loss = "mse", metrics = c("mae")) {
      self$model %>% compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrics
      )
      
      cat("Model compiled with optimizer:", optimizer, "and loss:", loss, "\n")
    },
    
    train_model = function(epochs = 100, batch_size = 32, verbose = 1,
                          early_stopping = TRUE, patience = 10) {
      
      callbacks <- list()
      
      if (early_stopping) {
        callbacks <- append(callbacks, 
                           callback_early_stopping(patience = patience, restore_best_weights = TRUE))
      }
      
      self$history <- self$model %>% fit(
        x = self$data$X_train,
        y = self$data$y_train,
        epochs = epochs,
        batch_size = batch_size,
        validation_data = list(self$data$X_val, self$data$y_val),
        callbacks = callbacks,
        verbose = verbose
      )
      
      # Plot training history
      plot(self$history)
      
      cat("Model training completed\n")
      return(self$history)
    },
    
    evaluate_model = function() {
      # Evaluate on test set
      test_results <- self$model %>% evaluate(
        x = self$data$X_test,
        y = self$data$y_test,
        verbose = 0
      )
      
      # Make predictions
      predictions <- self$model %>% predict(self$data$X_test)
      
      cat("\n=== MODEL EVALUATION ===\n")
      cat("Test Loss:", round(test_results[[1]], 4), "\n")
      
      if (length(test_results) > 1) {
        cat("Test Metric:", round(test_results[[2]], 4), "\n")
      }
      
      return(list(metrics = test_results, predictions = predictions))
    },
    
    save_model = function(filepath) {
      save_model_tf(self$model, filepath)
      cat("Model saved to", filepath, "\n")
    },
    
    load_model = function(filepath) {
      self$model <- load_model_tf(filepath)
      cat("Model loaded from", filepath, "\n")
    }
  )
)

# ═══════════════════════════════════════════════════════════════════════════════
#                           9. MODEL DEPLOYMENT AND PRODUCTION
# ═══════════════════════════════════════════════════════════════════════════════

# Model Deployment Framework
ModelDeployment <- R6Class("ModelDeployment",
  public = list(
    models = list(),
    api_endpoints = list(),
    
    initialize = function() {
      cat("ModelDeployment system initialized\n")
    },
    
    register_model = function(model, model_name, model_type = "ml") {
      self$models[[model_name]] <- list(
        model = model,
        type = model_type,
        created_at = Sys.time()
      )
      
      cat("Model", model_name, "registered successfully\n")
    },
    
    create_prediction_function = function(model_name, preprocessor = NULL) {
      if (!(model_name %in% names(self$models))) {
        stop("Model not found: ", model_name)
      }
      
      model_info <- self$models[[model_name]]
      model <- model_info$model
      
      prediction_function <- function(new_data) {
        tryCatch({
          # Apply preprocessing if provided
          if (!is.null(preprocessor)) {
            new_data <- preprocessor(new_data)
          }
          
          # Make predictions based on model type
          if (model_info$type == "random_forest") {
            predictions <- predict(model, new_data)
          } else if (model_info$type == "lm" || model_info$type == "glm") {
            predictions <- predict(model, new_data)
          } else if (model_info$type == "svm") {
            predictions <- predict(model, new_data)
          } else if (model_info$type == "xgboost") {
            feature_matrix <- xgb.DMatrix(data = as.matrix(new_data))
            predictions <- predict(model, feature_matrix)
          } else if (model_info$type == "keras") {
            predictions <- predict(model, as.matrix(new_data))
          } else {
            predictions <- predict(model, new_data)
          }
          
          return(list(
            predictions = predictions,
            status = "success",
            timestamp = Sys.time()
          ))
          
        }, error = function(e) {
          return(list(
            error = e$message,
            status = "error",
            timestamp = Sys.time()
          ))
        })
      }
      
      return(prediction_function)
    },
    
    create_rest_api = function(model_name, port = 8000) {
      # This would integrate with plumber for REST API
      cat("REST API endpoint created for model:", model_name, "\n")
      cat("API would be available at: http://localhost:", port, "/predict\n")
      
      # Example plumber API code (would be in separate file)
      api_code <- paste0('
#* @post /predict
function(req) {
  # Parse JSON input
  input_data <- jsonlite::fromJSON(req$postBody)
  
  # Make prediction
  prediction_func <- deployment$create_prediction_function("', model_name, '")
  result <- prediction_func(input_data)
  
  return(result)
}

#* @get /health
function() {
  return(list(status = "healthy", timestamp = Sys.time()))
}

#* @get /models
function() {
  return(list(available_models = names(deployment$models)))
}
')
      
      self$api_endpoints[[model_name]] <- list(
        code = api_code,
        port = port,
        created_at = Sys.time()
      )
      
      return(api_code)
    },
    
    batch_prediction = function(model_name, data_file, output_file = NULL) {
      if (!(model_name %in% names(self$models))) {
        stop("Model not found: ", model_name)
      }
      
      # Load data
      if (grepl("\\.csv$", data_file)) {
        data <- read.csv(data_file)
      } else if (grepl("\\.rds$", data_file)) {
        data <- readRDS(data_file)
      } else {
        stop("Unsupported file format")
      }
      
      # Make predictions
      prediction_func <- self$create_prediction_function(model_name)
      results <- prediction_func(data)
      
      # Add predictions to original data
      data$predictions <- results$predictions
      data$prediction_timestamp <- Sys.time()
      
      # Save results
      if (is.null(output_file)) {
        output_file <- paste0("predictions_", model_name, "_", 
                             format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv")
      }
      
      write.csv(data, output_file, row.names = FALSE)
      cat("Batch predictions saved to:", output_file, "\n")
      
      return(data)
    },
    
    model_monitoring = function(model_name, new_data, threshold = 0.05) {
      # Basic model monitoring for data drift
      if (!(model_name %in% names(self$models))) {
        stop("Model not found: ", model_name)
      }
      
      # Make predictions
      prediction_func <- self$create_prediction_function(model_name)
      results <- prediction_func(new_data)
      
      # Basic drift detection (simplified)
      predictions <- results$predictions
      
      monitoring_results <- list(
        model_name = model_name,
        num_predictions = length(predictions),
        mean_prediction = mean(predictions, na.rm = TRUE),
        std_prediction = sd(predictions, na.rm = TRUE),
        min_prediction = min(predictions, na.rm = TRUE),
        max_prediction = max(predictions, na.rm = TRUE),
        timestamp = Sys.time()
      )
      
      cat("Model monitoring completed for:", model_name, "\n")
      cat("Predictions summary:\n")
      print(monitoring_results)
      
      return(monitoring_results)
    }
  )
)

# ═══════════════════════════════════════════════════════════════════════════════
#                           10. ADVANCED ANALYTICS AND REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

# Automated Report Generator
ReportGenerator <- R6Class("ReportGenerator",
  public = list(
    data = NULL,
    analyses = list(),
    plots = list(),
    
    initialize = function(data) {
      self$data <- data
      cat("ReportGenerator initialized with", nrow(data), "observations\n")
    },
    
    generate_data_profiling_report = function(output_file = "data_profile_report.html") {
      # Use DataExplorer for automated profiling
      if (require(DataExplorer)) {
        create_report(self$data, output_file = output_file)
        cat("Data profiling report generated:", output_file, "\n")
      } else {
        cat("DataExplorer package required for automated reporting\n")
      }
    },
    
    generate_statistical_summary = function() {
      summary_stats <- list()
      
      # Basic statistics
      summary_stats$basic <- summary(self$data)
      
      # Data types
      summary_stats$data_types <- sapply(self$data, class)
      
      # Missing values
      summary_stats$missing_values <- sapply(self$data, function(x) sum(is.na(x)))
      
      # Unique values
      summary_stats$unique_values <- sapply(self$data, function(x) length(unique(x)))
      
      # Correlation matrix for numeric variables
      numeric_data <- select_if(self$data, is.numeric)
      if (ncol(numeric_data) > 1) {
        summary_stats$correlations <- cor(numeric_data, use = "complete.obs")
      }
      
      self$analyses[["statistical_summary"]] <- summary_stats
      
      return(summary_stats)
    },
    
    create_visualization_dashboard = function() {
      plots <- list()
      
      # Distribution plots for numeric variables
      numeric_vars <- names(select_if(self$data, is.numeric))
      if (length(numeric_vars) > 0) {
        for (var in numeric_vars[1:min(6, length(numeric_vars))]) {
          p <- ggplot(self$data, aes_string(x = var)) +
            geom_histogram(bins = 30, fill = "skyblue", alpha = 0.7) +
            labs(title = paste("Distribution of", var)) +
            theme_minimal()
          plots[[paste0("dist_", var)]] <- p
        }
      }
      
      # Box plots for categorical vs numeric
      categorical_vars <- names(select_if(self$data, function(x) is.factor(x) || is.character(x)))
      
      if (length(categorical_vars) > 0 && length(numeric_vars) > 0) {
        for (cat_var in categorical_vars[1:min(2, length(categorical_vars))]) {
          for (num_var in numeric_vars[1:min(2, length(numeric_vars))]) {
            if (length(unique(self$data[[cat_var]])) <= 10) {
              p <- ggplot(self$data, aes_string(x = cat_var, y = num_var)) +
                geom_boxplot(fill = "lightcoral", alpha = 0.7) +
                labs(title = paste(num_var, "by", cat_var)) +
                theme_minimal() +
                theme(axis.text.x = element_text(angle = 45, hjust = 1))
              plots[[paste0("box_", cat_var, "_", num_var)]] <- p
            }
          }
        }
      }
      
      # Correlation heatmap
      if (length(numeric_vars) > 1) {
        cor_matrix <- cor(self$data[numeric_vars], use = "complete.obs")
        cor_df <- expand.grid(Var1 = rownames(cor_matrix), Var2 = colnames(cor_matrix))
        cor_df$value <- as.vector(cor_matrix)
        
        p <- ggplot(cor_df, aes(Var1, Var2, fill = value)) +
          geom_tile() +
          scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
          labs(title = "Correlation Heatmap") +
          theme_minimal() +
          theme(axis.text.x = element_text(angle = 45, hjust = 1))
        plots[["correlation_heatmap"]] <- p
      }
      
      self$plots <- plots
      
      # Display plots
      for (plot_name in names(plots)) {
        print(plots[[plot_name]])
      }
      
      return(plots)
    },
    
    generate_model_comparison_report = function(model_results) {
      # Create comprehensive model comparison
      comparison_data <- data.frame()
      
      for (model_name in names(model_results)) {
        result <- model_results[[model_name]]
        
        if ("accuracy" %in% names(result)) {
          # Classification results
          row <- data.frame(
            Model = model_name,
            Accuracy = result$accuracy,
            Precision = result$precision,
            Recall = result$recall,
            F1_Score = result$f1_score
          )
        } else {
          # Regression results
          row <- data.frame(
            Model = model_name,
            RMSE = result$rmse,
            MAE = result$mae,
            R_Squared = result$r_squared
          )
        }
        
        comparison_data <- rbind(comparison_data, row)
      }
      
      # Create comparison plot
      if ("Accuracy" %in% names(comparison_data)) {
        p <- ggplot(comparison_data, aes(x = reorder(Model, F1_Score), y = F1_Score)) +
          geom_col(fill = "steelblue") +
          coord_flip() +
          labs(title = "Model Performance Comparison (F1 Score)", 
               x = "Model", y = "F1 Score") +
          theme_minimal()
      } else {
        p <- ggplot(comparison_data, aes(x = reorder(Model, R_Squared), y = R_Squared)) +
          geom_col(fill = "steelblue") +
          coord_flip() +
          labs(title = "Model Performance Comparison (R²)", 
               x = "Model", y = "R²") +
          theme_minimal()
      }
      
      print(p)
      
      return(comparison_data)
    },
    
    create_rmarkdown_report = function(output_file = "analysis_report.Rmd") {
      # Generate R Markdown template
      rmd_content <- '
---
title: "Automated Data Analysis Report"
author: "Generated by R AI/ML Framework"
date: "`r Sys.Date()`"
output: 
  html_document:
    toc: true
    toc_float: true
    theme: cosmo
    highlight: tango
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)
library(DT)
library(ggplot2)
library(dplyr)
```

# Data Overview

```{r data-summary}
# Display data summary
summary(data)
```

# Data Quality Assessment

```{r data-quality}
# Missing values analysis
missing_data <- data.frame(
  Variable = names(data),
  Missing_Count = sapply(data, function(x) sum(is.na(x))),
  Missing_Percentage = round(sapply(data, function(x) sum(is.na(x)) / length(x) * 100), 2)
)

DT::datatable(missing_data, caption = "Missing Values Analysis")
```

# Visualizations

```{r plots, fig.width=10, fig.height=6}
# Include generated plots
for (plot_name in names(plots)) {
  print(plots[[plot_name]])
}
```

# Statistical Analysis

```{r statistical-analysis}
# Include statistical test results
if (length(analyses) > 0) {
  for (analysis_name in names(analyses)) {
    cat("\\n\\n##", analysis_name, "\\n")
    print(analyses[[analysis_name]])
  }
}
```

# Conclusions and Recommendations

Based on the analysis performed, here are the key findings:

1. Data quality assessment shows...
2. Statistical tests reveal...
3. Model performance indicates...

## Next Steps

1. Address data quality issues
2. Consider additional feature engineering
3. Evaluate model performance in production
'
      
      writeLines(rmd_content, output_file)
      cat("R Markdown report template created:", output_file, "\n")
      
      return(output_file)
    }
  )
)

# ═══════════════════════════════════════════════════════════════════════════════
#                           11. EXAMPLE WORKFLOWS AND USE CASES
# ═══════════════════════════════════════════════════════════════════════════════

# Complete Machine Learning Pipeline Example
run_complete_ml_pipeline <- function(data_file, target_column, problem_type = "auto") {
  cat("=== COMPLETE MACHINE LEARNING PIPELINE ===\n\n")
  
  # 1. Load and explore data
  cat("1. Loading and exploring data...\n")
  if (grepl("\\.csv$", data_file)) {
    data <- read.csv(data_file, stringsAsFactors = FALSE)
  } else if (grepl("\\.rds$", data_file)) {
    data <- readRDS(data_file)
  } else {
    stop("Unsupported file format")
  }
  
  cat("Data loaded:", nrow(data), "rows,", ncol(data), "columns\n")
  
  # 2. Data preprocessing
  cat("\n2. Preprocessing data...\n")
  preprocessor <- DataPreprocessor$new(data, target_column)
  processed_data <- preprocessor$
    handle_missing_values()$
    handle_outliers()$
    encode_categorical_variables()$
    scale_features()$
    get_processed_data()
  
  # 3. Feature selection
  cat("\n3. Performing feature selection...\n")
  selected_features <- perform_feature_selection(processed_data, target_column, method = "correlation")
  
  if (length(selected_features) > 0) {
    final_data <- processed_data[, c(selected_features, target_column)]
  } else {
    final_data <- processed_data
  }
  
  # 4. Model training and evaluation
  cat("\n4. Training and evaluating models...\n")
  ml_manager <- MLModelManager$new(final_data, target_column)
  
  # Train multiple models
  ml_manager$train_linear_model()
  ml_manager$train_random_forest()
  ml_manager$train_svm()
  ml_manager$train_xgboost()
  
  # Evaluate models
  results <- ml_manager$evaluate_models()
  
  # 5. Feature importance
  cat("\n5. Analyzing feature importance...\n")
  importance <- ml_manager$get_feature_importance("random_forest")
  
  # 6. Save models
  cat("\n6. Saving models...\n")
  ml_manager$save_models("models")
  
  # 7. Generate report
  cat("\n7. Generating report...\n")
  report_gen <- ReportGenerator$new(data)
  report_gen$generate_statistical_summary()
  report_gen$create_visualization_dashboard()
  comparison_report <- report_gen$generate_model_comparison_report(results)
  
  cat("\n=== PIPELINE COMPLETED SUCCESSFULLY ===\n")
  
  return(list(
    processed_data = final_data,
    model_results = results,
    feature_importance = importance,
    ml_manager = ml_manager
  ))
}

# Time Series Forecasting Example
run_time_series_analysis <- function(data, date_col, value_col, forecast_horizon = 12) {
  cat("=== TIME SERIES ANALYSIS PIPELINE ===\n\n")
  
  # Initialize time series analyzer
  ts_analyzer <- TimeSeriesAnalyzer$new(data, date_col, value_col)
  
  # Decomposition
  cat("1. Decomposing time series...\n")
  decomposition <- ts_analyzer$decompose_series()
  
  # Stationarity tests
  cat("\n2. Testing stationarity...\n")
  stationarity <- ts_analyzer$test_stationarity()
  
  # Seasonal analysis
  cat("\n3. Analyzing seasonality...\n")
  seasonal_analysis <- ts_analyzer$seasonal_analysis()
  
  # Model fitting
  cat("\n4. Fitting forecasting models...\n")
  arima_model <- ts_analyzer$fit_arima()
  ets_model <- ts_analyzer$fit_exponential_smoothing()
  
  # Forecasting
  cat("\n5. Generating forecasts...\n")
  forecasts <- ts_analyzer$forecast_models(h = forecast_horizon)
  
  # Model evaluation
  cat("\n6. Evaluating forecast accuracy...\n")
  accuracy <- ts_analyzer$evaluate_forecasts(h = forecast_horizon)
  
  cat("\n=== TIME SERIES ANALYSIS COMPLETED ===\n")
  
  return(list(
    ts_analyzer = ts_analyzer,
    forecasts = forecasts,
    accuracy = accuracy
  ))
}

# Text Analysis Example
run_text_analysis_pipeline <- function(text_data, text_column = "text") {
  cat("=== TEXT ANALYSIS PIPELINE ===\n\n")
  
  # Initialize text analyzer
  text_analyzer <- TextAnalyzer$new(text_data)
  
  # Preprocessing
  cat("1. Preprocessing text...\n")
  processed_texts <- text_analyzer$preprocess_text(text_column)
  
  # Create document-term matrix
  cat("\n2. Creating document-term matrix...\n")
  dtm <- text_analyzer$create_document_term_matrix()
  
  # Word frequency analysis
  cat("\n3. Analyzing word frequencies...\n")
  word_freq <- text_analyzer$word_frequency_analysis()
  
  # Sentiment analysis
  cat("\n4. Performing sentiment analysis...\n")
  sentiment_results <- text_analyzer$sentiment_analysis(text_column)
  
  # Topic modeling
  cat("\n5. Performing topic modeling...\n")
  topic_results <- text_analyzer$topic_modeling_lda(num_topics = 5)
  
  # Text similarity
  cat("\n6. Computing text similarity...\n")
  similarity_matrix <- text_analyzer$text_similarity()
  
  # Named entity recognition
  cat("\n7. Extracting named entities...\n")
  entities <- text_analyzer$named_entity_recognition(text_column)
  
  cat("\n=== TEXT ANALYSIS COMPLETED ===\n")
  
  return(list(
    text_analyzer = text_analyzer,
    sentiment_results = sentiment_results,
    topic_results = topic_results,
    entities = entities
  ))
}