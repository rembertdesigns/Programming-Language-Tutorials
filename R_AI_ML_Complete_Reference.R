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