# JULIA AI/MACHINE LEARNING - Comprehensive Reference - by Richard Rembert
# Julia for high-performance scientific computing, machine learning, and AI
# with focus on speed, parallelism, and mathematical computing excellence

# ═══════════════════════════════════════════════════════════════════════════════
#                           1. SETUP AND PROJECT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

# Julia Installation and Setup:
# 1. Download Julia from https://julialang.org/downloads/
# 2. Install Julia packages using Pkg manager
# 3. Set up VS Code with Julia extension or use Jupyter notebooks
# 4. Configure environment for optimal performance

using Pkg

# Essential Package Installation
essential_packages = [
    # Core Data Science and ML
    "DataFrames", "CSV", "Statistics", "StatsBase", "StatsPlots",
    
    # Machine Learning
    "MLJ", "MLJModels", "MLJLinearModels", "MLJDecisionTreeInterface",
    "MLJXGBoostInterface", "MLJFlux", "ScikitLearn",
    
    # Deep Learning and Neural Networks
    "Flux", "MLUtils", "CUDA", "Zygote", "ChainRules",
    
    # Linear Algebra and Mathematics
    "LinearAlgebra", "SparseArrays", "Random", "Distributions",
    
    # Optimization
    "Optim", "JuMP", "Ipopt", "GLPK",
    
    # Parallel Computing
    "Distributed", "SharedArrays", "ThreadsX", "FLoops",
    
    # Data Visualization
    "Plots", "PlotlyJS", "StatsPlots", "Makie", "GLMakie",
    
    # Time Series and Signal Processing
    "DSP", "TimeseriesTools", "StateSpaceModels",
    
    # Text Processing and NLP
    "TextAnalysis", "Languages", "WordTokenizers",
    
    # Scientific Computing
    "DifferentialEquations", "ForwardDiff", "FiniteDiff",
    
    # Data Import/Export
    "JSON", "BSON", "HDF5", "FileIO", "JLD2",
    
    # Benchmarking and Performance
    "BenchmarkTools", "ProfileView", "TimerOutputs",
    
    # Utilities
    "ProgressMeter", "Dates", "Printf", "Logging"
]

# Install packages if not already installed
for package in essential_packages
    try
        eval(Meta.parse("using $package"))
        println("✓ $package already installed")
    catch
        println("Installing $package...")
        Pkg.add(package)
    end
end

# Load essential packages
using DataFrames, CSV, Statistics, StatsBase
using LinearAlgebra, Random, Distributions
using Plots, StatsPlots
using MLJ, MLJModels
using Flux
using BenchmarkTools
using ProgressMeter
using Printf

# Set random seed for reproducibility
Random.seed!(42)

# Configure plotting backend
plotlyjs()

# ═══════════════════════════════════════════════════════════════════════════════
#                           2. PERFORMANCE OPTIMIZATION SETUP
# ═══════════════════════════════════════════════════════════════════════════════

"""
Performance optimization utilities for Julia AI/ML applications.
"""
module PerformanceOptimizer

using CUDA, Distributed, SharedArrays, ThreadsX
using BenchmarkTools, TimerOutputs

export setup_parallel_environment, optimize_blas, benchmark_system, check_gpu_availability

"""
Set up parallel computing environment with optimal thread and process configuration.
"""
function setup_parallel_environment()
    # Configure threading
    num_threads = Threads.nthreads()
    println("Available Julia threads: $num_threads")
    
    if num_threads == 1
        println("⚠️  Warning: Running on single thread. Set JULIA_NUM_THREADS for better performance.")
    end
    
    # Set up distributed computing if needed
    if nprocs() == 1
        addprocs(min(4, Sys.CPU_THREADS ÷ 2))
        println("Added $(nprocs() - 1) worker processes")
    end
    
    @everywhere using SharedArrays, ProgressMeter
    
    return (threads=num_threads, processes=nprocs())
end

"""
Optimize BLAS configuration for mathematical operations.
"""
function optimize_blas()
    # Set optimal BLAS thread count
    BLAS.set_num_threads(min(8, Sys.CPU_THREADS))
    
    println("BLAS Configuration:")
    println("  Vendor: $(BLAS.vendor())")
    println("  Threads: $(BLAS.get_num_threads())")
    
    # Linear algebra performance test
    n = 1000
    A = randn(n, n)
    B = randn(n, n)
    
    blas_time = @belapsed $A * $B
    println("  Matrix multiplication benchmark (1000×1000): $(round(blas_time * 1000, digits=2)) ms")
    
    return blas_time
end

"""
Check GPU availability and configure CUDA if available.
"""
function check_gpu_availability()
    if CUDA.functional()
        devices = CUDA.devices()
        println("GPU Configuration:")
        println("  CUDA functional: ✓")
        println("  Available devices: $(length(devices))")
        
        for (i, device) in enumerate(devices)
            CUDA.device!(device)
            props = CUDA.properties(device)
            mem_info = CUDA.MemoryInfo()
            
            println("  Device $i: $(props.name)")
            println("    Compute capability: $(props.major).$(props.minor)")
            println("    Memory: $(round(mem_info.total / 1024^3, digits=1)) GB")
            println("    Free memory: $(round(mem_info.free / 1024^3, digits=1)) GB")
        end
        
        return true
    else
        println("GPU Configuration: CUDA not available")
        return false
    end
end

"""
Comprehensive system benchmark for ML workloads.
"""
function benchmark_system()
    println("=== Julia ML System Benchmark ===")
    
    # CPU benchmark
    println("\n1. CPU Performance:")
    cpu_result = @benchmark sum(randn(10^6))
    println("   Vector sum (1M elements): $(round(median(cpu_result.times) / 1e6, digits=2)) ms")
    
    # Memory bandwidth
    println("\n2. Memory Performance:")
    n = 10^7
    x = randn(n)
    mem_result = @benchmark copy($x)
    bandwidth = (n * sizeof(Float64) * 2) / (median(mem_result.times) / 1e9) / 1024^3
    println("   Memory bandwidth: $(round(bandwidth, digits=1)) GB/s")
    
    # Matrix operations
    println("\n3. Linear Algebra:")
    n = 2000
    A = randn(n, n)
    B = randn(n, n)
    matmul_result = @benchmark $A * $B
    println("   Matrix multiplication (2000×2000): $(round(median(matmul_result.times) / 1e6, digits=0)) ms")
    
    # Eigenvalue decomposition
    n = 1000
    A_sym = Symmetric(randn(n, n))
    eigen_result = @benchmark eigvals($A_sym)
    println("   Eigenvalue decomposition (1000×1000): $(round(median(eigen_result.times) / 1e6, digits=0)) ms")
    
    return (cpu=cpu_result, memory=mem_result, matmul=matmul_result, eigen=eigen_result)
end

end # module PerformanceOptimizer

# Initialize performance optimizations
using .PerformanceOptimizer
setup_info = setup_parallel_environment()
blas_time = optimize_blas()
gpu_available = check_gpu_availability()

# ═══════════════════════════════════════════════════════════════════════════════
#                           3. DATA STRUCTURES AND PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

"""
High-performance data preprocessing module optimized for Julia's type system.
"""
module DataPreprocessing

using DataFrames, Statistics, StatsBase
using LinearAlgebra, Random
using ProgressMeter

export DataProcessor, preprocess_pipeline, feature_engineering, split_data

"""
Advanced data preprocessing with Julia's type system and performance optimizations.
"""
struct DataProcessor{T<:Real}
    data::DataFrame
    target_column::Union{String, Symbol, Nothing}
    categorical_columns::Vector{String}
    numerical_columns::Vector{String}
    preprocessing_steps::Vector{String}
    scalers::Dict{String, Any}
    encoders::Dict{String, Any}
    
    function DataProcessor{T}(data::DataFrame, target_column=nothing) where T<:Real
        # Identify column types
        categorical_cols = String[]
        numerical_cols = String[]
        
        for col in names(data)
            if eltype(data[!, col]) <: Union{String, AbstractString} || 
               eltype(data[!, col]) <: CategoricalValue
                push!(categorical_cols, string(col))
            elseif eltype(data[!, col]) <: Real
                push!(numerical_cols, string(col))
            end
        end
        
        # Remove target from feature columns
        if target_column !== nothing
            target_str = string(target_column)
            filter!(x -> x != target_str, categorical_cols)
            filter!(x -> x != target_str, numerical_cols)
        end
        
        new{T}(data, target_column, categorical_cols, numerical_cols, 
                String[], Dict{String, Any}(), Dict{String, Any}())
    end
end

# Convenient constructor
DataProcessor(data::DataFrame, target_column=nothing) = DataProcessor{Float64}(data, target_column)

"""
Handle missing values with various strategies optimized for Julia.
"""
function handle_missing_values!(processor::DataProcessor{T}, 
                               strategy::Symbol=:median, 
                               threshold::Float64=0.5) where T
    
    data = processor.data
    n_rows = nrow(data)
    
    # Remove columns with too many missing values
    cols_to_remove = String[]
    for col in names(data)
        missing_ratio = count(ismissing, data[!, col]) / n_rows
        if missing_ratio > threshold
            push!(cols_to_remove, col)
        end
    end
    
    if !isempty(cols_to_remove)
        select!(data, Not(cols_to_remove))
        println("Removed $(length(cols_to_remove)) columns with >$(threshold*100)% missing values")
    end
    
    # Handle missing values in remaining columns
    for col in names(data)
        if any(ismissing, data[!, col])
            if col in processor.numerical_columns
                if strategy == :median
                    replacement = median(skipmissing(data[!, col]))
                elseif strategy == :mean
                    replacement = mean(skipmissing(data[!, col]))
                elseif strategy == :mode
                    replacement = mode(skipmissing(data[!, col]))
                else
                    replacement = zero(T)
                end
                
                data[!, col] = coalesce.(data[!, col], replacement)
            elseif col in processor.categorical_columns
                replacement = mode(skipmissing(data[!, col]))
                data[!, col] = coalesce.(data[!, col], replacement)
            end
        end
    end
    
    push!(processor.preprocessing_steps, "Missing values handled with $strategy strategy")
    return processor
end

"""
Detect and handle outliers using IQR or Z-score methods.
"""
function handle_outliers!(processor::DataProcessor{T}, 
                         method::Symbol=:iqr, 
                         threshold::Float64=1.5) where T
    
    data = processor.data
    outlier_count = 0
    
    for col in processor.numerical_columns
        if col in names(data)
            values = data[!, col]
            
            if method == :iqr
                q1, q3 = quantile(values, [0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                # Cap outliers instead of removing
                outliers = (values .< lower_bound) .| (values .> upper_bound)
                outlier_count += sum(outliers)
                
                data[!, col] = clamp.(values, lower_bound, upper_bound)
                
            elseif method == :zscore
                μ, σ = mean(values), std(values)
                z_scores = abs.((values .- μ) ./ σ)
                outliers = z_scores .> threshold
                outlier_count += sum(outliers)
                
                # Cap at threshold standard deviations
                data[values .> μ + threshold*σ, col] .= μ + threshold*σ
                data[values .< μ - threshold*σ, col] .= μ - threshold*σ
            end
        end
    end
    
    push!(processor.preprocessing_steps, 
          "Outliers handled using $method method ($(outlier_count) outliers capped)")
    return processor
end

"""
Encode categorical variables using one-hot or ordinal encoding.
"""
function encode_categorical!(processor::DataProcessor{T}, 
                           method::Symbol=:onehot,
                           max_categories::Int=10) where T
    
    data = processor.data
    
    for col in processor.categorical_columns
        if col in names(data)
            unique_vals = unique(skipmissing(data[!, col]))
            n_unique = length(unique_vals)
            
            if n_unique <= max_categories
                if method == :onehot
                    # One-hot encoding
                    for val in unique_vals
                        new_col = "$(col)_$(val)"
                        data[!, new_col] = Int.(data[!, col] .== val)
                    end
                    # Remove original column
                    select!(data, Not(col))
                    
                elseif method == :ordinal
                    # Ordinal encoding
                    val_to_int = Dict(val => i for (i, val) in enumerate(unique_vals))
                    data[!, col] = [get(val_to_int, val, 0) for val in data[!, col]]
                end
                
                processor.encoders[col] = unique_vals
            end
        end
    end
    
    push!(processor.preprocessing_steps, "Categorical variables encoded using $method method")
    return processor
end

"""
Scale numerical features using standardization or normalization.
"""
function scale_features!(processor::DataProcessor{T}, method::Symbol=:standardize) where T
    data = processor.data
    
    for col in processor.numerical_columns
        if col in names(data)
            values = data[!, col]
            
            if method == :standardize
                μ, σ = mean(values), std(values)
                if σ > 0
                    data[!, col] = (values .- μ) ./ σ
                    processor.scalers[col] = (mean=μ, std=σ, method=:standardize)
                end
                
            elseif method == :normalize
                min_val, max_val = minimum(values), maximum(values)
                if max_val > min_val
                    data[!, col] = (values .- min_val) ./ (max_val - min_val)
                    processor.scalers[col] = (min=min_val, max=max_val, method=:normalize)
                end
                
            elseif method == :robust
                med_val = median(values)
                mad_val = median(abs.(values .- med_val))
                if mad_val > 0
                    data[!, col] = (values .- med_val) ./ mad_val
                    processor.scalers[col] = (median=med_val, mad=mad_val, method=:robust)
                end
            end
        end
    end
    
    push!(processor.preprocessing_steps, "Features scaled using $method method")
    return processor
end

"""
Advanced feature engineering with polynomial and interaction features.
"""
function create_polynomial_features!(processor::DataProcessor{T}, degree::Int=2) where T
    data = processor.data
    original_cols = copy(processor.numerical_columns)
    
    # Polynomial features
    for col in original_cols[1:min(5, length(original_cols))]
        if col in names(data)
            values = data[!, col]
            
            for d in 2:degree
                new_col = "$(col)_poly$(d)"
                data[!, new_col] = values .^ d
                push!(processor.numerical_columns, new_col)
            end
        end
    end
    
    push!(processor.preprocessing_steps, "Polynomial features created up to degree $degree")
    return processor
end

"""
Create interaction features between numerical variables.
"""
function create_interaction_features!(processor::DataProcessor{T}, max_interactions::Int=10) where T
    data = processor.data
    original_cols = copy(processor.numerical_columns)
    interactions_created = 0
    
    for i in 1:min(5, length(original_cols))
        for j in (i+1):min(5, length(original_cols))
            if interactions_created >= max_interactions
                break
            end
            
            col1, col2 = original_cols[i], original_cols[j]
            if col1 in names(data) && col2 in names(data)
                new_col = "$(col1)_x_$(col2)"
                data[!, new_col] = data[!, col1] .* data[!, col2]
                push!(processor.numerical_columns, new_col)
                interactions_created += 1
            end
        end
        
        if interactions_created >= max_interactions
            break
        end
    end
    
    push!(processor.preprocessing_steps, "Created $interactions_created interaction features")
    return processor
end

"""
Complete preprocessing pipeline with optimal defaults.
"""
function preprocess_pipeline(data::DataFrame, target_column=nothing; 
                           missing_strategy::Symbol=:median,
                           outlier_method::Symbol=:iqr,
                           encoding_method::Symbol=:onehot,
                           scaling_method::Symbol=:standardize,
                           create_polynomials::Bool=false,
                           create_interactions::Bool=false) :: DataProcessor
    
    println("Starting preprocessing pipeline...")
    
    processor = DataProcessor(copy(data), target_column)
    
    # Apply preprocessing steps
    handle_missing_values!(processor, missing_strategy)
    handle_outliers!(processor, outlier_method)
    encode_categorical!(processor, encoding_method)
    
    if create_polynomials
        create_polynomial_features!(processor)
    end
    
    if create_interactions
        create_interaction_features!(processor)
    end
    
    scale_features!(processor, scaling_method)
    
    println("Preprocessing completed. Applied $(length(processor.preprocessing_steps)) steps:")
    for (i, step) in enumerate(processor.preprocessing_steps)
        println("  $i. $step")
    end
    
    return processor
end

"""
Efficient train-validation-test split with stratification support.
"""
function split_data(X::AbstractMatrix{T}, y::AbstractVector; 
                   train_ratio::Float64=0.7, 
                   val_ratio::Float64=0.15,
                   stratify::Bool=false,
                   random_state::Int=42) where T
    
    Random.seed!(random_state)
    n = size(X, 1)
    
    if stratify && eltype(y) <: Union{String, Symbol, Integer}
        # Stratified split for classification
        unique_classes = unique(y)
        train_indices = Int[]
        val_indices = Int[]
        test_indices = Int[]
        
        for class in unique_classes
            class_indices = findall(==(class), y)
            n_class = length(class_indices)
            
            # Shuffle class indices
            shuffle!(class_indices)
            
            # Calculate splits
            n_train = round(Int, n_class * train_ratio)
            n_val = round(Int, n_class * val_ratio)
            
            append!(train_indices, class_indices[1:n_train])
            append!(val_indices, class_indices[n_train+1:n_train+n_val])
            append!(test_indices, class_indices[n_train+n_val+1:end])
        end
        
        # Shuffle the final indices
        shuffle!(train_indices)
        shuffle!(val_indices)
        shuffle!(test_indices)
        
    else
        # Random split
        indices = shuffle(1:n)
        
        n_train = round(Int, n * train_ratio)
        n_val = round(Int, n * val_ratio)
        
        train_indices = indices[1:n_train]
        val_indices = indices[n_train+1:n_train+n_val]
        test_indices = indices[n_train+n_val+1:end]
    end
    
    # Create splits
    X_train, X_val, X_test = X[train_indices, :], X[val_indices, :], X[test_indices, :]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
    
    println("Data split complete:")
    println("  Training: $(length(train_indices)) samples")
    println("  Validation: $(length(val_indices)) samples") 
    println("  Test: $(length(test_indices)) samples")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test)
end

end # module DataPreprocessing

# ═══════════════════════════════════════════════════════════════════════════════
#                           4. MACHINE LEARNING ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

"""
High-performance machine learning implementations leveraging Julia's speed.
"""
module MachineLearning

using LinearAlgebra, Statistics, Random
using MLJ, MLJModels, MLJLinearModels
using DataFrames
using ProgressMeter
using Distributions

export LinearRegression, LogisticRegression, RandomForest, GradientBoosting
export KMeans, PCA, SVM, NeuralNetwork
export cross_validate, hyperparameter_tune, ensemble_predict

"""
High-performance linear regression with analytical solution.
"""
struct LinearRegression{T<:Real}
    coefficients::Vector{T}
    intercept::T
    fitted::Bool
    
    LinearRegression{T}() where T = new{T}(T[], zero(T), false)
end

LinearRegression() = LinearRegression{Float64}()

function fit!(model::LinearRegression{T}, X::AbstractMatrix{T}, y::AbstractVector{T}) where T
    # Add intercept column
    X_with_intercept = hcat(ones(T, size(X, 1)), X)
    
    # Analytical solution: θ = (X'X)^(-1)X'y
    θ = (X_with_intercept' * X_with_intercept) \ (X_with_intercept' * y)
    
    model.intercept = θ[1]
    model.coefficients = θ[2:end]
    model.fitted = true
    
    return model
end

function predict(model::LinearRegression{T}, X::AbstractMatrix{T}) where T
    @assert model.fitted "Model must be fitted before prediction"
    return X * model.coefficients .+ model.intercept
end

"""
Logistic regression with gradient descent optimization.
"""
mutable struct LogisticRegression{T<:Real}
    coefficients::Vector{T}
    intercept::T
    learning_rate::T
    max_iterations::Int
    tolerance::T
    fitted::Bool
    
    function LogisticRegression{T}(; learning_rate::T=T(0.01), 
                                  max_iterations::Int=1000,
                                  tolerance::T=T(1e-6)) where T
        new{T}(T[], zero(T), learning_rate, max_iterations, tolerance, false)
    end
end

LogisticRegression(; kwargs...) = LogisticRegression{Float64}(; kwargs...)

sigmoid(z::T) where T = one(T) / (one(T) + exp(-z))

function fit!(model::LogisticRegression{T}, X::AbstractMatrix{T}, y::AbstractVector{T}) where T
    n, p = size(X)
    
    # Initialize parameters
    model.coefficients = zeros(T, p)
    model.intercept = zero(T)
    
    @showprogress "Training logistic regression..." for iter in 1:model.max_iterations
        # Forward pass
        z = X * model.coefficients .+ model.intercept
        predictions = sigmoid.(z)
        
        # Compute cost
        cost = -mean(y .* log.(predictions .+ eps(T)) .+ 
                    (one(T) .- y) .* log.(one(T) .- predictions .+ eps(T)))
        
        # Compute gradients
        error = predictions .- y
        grad_coef = (X' * error) ./ n
        grad_intercept = mean(error)
        
        # Update parameters
        model.coefficients .-= model.learning_rate .* grad_coef
        model.intercept -= model.learning_rate * grad_intercept
        
        # Check convergence
        if norm(grad_coef) < model.tolerance && abs(grad_intercept) < model.tolerance
            println("Converged after $iter iterations")
            break
        end
    end
    
    model.fitted = true
    return model
end

function predict_proba(model::LogisticRegression{T}, X::AbstractMatrix{T}) where T
    @assert model.fitted "Model must be fitted before prediction"
    z = X * model.coefficients .+ model.intercept
    return sigmoid.(z)
end

function predict(model::LogisticRegression{T}, X::AbstractMatrix{T}) where T
    proba = predict_proba(model, X)
    return Int.(proba .> 0.5)
end

"""
K-Means clustering with K-means++ initialization.
"""
mutable struct KMeans{T<:Real}
    n_clusters::Int
    centroids::Matrix{T}
    labels::Vector{Int}
    max_iterations::Int
    tolerance::T
    fitted::Bool
    
    function KMeans{T}(n_clusters::Int; max_iterations::Int=300, tolerance::T=T(1e-4)) where T
        new{T}(n_clusters, Matrix{T}(undef, 0, 0), Int[], max_iterations, tolerance, false)
    end
end

KMeans(n_clusters::Int; kwargs...) = KMeans{Float64}(n_clusters; kwargs...)

"""
K-means++ initialization for better convergence.
"""
function kmeans_plus_plus_init(X::AbstractMatrix{T}, k::Int) where T
    n, d = size(X)
    centroids = zeros(T, k, d)
    
    # Choose first centroid randomly
    centroids[1, :] = X[rand(1:n), :]
    
    # Choose remaining centroids
    for i in 2:k
        # Compute distances to nearest centroid
        distances = zeros(T, n)
        for j in 1:n
            min_dist = Inf
            for c in 1:i-1
                dist = sum((X[j, :] .- centroids[c, :]) .^ 2)
                min_dist = min(min_dist, dist)
            end
            distances[j] = min_dist
        end
        
        # Choose next centroid with probability proportional to squared distance
        probs = distances ./ sum(distances)
        cumsum_probs = cumsum(probs)
        r = rand()
        next_idx = findfirst(x -> x >= r, cumsum_probs)
        centroids[i, :] = X[next_idx, :]
    end
    
    return centroids
end

function fit!(model::KMeans{T}, X::AbstractMatrix{T}) where T
    n, d = size(X)
    
    # Initialize centroids with K-means++
    model.centroids = kmeans_plus_plus_init(X, model.n_clusters)
    model.labels = zeros(Int, n)
    
    @showprogress "K-means clustering..." for iter in 1:model.max_iterations
        old_centroids = copy(model.centroids)
        
        # Assign points to nearest centroids
        for i in 1:n
            min_dist = Inf
            best_cluster = 1
            
            for k in 1:model.n_clusters
                dist = sum((X[i, :] .- model.centroids[k, :]) .^ 2)
                if dist < min_dist
                    min_dist = dist
                    best_cluster = k
                end
            end
            
            model.labels[i] = best_cluster
        end
        
        # Update centroids
        for k in 1:model.n_clusters
            cluster_points = X[model.labels .== k, :]
            if size(cluster_points, 1) > 0
                model.centroids[k, :] = mean(cluster_points, dims=1)[:]
            end
        end
        
        # Check convergence
        if norm(model.centroids - old_centroids) < model.tolerance
            println("K-means converged after $iter iterations")
            break
        end
    end
    
    model.fitted = true
    return model
end

function predict(model::KMeans{T}, X::AbstractMatrix{T}) where T
    @assert model.fitted "Model must be fitted before prediction"
    
    n = size(X, 1)
    labels = zeros(Int, n)
    
    for i in 1:n
        min_dist = Inf
        best_cluster = 1
        
        for k in 1:model.n_clusters
            dist = sum((X[i, :] .- model.centroids[k, :]) .^ 2)
            if dist < min_dist
                min_dist = dist
                best_cluster = k
            end
        end
        
        labels[i] = best_cluster
    end
    
    return labels
end

"""
Principal Component Analysis with SVD for numerical stability.
"""
mutable struct PCA{T<:Real}
    n_components::Int
    components::Matrix{T}
    explained_variance::Vector{T}
    explained_variance_ratio::Vector{T}
    mean::Vector{T}
    fitted::Bool
    
    function PCA{T}(n_components::Int) where T
        new{T}(n_components, Matrix{T}(undef, 0, 0), T[], T[], T[], false)
    end
end

PCA(n_components::Int) = PCA{Float64}(n