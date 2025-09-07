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

PCA(n_components::Int) = PCA{Float64}(n_components)

function fit!(model::PCA{T}, X::AbstractMatrix{T}) where T
    n, p = size(X)
    
    # Center the data
    model.mean = mean(X, dims=1)[:]
    X_centered = X .- model.mean'
    
    # Perform SVD
    U, S, V = svd(X_centered)
    
    # Store components and explained variance
    n_comp = min(model.n_components, p, n-1)
    model.components = V[:, 1:n_comp]'
    
    # Calculate explained variance
    total_var = sum(S .^ 2) / (n - 1)
    model.explained_variance = (S[1:n_comp] .^ 2) ./ (n - 1)
    model.explained_variance_ratio = model.explained_variance ./ total_var
    
    model.fitted = true
    
    println("PCA fitted: $(sum(model.explained_variance_ratio) * 100)% variance explained by $n_comp components")
    return model
end

function transform(model::PCA{T}, X::AbstractMatrix{T}) where T
    @assert model.fitted "Model must be fitted before transformation"
    X_centered = X .- model.mean'
    return X_centered * model.components'
end

function fit_transform!(model::PCA{T}, X::AbstractMatrix{T}) where T
    fit!(model, X)
    return transform(model, X)
end

"""
Cross-validation with multiple performance metrics.
"""
function cross_validate(model_constructor, X::AbstractMatrix{T}, y::AbstractVector, 
                       cv_folds::Int=5; metrics=[:accuracy], random_state::Int=42) where T
    
    Random.seed!(random_state)
    n = size(X, 1)
    fold_size = n ÷ cv_folds
    indices = shuffle(1:n)
    
    results = Dict(metric => Float64[] for metric in metrics)
    
    @showprogress "Cross-validation..." for fold in 1:cv_folds
        # Create train/test split for this fold
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == cv_folds ? n : fold * fold_size
        
        test_indices = indices[test_start:test_end]
        train_indices = setdiff(indices, test_indices)
        
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Train model
        model = model_constructor()
        fit!(model, X_train, y_train)
        
        # Make predictions
        if hasmethod(predict_proba, (typeof(model), typeof(X_test)))
            y_pred_proba = predict_proba(model, X_test)
            y_pred = Int.(y_pred_proba .> 0.5)
        else
            y_pred = predict(model, X_test)
        end
        
        # Calculate metrics
        for metric in metrics
            if metric == :accuracy
                score = mean(y_pred .== y_test)
            elseif metric == :mse
                score = mean((y_pred .- y_test) .^ 2)
            elseif metric == :mae
                score = mean(abs.(y_pred .- y_test))
            elseif metric == :r2
                ss_res = sum((y_test .- y_pred) .^ 2)
                ss_tot = sum((y_test .- mean(y_test)) .^ 2)
                score = 1 - ss_res / ss_tot
            end
            
            push!(results[metric], score)
        end
    end
    
    # Print results
    println("\nCross-validation results:")
    for metric in metrics
        scores = results[metric]
        println("  $(metric): $(round(mean(scores), digits=4)) ± $(round(std(scores), digits=4))")
    end
    
    return results
end

end # module MachineLearning

# ═══════════════════════════════════════════════════════════════════════════════
#                           5. DEEP LEARNING WITH FLUX.jl
# ═══════════════════════════════════════════════════════════════════════════════

"""
High-performance deep learning implementations using Flux.jl.
"""
module DeepLearning

using Flux, MLUtils
using CUDA
using Statistics, Random
using ProgressMeter
using Plots

export NeuralNetwork, ConvolutionalNetwork, RecurrentNetwork, Autoencoder
export train_model!, evaluate_model, create_optimizer, plot_training_history

"""
Flexible neural network architecture builder.
"""
struct NeuralNetwork
    model::Chain
    optimizer
    loss_function
    metrics::Vector{Function}
    training_history::Dict{String, Vector{Float64}}
    
    function NeuralNetwork(input_dim::Int, hidden_dims::Vector{Int}, output_dim::Int;
                          activation=relu, output_activation=identity,
                          dropout_rate::Float64=0.0, use_batch_norm::Bool=false)
        
        layers = []
        
        # Input to first hidden layer
        push!(layers, Dense(input_dim, hidden_dims[1], activation))
        if use_batch_norm
            push!(layers, BatchNorm(hidden_dims[1]))
        end
        if dropout_rate > 0
            push!(layers, Dropout(dropout_rate))
        end
        
        # Hidden layers
        for i in 1:length(hidden_dims)-1
            push!(layers, Dense(hidden_dims[i], hidden_dims[i+1], activation))
            if use_batch_norm
                push!(layers, BatchNorm(hidden_dims[i+1]))
            end
            if dropout_rate > 0
                push!(layers, Dropout(dropout_rate))
            end
        end
        
        # Output layer
        push!(layers, Dense(hidden_dims[end], output_dim, output_activation))
        
        model = Chain(layers...)
        optimizer = Adam(0.001)
        loss_function = Flux.mse
        metrics = [mse_metric]
        training_history = Dict("train_loss" => Float64[], "val_loss" => Float64[])
        
        new(model, optimizer, loss_function, metrics, training_history)
    end
end

# Metric functions
mse_metric(ŷ, y) = mean((ŷ .- y) .^ 2)
mae_metric(ŷ, y) = mean(abs.(ŷ .- y))
accuracy_metric(ŷ, y) = mean(Flux.onecold(ŷ) .== Flux.onecold(y))

"""
Convolutional Neural Network for image data.
"""
struct ConvolutionalNetwork
    model::Chain
    optimizer
    loss_function
    training_history::Dict{String, Vector{Float64}}
    
    function ConvolutionalNetwork(input_shape::Tuple, num_classes::Int;
                                conv_layers::Vector{Tuple{Int,Int}}=[(32,3), (64,3), (128,3)],
                                dense_layers::Vector{Int}=[128, 64])
        
        layers = []
        
        # Convolutional layers
        in_channels = input_shape[3]  # Assuming (height, width, channels, batch)
        
        for (out_channels, kernel_size) in conv_layers
            push!(layers, Conv((kernel_size, kernel_size), in_channels => out_channels, relu))
            push!(layers, BatchNorm(out_channels))
            push!(layers, MaxPool((2, 2)))
            in_channels = out_channels
        end
        
        # Flatten
        push!(layers, Flux.flatten)
        
        # Calculate flattened size (approximate)
        flattened_size = div(input_shape[1], 2^length(conv_layers)) * 
                        div(input_shape[2], 2^length(conv_layers)) * in_channels
        
        # Dense layers
        prev_size = flattened_size
        for hidden_size in dense_layers
            push!(layers, Dense(prev_size, hidden_size, relu))
            push!(layers, Dropout(0.5))
            prev_size = hidden_size
        end
        
        # Output layer
        if num_classes == 1
            push!(layers, Dense(prev_size, 1, sigmoid))  # Binary classification
        else
            push!(layers, Dense(prev_size, num_classes))  # Multi-class
        end
        
        model = Chain(layers...)
        optimizer = Adam(0.001)
        loss_function = num_classes == 1 ? Flux.binarycrossentropy : Flux.crossentropy
        training_history = Dict("train_loss" => Float64[], "val_loss" => Float64[], 
                               "train_acc" => Float64[], "val_acc" => Float64[])
        
        new(model, optimizer, loss_function, training_history)
    end
end

"""
Recurrent Neural Network for sequence data.
"""
struct RecurrentNetwork
    model::Chain
    optimizer
    loss_function
    training_history::Dict{String, Vector{Float64}}
    
    function RecurrentNetwork(input_size::Int, hidden_size::Int, output_size::Int, seq_length::Int;
                            cell_type::Symbol=:LSTM, num_layers::Int=2, dropout_rate::Float64=0.2)
        
        layers = []
        
        # Recurrent layers
        for i in 1:num_layers
            in_size = i == 1 ? input_size : hidden_size
            
            if cell_type == :LSTM
                push!(layers, LSTM(in_size, hidden_size))
            elseif cell_type == :GRU
                push!(layers, GRU(in_size, hidden_size))
            else
                push!(layers, RNN(in_size, hidden_size))
            end
            
            if dropout_rate > 0 && i < num_layers
                push!(layers, Dropout(dropout_rate))
            end
        end
        
        # Output layer
        push!(layers, Dense(hidden_size, output_size))
        
        model = Chain(layers...)
        optimizer = Adam(0.001)
        loss_function = Flux.mse
        training_history = Dict("train_loss" => Float64[], "val_loss" => Float64[])
        
        new(model, optimizer, loss_function, training_history)
    end
end

"""
Autoencoder for dimensionality reduction and feature learning.
"""
struct Autoencoder
    encoder::Chain
    decoder::Chain
    optimizer
    loss_function
    training_history::Dict{String, Vector{Float64}}
    
    function Autoencoder(input_dim::Int, encoding_dims::Vector{Int};
                        activation=relu, final_activation=sigmoid)
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims
            push!(encoder_layers, Dense(prev_dim, dim, activation))
            prev_dim = dim
        end
        encoder = Chain(encoder_layers...)
        
        # Decoder (reverse of encoder)
        decoder_layers = []
        for i in length(encoding_dims):-1:2
            push!(decoder_layers, Dense(encoding_dims[i], encoding_dims[i-1], activation))
        end
        push!(decoder_layers, Dense(encoding_dims[1], input_dim, final_activation))
        decoder = Chain(decoder_layers...)
        
        optimizer = Adam(0.001)
        loss_function = Flux.mse
        training_history = Dict("train_loss" => Float64[], "val_loss" => Float64[])
        
        new(encoder, decoder, optimizer, loss_function, training_history)
    end
end

# Forward pass for autoencoder
(ae::Autoencoder)(x) = ae.decoder(ae.encoder(x))

"""
Generic training function for all network types.
"""
function train_model!(network, train_data, val_data=nothing; 
                     epochs::Int=100, batch_size::Int=32, 
                     verbose::Bool=true, early_stopping::Bool=true,
                     patience::Int=10, min_delta::Float64=1e-4)
    
    # Setup for GPU if available
    device = gpu_available ? gpu : cpu
    
    if isa(network, NeuralNetwork)
        model = network.model |> device
    elseif isa(network, Union{ConvolutionalNetwork, RecurrentNetwork})
        model = network.model |> device
    elseif isa(network, Autoencoder)
        encoder = network.encoder |> device
        decoder = network.decoder |> device
        model = Chain(encoder, decoder) |> device
    end
    
    # Create data loaders
    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    val_loader = val_data !== nothing ? DataLoader(val_data, batchsize=batch_size) : nothing
    
    # Training state
    best_val_loss = Inf
    patience_counter = 0
    
    # Training loop
    progress = Progress(epochs, desc="Training epochs...")
    
    for epoch in 1:epochs
        # Training phase
        train_losses = Float64[]
        
        for (x_batch, y_batch) in train_loader
            x_batch, y_batch = x_batch |> device, y_batch |> device
            
            # Forward and backward pass
            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x_batch)
                network.loss_function(ŷ, y_batch)
            end
            
            # Update parameters
            Flux.update!(network.optimizer, model, grads[1])
            
            push!(train_losses, loss)
        end
        
        train_loss = mean(train_losses)
        push!(network.training_history["train_loss"], train_loss)
        
        # Validation phase
        if val_loader !== nothing
            val_losses = Float64[]
            
            for (x_batch, y_batch) in val_loader
                x_batch, y_batch = x_batch |> device, y_batch |> device
                ŷ = model(x_batch)
                loss = network.loss_function(ŷ, y_batch)
                push!(val_losses, loss)
            end
            
            val_loss = mean(val_losses)
            push!(network.training_history["val_loss"], val_loss)
            
            # Early stopping check
            if early_stopping
                if val_loss < best_val_loss - min_delta
                    best_val_loss = val_loss
                    patience_counter = 0
                else
                    patience_counter += 1
                    if patience_counter >= patience
                        println("\nEarly stopping at epoch $epoch")
                        break
                    end
                end
            end
            
            if verbose && epoch % 10 == 0
                println("Epoch $epoch: Train Loss = $(round(train_loss, digits=4)), Val Loss = $(round(val_loss, digits=4))")
            end
        else
            if verbose && epoch % 10 == 0
                println("Epoch $epoch: Train Loss = $(round(train_loss, digits=4))")
            end
        end
        
        next!(progress)
    end
    
    # Move model back to CPU for compatibility
    if isa(network, NeuralNetwork)
        network.model = network.model |> cpu
    elseif isa(network, Union{ConvolutionalNetwork, RecurrentNetwork})
        network.model = network.model |> cpu
    elseif isa(network, Autoencoder)
        network.encoder = network.encoder |> cpu
        network.decoder = network.decoder |> cpu
    end
    
    return network
end

"""
Evaluate model performance on test data.
"""
function evaluate_model(network, test_data; batch_size::Int=32)
    device = gpu_available ? gpu : cpu
    
    if isa(network, NeuralNetwork)
        model = network.model |> device
    elseif isa(network, Union{ConvolutionalNetwork, RecurrentNetwork})
        model = network.model |> device
    elseif isa(network, Autoencoder)
        encoder = network.encoder |> device
        decoder = network.decoder |> device
        model = Chain(encoder, decoder) |> device
    end
    
    test_loader = DataLoader(test_data, batchsize=batch_size)
    
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []
    
    for (x_batch, y_batch) in test_loader
        x_batch, y_batch = x_batch |> device, y_batch |> device
        
        ŷ = model(x_batch)
        loss = network.loss_function(ŷ, y_batch)
        
        batch_size_actual = size(x_batch)[end]
        total_loss += loss * batch_size_actual
        total_samples += batch_size_actual
        
        # Store predictions and targets for additional metrics
        push!(predictions, ŷ |> cpu)
        push!(targets, y_batch |> cpu)
    end
    
    avg_loss = total_loss / total_samples
    
    # Concatenate all predictions and targets
    all_predictions = vcat(predictions...)
    all_targets = vcat(targets...)
    
    # Calculate additional metrics
    metrics = Dict("loss" => avg_loss)
    
    # For regression tasks
    if size(all_predictions, 1) == 1 || length(size(all_predictions)) == 1
        mse = mean((all_predictions .- all_targets) .^ 2)
        mae = mean(abs.(all_predictions .- all_targets))
        
        # R-squared
        ss_res = sum((all_targets .- all_predictions) .^ 2)
        ss_tot = sum((all_targets .- mean(all_targets)) .^ 2)
        r2 = 1 - ss_res / ss_tot
        
        metrics["mse"] = mse
        metrics["mae"] = mae
        metrics["r2"] = r2
    else
        # For classification tasks
        predicted_classes = Flux.onecold(all_predictions)
        true_classes = Flux.onecold(all_targets)
        accuracy = mean(predicted_classes .== true_classes)
        metrics["accuracy"] = accuracy
    end
    
    return metrics
end

"""
Plot training history for visualization.
"""
function plot_training_history(network; save_path::String="")
    history = network.training_history
    
    if haskey(history, "val_loss")
        # Plot training and validation loss
        p1 = plot(history["train_loss"], label="Training Loss", 
                 title="Model Loss", xlabel="Epoch", ylabel="Loss")
        plot!(p1, history["val_loss"], label="Validation Loss")
        
        if haskey(history, "train_acc")
            # Plot training and validation accuracy
            p2 = plot(history["train_acc"], label="Training Accuracy",
                     title="Model Accuracy", xlabel="Epoch", ylabel="Accuracy")
            plot!(p2, history["val_acc"], label="Validation Accuracy")
            
            final_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
        else
            final_plot = p1
        end
    else
        # Plot only training loss
        final_plot = plot(history["train_loss"], label="Training Loss",
                         title="Training Loss", xlabel="Epoch", ylabel="Loss")
    end
    
    if !isempty(save_path)
        savefig(final_plot, save_path)
        println("Training history plot saved to: $save_path")
    end
    
    return final_plot
end

"""
Create optimizers with different configurations.
"""
function create_optimizer(optimizer_type::Symbol; learning_rate::Float64=0.001, kwargs...)
    if optimizer_type == :adam
        return Adam(learning_rate)
    elseif optimizer_type == :sgd
        momentum = get(kwargs, :momentum, 0.9)
        return Momentum(learning_rate, momentum)
    elseif optimizer_type == :rmsprop
        return RMSProp(learning_rate)
    elseif optimizer_type == :adagrad
        return AdaGrad(learning_rate)
    else
        throw(ArgumentError("Unknown optimizer type: $optimizer_type"))
    end
end

end # module DeepLearning

# ═══════════════════════════════════════════════════════════════════════════════
#                           6. PARALLEL AND DISTRIBUTED COMPUTING
# ═══════════════════════════════════════════════════════════════════════════════

"""
Parallel computing utilities for scaling ML algorithms.
"""
module ParallelComputing

using Distributed, SharedArrays, ThreadsX
using Statistics, Random
using ProgressMeter

export parallel_cross_validation, distributed_grid_search, parallel_bootstrap
export threaded_matrix_operations, distributed_kmeans

"""
Parallel cross-validation with distributed computing.
"""
function parallel_cross_validation(model_func, X, y, cv_folds::Int=5; 
                                 metrics=[:accuracy], random_state::Int=42)
    
    Random.seed!(random_state)
    n = size(X, 1)
    fold_size = n ÷ cv_folds
    indices = shuffle(1:n)
    
    # Create fold indices
    fold_ranges = [(i-1)*fold_size+1 : (i==cv_folds ? n : i*fold_size) for i in 1:cv_folds]
    
    # Distribute data to workers
    @everywhere X_shared = $X
    @everywhere y_shared = $y
    @everywhere model_func_shared = $model_func
    @everywhere indices_shared = $indices
    
    # Parallel execution of CV folds
    results = @distributed (append!) for fold in 1:cv_folds
        # Get test indices for this fold
        test_indices = indices_shared[fold_ranges[fold]]
        train_indices = setdiff(indices_shared, test_indices)
        
        # Split data
        X_train = X_shared[train_indices, :]
        X_test = X_shared[test_indices, :]
        y_train = y_shared[train_indices]
        y_test = y_shared[test_indices]
        
        # Train model
        model = model_func_shared()
        fit!(model, X_train, y_train)
        
        # Make predictions and calculate metrics
        y_pred = predict(model, X_test)
        
        fold_results = Dict()
        for metric in metrics
            if metric == :accuracy
                score = mean(y_pred .== y_test)
            elseif metric == :mse
                score = mean((y_pred .- y_test) .^ 2)
            elseif metric == :mae
                score = mean(abs.(y_pred .- y_test))
            end
            fold_results[metric] = score
        end
        
        [fold_results]
    end
    
    # Aggregate results
    aggregated = Dict()
    for metric in metrics
        scores = [result[metric] for result in results]
        aggregated[metric] = (mean=mean(scores), std=std(scores), scores=scores)
    end
    
    return aggregated
end

"""
Distributed grid search for hyperparameter optimization.
"""
function distributed_grid_search(model_func, param_grid::Dict, X, y; 
                                cv_folds::Int=3, scoring::Symbol=:accuracy,
                                verbose::Bool=true)
    
    # Generate parameter combinations
    param_names = collect(keys(param_grid))
    param_values = collect(values(param_grid))
    param_combinations = vec(collect(Iterators.product(param_values...)))
    
    n_combinations = length(param_combinations)
    println("Grid search with $n_combinations parameter combinations")
    
    # Distribute data and functions
    @everywhere X_grid = $X
    @everywhere y_grid = $y
    @everywhere model_func_grid = $model_func
    @everywhere param_names_grid = $param_names
    
    # Parallel grid search
    results = @distributed (append!) for i in 1:n_combinations
        params = param_combinations[i]
        param_dict = Dict(zip(param_names_grid, params))
        
        # Create model with current parameters
        model = model_func_grid(; param_dict...)
        
        # Perform cross-validation
        cv_results = parallel_cross_validation(() -> model_func_grid(; param_dict...),
                                             X_grid, y_grid, cv_folds, 
                                             metrics=[scoring])
        
        score = cv_results[scoring].mean
        
        [(params=param_dict, score=score, cv_std=cv_results[scoring].std)]
    end
    
    # Find best parameters
    best_idx = argmax([r.score for r in results])
    best_result = results[best_idx]
    
    if verbose
        println("Best parameters: $(best_result.params)")
        println("Best cross-validation score: $(round(best_result.score, digits=4)) ± $(round(best_result.cv_std, digits=4))")
    end
    
    return (best_params=best_result.params, best_score=best_result.score, all_results=results)
end

"""
Parallel bootstrap sampling for statistical inference.
"""
function parallel_bootstrap(statistic_func, data, n_bootstrap::Int=1000; 
                          confidence_level::Float64=0.95, random_state::Int=42)
    
    Random.seed!(random_state)
    n_samples = size(data, 1)
    
    # Generate bootstrap sample indices
    bootstrap_indices = [rand(1:n_samples, n_samples) for _ in 1:n_bootstrap]
    
    @everywhere data_bootstrap = $data
    @everywhere statistic_func_bootstrap = $statistic_func
    
    # Parallel bootstrap computation
    bootstrap_stats = @distributed (append!) for indices in bootstrap_indices
        bootstrap_sample = data_bootstrap[indices, :]
        stat = statistic_func_bootstrap(bootstrap_sample)
        [stat]
    end
    
    # Calculate confidence intervals
    α = 1 - confidence_level
    lower_percentile = (α/2) * 100
    upper_percentile = (1 - α/2) * 100
    
    ci_lower = percentile(bootstrap_stats, lower_percentile)
    ci_upper = percentile(bootstrap_stats, upper_percentile)
    
    return (
        statistic=mean(bootstrap_stats),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        bootstrap_distribution=bootstrap_stats
    )
end

"""
Threaded matrix operations for improved performance.
"""
function threaded_matrix_multiply(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    m, k = size(A)
    k2, n = size(B)
    @assert k == k2 "Matrix dimensions must match"
    
    C = zeros(T, m, n)
    
    ThreadsX.foreach(1:m) do i
        for j in 1:n
            for l in 1:k
                C[i, j] += A[i, l] * B[l, j]
            end
        end
    end
    
    return C
end

"""
Threaded element-wise operations.
"""
function threaded_map(f, arrays...)
    result = similar(arrays[1])
    ThreadsX.map!(f, result, arrays...)
    return result
end

"""
Distributed K-means clustering for large datasets.
"""
function distributed_kmeans(X::AbstractMatrix{T}, k::Int; 
                          max_iterations::Int=300, tolerance::T=T(1e-4),
                          n_init::Int=10) where T
    
    n, d = size(X)
    best_centroids = nothing
    best_inertia = Inf
    
    # Try multiple initializations in parallel
    @everywhere X_kmeans = $X
    @everywhere k_kmeans = $k
    @everywhere max_iterations_kmeans = $max_iterations
    @everywhere tolerance_kmeans = $tolerance
    
    results = @distributed (append!) for init in 1:n_init
        # K-means++ initialization
        centroids = kmeans_plus_plus_init(X_kmeans, k_kmeans)
        
        # Lloyd's algorithm
        for iter in 1:max_iterations_kmeans
            old_centroids = copy(centroids)
            
            # Assign points to nearest centroids (parallel)
            labels = ThreadsX.map(1:size(X_kmeans, 1)) do i
                min_dist = Inf
                best_cluster = 1
                
                for j in 1:k_kmeans
                    dist = sum((X_kmeans[i, :] .- centroids[j, :]) .^ 2)
                    if dist < min_dist
                        min_dist = dist
                        best_cluster = j
                    end
                end
                
                best_cluster
            end
            
            # Update centroids
            for j in 1:k_kmeans
                cluster_points = X_kmeans[labels .== j, :]
                if size(cluster_points, 1) > 0
                    centroids[j, :] = mean(cluster_points, dims=1)[:]
                end
            end
            
            # Check convergence
            if norm(centroids - old_centroids) < tolerance_kmeans
                break
            end
        end
        
        # Calculate inertia
        inertia = 0.0
        for i in 1:size(X_kmeans, 1)
            min_dist = Inf
            for j in 1:k_kmeans
                dist = sum((X_kmeans[i, :] .- centroids[j, :]) .^ 2)
                min_dist = min(min_dist, dist)
            end
            inertia += min_dist
        end
        
        [(centroids=centroids, inertia=inertia)]
    end
    
    # Select best result
    best_idx = argmin([r.inertia for r in results])
    best_result = results[best_idx]
    
    return (centroids=best_result.centroids, inertia=best_result.inertia)
end

end # module ParallelComputing

# ═══════════════════════════════════════════════════════════════════════════════
#                           7. TIME SERIES ANALYSIS AND FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

"""
Advanced time series analysis with Julia's performance advantages.
"""
module TimeSeriesAnalysis

using Statistics, LinearAlgebra, Random
using DSP, Distributions
using Plots
using ProgressMeter

export TimeSeriesModel, ARIMA, ExponentialSmoothing, StateSpaceModel
export seasonal_decompose, detect_changepoints, forecast_accuracy

"""
Time series decomposition using STL (Seasonal and Trend decomposition using Loess).
"""
function seasonal_decompose(y::Vector{T}, period::Int; 
                          trend_window::Union{Int,Nothing}=nothing,
                          seasonal_window::Union{Int,Nothing}=nothing) where T<:Real
    
    n = length(y)
    
    # Default parameters
    if trend_window === nothing
        trend_window = ceil(Int, 1.5 * period / (1 - 1