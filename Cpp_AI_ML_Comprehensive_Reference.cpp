// C++ AI/MACHINE LEARNING - Comprehensive Reference - by Richard Rembert
// C++ for performance-critical ML/AI libraries with focus on optimization,
// SIMD, GPU computing, and high-performance numerical computing

// ═══════════════════════════════════════════════════════════════════════════════
//                           1. SETUP AND PROJECT STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════════

/*
C++ AI/ML PROJECT SETUP:

1. Build System (CMakeLists.txt):
cmake_minimum_required(VERSION 3.20)
project(MLFramework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(BLAS REQUIRED blas)
pkg_check_modules(LAPACK REQUIRED lapack)

# Compiler optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -DNDEBUG")
endif()

# SIMD support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

2. Directory Structure:
MLFramework/
├── include/
│   ├── ml/
│   │   ├── core/
│   │   │   ├── tensor.hpp
│   │   │   ├── matrix.hpp
│   │   │   └── memory.hpp
│   │   ├── algorithms/
│   │   │   ├── linear.hpp
│   │   │   ├── neural.hpp
│   │   │   └── clustering.hpp
│   │   ├── optimizers/
│   │   │   ├── sgd.hpp
│   │   │   ├── adam.hpp
│   │   │   └── optimizer_base.hpp
│   │   ├── utils/
│   │   │   ├── simd.hpp
│   │   │   ├── parallel.hpp
│   │   │   └── profiler.hpp
│   │   └── gpu/
│   │       ├── cuda_ops.cuh
│   │       └── cuda_kernels.cuh
│   └── external/
├── src/
│   ├── core/
│   ├── algorithms/
│   ├── optimizers/
│   ├── utils/
│   └── gpu/
├── tests/
├── benchmarks/
├── examples/
└── docs/

3. Essential Dependencies:
- Eigen3 (Linear algebra)
- OpenBLAS/Intel MKL (Optimized BLAS)
- CUDA/cuBLAS (GPU computing)
- OpenMP (CPU parallelization)
- Google Benchmark (Performance testing)
- Google Test (Unit testing)
*/

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>
#include <execution>
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <cassert>
#include <type_traits>
#include <concepts>

// ═══════════════════════════════════════════════════════════════════════════════
//                           2. CORE DATA STRUCTURES AND MEMORY MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

namespace ml {

// Modern C++20 concepts for type safety
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<typename T>
concept FloatingPoint = std::floating_point<T>;

// High-performance memory allocator with alignment
template<typename T, size_t Alignment = 32>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;
    
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        
        void* ptr = nullptr;
        size_t size = n * sizeof(T);
        
        #ifdef _WIN32
            ptr = _aligned_malloc(size, Alignment);
        #else
            if (posix_memalign(&ptr, Alignment, size) != 0) {
                ptr = nullptr;
            }
        #endif
        
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        if (p) {
            #ifdef _WIN32
                _aligned_free(p);
            #else
                free(p);
            #endif
        }
    }

    template<typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }
};

// High-performance tensor class with SIMD optimization
template<Arithmetic T, size_t Dims = 2>
class Tensor {
private:
    std::vector<T, AlignedAllocator<T, 32>> data_;
    std::array<size_t, Dims> shape_;
    std::array<size_t, Dims> strides_;
    size_t size_;

    void compute_strides() {
        if constexpr (Dims > 0) {
            strides_[Dims - 1] = 1;
            for (int i = Dims - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        }
    }

public:
    using value_type = T;
    using allocator_type = AlignedAllocator<T, 32>;

    // Constructor with shape
    template<typename... Args>
    explicit Tensor(Args... dims) requires (sizeof...(Args) == Dims) {
        static_assert(sizeof...(Args) == Dims, "Number of dimensions must match template parameter");
        shape_ = {static_cast<size_t>(dims)...};
        compute_strides();
        size_ = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
        data_.resize(size_);
    }

    // Copy constructor
    Tensor(const Tensor& other) = default;
    
    // Move constructor
    Tensor(Tensor&& other) noexcept = default;
    
    // Assignment operators
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    // Element access
    template<typename... Indices>
    T& operator()(Indices... indices) requires (sizeof...(Indices) == Dims) {
        static_assert(sizeof...(Indices) == Dims, "Number of indices must match tensor dimensions");
        std::array<size_t, Dims> idx_array = {static_cast<size_t>(indices)...};
        size_t linear_idx = 0;
        for (size_t i = 0; i < Dims; ++i) {
            assert(idx_array[i] < shape_[i]);
            linear_idx += idx_array[i] * strides_[i];
        }
        return data_[linear_idx];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const requires (sizeof...(Indices) == Dims) {
        static_assert(sizeof...(Indices) == Dims, "Number of indices must match tensor dimensions");
        std::array<size_t, Dims> idx_array = {static_cast<size_t>(indices)...};
        size_t linear_idx = 0;
        for (size_t i = 0; i < Dims; ++i) {
            assert(idx_array[i] < shape_[i]);
            linear_idx += idx_array[i] * strides_[i];
        }
        return data_[linear_idx];
    }

    // Raw data access
    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    // Shape and size
    const std::array<size_t, Dims>& shape() const noexcept { return shape_; }
    size_t size() const noexcept { return size_; }
    constexpr size_t dimensions() const noexcept { return Dims; }

    // Fill with value
    void fill(T value) {
        std::fill(std::execution::par_unseq, data_.begin(), data_.end(), value);
    }

    // Random initialization
    void random_normal(T mean = T(0), T stddev = T(1)) {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        std::normal_distribution<T> dist(mean, stddev);
        
        std::generate(std::execution::par_unseq, data_.begin(), data_.end(),
                     [&]() { return dist(gen); });
    }

    void random_uniform(T min_val = T(0), T max_val = T(1)) {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(min_val, max_val);
        
        std::generate(std::execution::par_unseq, data_.begin(), data_.end(),
                     [&]() { return dist(gen); });
    }

    // Mathematical operations with SIMD
    Tensor& operator+=(const Tensor& other) {
        assert(shape_ == other.shape_);
        
        if constexpr (std::is_same_v<T, float>) {
            simd_add_float(data_.data(), other.data_.data(), data_.data(), size_);
        } else {
            std::transform(std::execution::par_unseq, 
                          data_.begin(), data_.end(), 
                          other.data_.begin(), data_.begin(),
                          std::plus<T>());
        }
        return *this;
    }

    Tensor& operator*=(T scalar) {
        if constexpr (std::is_same_v<T, float>) {
            simd_scale_float(data_.data(), scalar, data_.data(), size_);
        } else {
            std::transform(std::execution::par_unseq,
                          data_.begin(), data_.end(), data_.begin(),
                          [scalar](T x) { return x * scalar; });
        }
        return *this;
    }

private:
    // SIMD operations for float
    void simd_add_float(const float* a, const float* b, float* result, size_t n) {
        const size_t simd_width = 8; // AVX2 processes 8 floats at once
        const size_t simd_end = (n / simd_width) * simd_width;
        
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 va = _mm256_load_ps(&a[i]);
            __m256 vb = _mm256_load_ps(&b[i]);
            __m256 vresult = _mm256_add_ps(va, vb);
            _mm256_store_ps(&result[i], vresult);
        }
        
        // Handle remaining elements
        for (size_t i = simd_end; i < n; ++i) {
            result[i] = a[i] + b[i];
        }
    }

    void simd_scale_float(const float* input, float scalar, float* output, size_t n) {
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        
        __m256 vscalar = _mm256_set1_ps(scalar);
        
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 vinput = _mm256_load_ps(&input[i]);
            __m256 vresult = _mm256_mul_ps(vinput, vscalar);
            _mm256_store_ps(&output[i], vresult);
        }
        
        // Handle remaining elements
        for (size_t i = simd_end; i < n; ++i) {
            output[i] = input[i] * scalar;
        }
    }
};

// Specialized Matrix class (2D Tensor) with optimized operations
template<Arithmetic T>
class Matrix : public Tensor<T, 2> {
public:
    using BaseType = Tensor<T, 2>;
    using value_type = T;

    Matrix(size_t rows, size_t cols) : BaseType(rows, cols) {}

    size_t rows() const noexcept { return this->shape()[0]; }
    size_t cols() const noexcept { return this->shape()[1]; }

    // Matrix multiplication with BLAS integration
    Matrix multiply(const Matrix& other) const {
        assert(cols() == other.rows());
        
        Matrix result(rows(), other.cols());
        
        if constexpr (std::is_same_v<T, float>) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       static_cast<int>(rows()), static_cast<int>(other.cols()), 
                       static_cast<int>(cols()),
                       1.0f, this->data(), static_cast<int>(cols()),
                       other.data(), static_cast<int>(other.cols()),
                       0.0f, result.data(), static_cast<int>(result.cols()));
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       static_cast<int>(rows()), static_cast<int>(other.cols()), 
                       static_cast<int>(cols()),
                       1.0, this->data(), static_cast<int>(cols()),
                       other.data(), static_cast<int>(other.cols()),
                       0.0, result.data(), static_cast<int>(result.cols()));
        } else {
            // Fallback for other types
            parallel_matrix_multiply(*this, other, result);
        }
        
        return result;
    }

    // Transpose operation
    Matrix transpose() const {
        Matrix result(cols(), rows());
        
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        
        return result;
    }

    // Element-wise operations
    Matrix hadamard(const Matrix& other) const {
        assert(this->shape() == other.shape());
        Matrix result = *this;
        
        #pragma omp parallel for
        for (size_t i = 0; i < this->size(); ++i) {
            result.data()[i] *= other.data()[i];
        }
        
        return result;
    }

private:
    // Fallback parallel matrix multiplication
    void parallel_matrix_multiply(const Matrix& a, const Matrix& b, Matrix& c) const {
        const size_t M = a.rows();
        const size_t N = b.cols();
        const size_t K = a.cols();
        
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                T sum = T(0);
                for (size_t k = 0; k < K; ++k) {
                    sum += a(i, k) * b(k, j);
                }
                c(i, j) = sum;
            }
        }
    }
};

} // namespace ml

// ═══════════════════════════════════════════════════════════════════════════════
//                           3. SIMD OPTIMIZED OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

namespace ml::simd {

// SIMD utility functions for different instruction sets
class SIMDOperations {
public:
    // Check CPU capabilities at runtime
    static bool has_avx2() {
        #ifdef __AVX2__
            return true;
        #else
            return false;
        #endif
    }

    static bool has_fma() {
        #ifdef __FMA__
            return true;
        #else
            return false;
        #endif
    }

    // Vectorized dot product for float arrays
    static float dot_product_avx2(const float* a, const float* b, size_t n) {
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        
        __m256 sum_vec = _mm256_setzero_ps();
        
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            
            #ifdef __FMA__
                sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
            #else
                __m256 prod = _mm256_mul_ps(va, vb);
                sum_vec = _mm256_add_ps(sum_vec, prod);
            #endif
        }
        
        // Horizontal sum of the vector
        __m128 high = _mm256_extractf128_ps(sum_vec, 1);
        __m128 low = _mm256_castps256_ps128(sum_vec);
        __m128 sum128 = _mm_add_ps(high, low);
        
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        
        float result = _mm_cvtss_f32(sum128);
        
        // Handle remaining elements
        for (size_t i = simd_end; i < n; ++i) {
            result += a[i] * b[i];
        }
        
        return result;
    }

    // Vectorized matrix-vector multiplication
    static void matvec_avx2(const float* matrix, const float* vector, 
                           float* result, size_t rows, size_t cols) {
        #pragma omp parallel for
        for (size_t i = 0; i < rows; ++i) {
            result[i] = dot_product_avx2(&matrix[i * cols], vector, cols);
        }
    }

    // Vectorized ReLU activation
    static void relu_avx2(const float* input, float* output, size_t n) {
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        
        __m256 zero = _mm256_setzero_ps();
        
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 x = _mm256_loadu_ps(&input[i]);
            __m256 result = _mm256_max_ps(x, zero);
            _mm256_storeu_ps(&output[i], result);
        }
        
        // Handle remaining elements
        for (size_t i = simd_end; i < n; ++i) {
            output[i] = std::max(input[i], 0.0f);
        }
    }

    // Vectorized sigmoid activation
    static void sigmoid_avx2(const float* input, float* output, size_t n) {
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        
        __m256 one = _mm256_set1_ps(1.0f);
        
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 x = _mm256_loadu_ps(&input[i]);
            
            // Approximate exp using polynomial (for demonstration)
            // In production, use more accurate methods or libraries
            __m256 exp_x = fast_exp_avx2(x);
            __m256 result = _mm256_div_ps(one, _mm256_add_ps(one, exp_x));
            
            _mm256_storeu_ps(&output[i], result);
        }
        
        // Handle remaining elements
        for (size_t i = simd_end; i < n; ++i) {
            output[i] = 1.0f / (1.0f + std::exp(-input[i]));
        }
    }

private:
    // Fast exponential approximation using AVX2
    static __m256 fast_exp_avx2(__m256 x) {
        // Polynomial approximation for exp(-x)
        // This is a simplified version - use proper implementations in production
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 two = _mm256_set1_ps(2.0f);
        __m256 half = _mm256_set1_ps(0.5f);
        
        // Negate x for exp(-x)
        x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        
        // Taylor series approximation: 1 + x + x²/2 + x³/6 + ...
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        
        __m256 term1 = x;
        __m256 term2 = _mm256_mul_ps(x2, half);
        __m256 term3 = _mm256_mul_ps(x3, _mm256_set1_ps(1.0f/6.0f));
        
        return _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(one, term1), term2), term3);
    }
};

} // namespace ml::simd

// ═══════════════════════════════════════════════════════════════════════════════
//                           4. NEURAL NETWORK IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════════

namespace ml::neural {

// Activation functions with vectorized implementations
class ActivationFunctions {
public:
    template<FloatingPoint T>
    static void relu(const T* input, T* output, size_t size) {
        if constexpr (std::is_same_v<T, float>) {
            simd::SIMDOperations::relu_avx2(input, output, size);
        } else {
            #pragma omp parallel for
            for (size_t i = 0; i < size; ++i) {
                output[i] = std::max(input[i], T(0));
            }
        }
    }

    template<FloatingPoint T>
    static void relu_derivative(const T* input, T* output, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            output[i] = input[i] > T(0) ? T(1) : T(0);
        }
    }

    template<FloatingPoint T>
    static void sigmoid(const T* input, T* output, size_t size) {
        if constexpr (std::is_same_v<T, float>) {
            simd::SIMDOperations::sigmoid_avx2(input, output, size);
        } else {
            #pragma omp parallel for
            for (size_t i = 0; i < size; ++i) {
                output[i] = T(1) / (T(1) + std::exp(-input[i]));
            }
        }
    }

    template<FloatingPoint T>
    static void tanh_activation(const T* input, T* output, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            output[i] = std::tanh(input[i]);
        }
    }

    template<FloatingPoint T>
    static void softmax(const T* input, T* output, size_t size) {
        // Find maximum for numerical stability
        T max_val = *std::max_element(input, input + size);
        
        // Compute exp(x - max) and sum
        T sum = T(0);
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < size; ++i) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }
        
        // Normalize
        T inv_sum = T(1) / sum;
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            output[i] *= inv_sum;
        }
    }
};

// Dense layer implementation with optimized forward/backward passes
template<FloatingPoint T>
class DenseLayer {
private:
    Matrix<T> weights_;
    Tensor<T, 1> biases_;
    size_t input_size_;
    size_t output_size_;
    
    // Cached values for backward pass
    mutable Matrix<T> last_input_;
    mutable Matrix<T> last_output_;
    
public:
    DenseLayer(size_t input_size, size_t output_size) 
        : weights_(output_size, input_size)
        , biases_(output_size)
        , input_size_(input_size)
        , output_size_(output_size)
        , last_input_(1, input_size)
        , last_output_(1, output_size) {
        
        // Xavier initialization
        T std_dev = std::sqrt(T(2) / T(input_size + output_size));
        weights_.random_normal(T(0), std_dev);
        biases_.fill(T(0));
    }

    // Forward pass
    Matrix<T> forward(const Matrix<T>& input) const {
        assert(input.cols() == input_size_);
        
        last_input_ = input;  // Cache for backward pass
        
        // Compute: output = input * weights^T + bias
        Matrix<T> weights_t = weights_.transpose();
        Matrix<T> output = input.multiply(weights_t);
        
        // Add bias to each row
        #pragma omp parallel for
        for (size_t i = 0; i < output.rows(); ++i) {
            for (size_t j = 0; j < output.cols(); ++j) {
                output(i, j) += biases_(j);
            }
        }
        
        last_output_ = output;
        return output;
    }

    // Backward pass
    std::pair<Matrix<T>, Matrix<T>> backward(const Matrix<T>& grad_output) const {
        assert(grad_output.cols() == output_size_);
        
        // Gradient w.r.t. weights: grad_weights = grad_output^T * input
        Matrix<T> grad_output_t = grad_output.transpose();
        Matrix<T> grad_weights = grad_output_t.multiply(last_input_);
        
        // Gradient w.r.t. input: grad_input = grad_output * weights
        Matrix<T> grad_input = grad_output.multiply(weights_);
        
        return {grad_weights, grad_input};
    }

    // Parameter accessors
    Matrix<T>& weights() { return weights_; }
    const Matrix<T>& weights() const { return weights_; }
    Tensor<T, 1>& biases() { return biases_; }
    const Tensor<T, 1>& biases() const { return biases_; }
    
    size_t input_size() const { return input_size_; }
    size_t output_size() const { return output_size_; }
};

// Multi-layer perceptron with configurable architecture
template<FloatingPoint T>
class MLP {
private:
    std::vector<std::unique_ptr<DenseLayer<T>>> layers_;
    std::vector<std::function<void(const T*, T*, size_t)>> activations_;
    std::vector<Matrix<T>> cached_outputs_;
    
public:
    MLP() = default;
    
    void add_layer(size_t input_size, size_t output_size, 
                   std::function<void(const T*, T*, size_t)> activation = nullptr) {
        layers_.emplace_back(std::make_unique<DenseLayer<T>>(input_size, output_size));
        activations_.push_back(activation);
    }

    Matrix<T> forward(const Matrix<T>& input) {
        cached_outputs_.clear();
        cached_outputs_.reserve(layers_.size());
        
        Matrix<T> current_output = input;
        
        for (size_t i = 0; i < layers_.size(); ++i) {
            current_output = layers_[i]->forward(current_output);
            
            // Apply activation function if specified
            if (activations_[i]) {
                Matrix<T> activated_output = current_output;
                activations_[i](current_output.data(), activated_output.data(), 
                              current_output.size());
                current_output = std::move(activated_output);
            }
            
            cached_outputs_.push_back(current_output);
        }
        
        return current_output;
    }

    // Backward pass with gradient computation
    void backward(const Matrix<T>& loss_gradient) {
        Matrix