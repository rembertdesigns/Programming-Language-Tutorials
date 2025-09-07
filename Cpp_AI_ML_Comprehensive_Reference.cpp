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
            Matrix<T> current_gradient = loss_gradient;
            
            // Backpropagate through layers in reverse order
            for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
                auto [weight_grad, input_grad] = layers_[i]->backward(current_gradient);
                
                // Update current gradient for next layer
                current_gradient = input_grad;
                
                // Store gradients for optimizer (implementation depends on optimizer design)
                // This would typically be handled by an optimizer class
            }
        }
    
        // Get layers for parameter access
        const std::vector<std::unique_ptr<DenseLayer<T>>>& layers() const { return layers_; }
        std::vector<std::unique_ptr<DenseLayer<T>>>& layers() { return layers_; }
    };
    
    } // namespace ml::neural
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                           5. OPTIMIZATION ALGORITHMS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    namespace ml::optimizers {
    
    // Base optimizer interface
    template<FloatingPoint T>
    class OptimizerBase {
    public:
        virtual ~OptimizerBase() = default;
        virtual void update(Matrix<T>& weights, const Matrix<T>& gradients) = 0;
        virtual void update(Tensor<T, 1>& biases, const Tensor<T, 1>& gradients) = 0;
        virtual void reset() {}
    };
    
    // Stochastic Gradient Descent optimizer
    template<FloatingPoint T>
    class SGDOptimizer : public OptimizerBase<T> {
    private:
        T learning_rate_;
        T momentum_;
        T weight_decay_;
        
        // Momentum buffers
        std::unordered_map<void*, Matrix<T>> weight_momentum_;
        std::unordered_map<void*, Tensor<T, 1>> bias_momentum_;
        
    public:
        SGDOptimizer(T learning_rate = T(0.01), T momentum = T(0.9), T weight_decay = T(0))
            : learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {}
    
        void update(Matrix<T>& weights, const Matrix<T>& gradients) override {
            void* weights_ptr = static_cast<void*>(&weights);
            
            // Initialize momentum buffer if needed
            if (weight_momentum_.find(weights_ptr) == weight_momentum_.end()) {
                weight_momentum_[weights_ptr] = Matrix<T>(weights.rows(), weights.cols());
                weight_momentum_[weights_ptr].fill(T(0));
            }
            
            Matrix<T>& momentum_buffer = weight_momentum_[weights_ptr];
            
            // Apply weight decay if specified
            Matrix<T> effective_gradients = gradients;
            if (weight_decay_ > T(0)) {
                #pragma omp parallel for
                for (size_t i = 0; i < weights.size(); ++i) {
                    effective_gradients.data()[i] += weight_decay_ * weights.data()[i];
                }
            }
            
            // Update momentum buffer: v = momentum * v + learning_rate * grad
            #pragma omp parallel for
            for (size_t i = 0; i < momentum_buffer.size(); ++i) {
                momentum_buffer.data()[i] = momentum_ * momentum_buffer.data()[i] + 
                                           learning_rate_ * effective_gradients.data()[i];
            }
            
            // Update weights: w = w - v
            #pragma omp parallel for
            for (size_t i = 0; i < weights.size(); ++i) {
                weights.data()[i] -= momentum_buffer.data()[i];
            }
        }
    
        void update(Tensor<T, 1>& biases, const Tensor<T, 1>& gradients) override {
            void* biases_ptr = static_cast<void*>(&biases);
            
            // Initialize momentum buffer if needed
            if (bias_momentum_.find(biases_ptr) == bias_momentum_.end()) {
                bias_momentum_[biases_ptr] = Tensor<T, 1>(biases.size());
                bias_momentum_[biases_ptr].fill(T(0));
            }
            
            Tensor<T, 1>& momentum_buffer = bias_momentum_[biases_ptr];
            
            // Update momentum and biases
            #pragma omp parallel for
            for (size_t i = 0; i < biases.size(); ++i) {
                momentum_buffer.data()[i] = momentum_ * momentum_buffer.data()[i] + 
                                           learning_rate_ * gradients.data()[i];
                biases.data()[i] -= momentum_buffer.data()[i];
            }
        }
    
        void reset() override {
            weight_momentum_.clear();
            bias_momentum_.clear();
        }
    };
    
    // Adam optimizer with bias correction
    template<FloatingPoint T>
    class AdamOptimizer : public OptimizerBase<T> {
    private:
        T learning_rate_;
        T beta1_;
        T beta2_;
        T epsilon_;
        T weight_decay_;
        size_t t_; // time step
        
        // Adam state buffers
        std::unordered_map<void*, Matrix<T>> weight_m_;  // first moment
        std::unordered_map<void*, Matrix<T>> weight_v_;  // second moment
        std::unordered_map<void*, Tensor<T, 1>> bias_m_;
        std::unordered_map<void*, Tensor<T, 1>> bias_v_;
        
    public:
        AdamOptimizer(T learning_rate = T(0.001), T beta1 = T(0.9), T beta2 = T(0.999), 
                      T epsilon = T(1e-8), T weight_decay = T(0))
            : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), 
              epsilon_(epsilon), weight_decay_(weight_decay), t_(0) {}
    
        void update(Matrix<T>& weights, const Matrix<T>& gradients) override {
            ++t_;
            void* weights_ptr = static_cast<void*>(&weights);
            
            // Initialize buffers if needed
            if (weight_m_.find(weights_ptr) == weight_m_.end()) {
                weight_m_[weights_ptr] = Matrix<T>(weights.rows(), weights.cols());
                weight_v_[weights_ptr] = Matrix<T>(weights.rows(), weights.cols());
                weight_m_[weights_ptr].fill(T(0));
                weight_v_[weights_ptr].fill(T(0));
            }
            
            Matrix<T>& m = weight_m_[weights_ptr];
            Matrix<T>& v = weight_v_[weights_ptr];
            
            // Apply weight decay if specified
            Matrix<T> effective_gradients = gradients;
            if (weight_decay_ > T(0)) {
                #pragma omp parallel for
                for (size_t i = 0; i < weights.size(); ++i) {
                    effective_gradients.data()[i] += weight_decay_ * weights.data()[i];
                }
            }
            
            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
            // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * grad²
            #pragma omp parallel for
            for (size_t i = 0; i < weights.size(); ++i) {
                T grad = effective_gradients.data()[i];
                m.data()[i] = beta1_ * m.data()[i] + (T(1) - beta1_) * grad;
                v.data()[i] = beta2_ * v.data()[i] + (T(1) - beta2_) * grad * grad;
            }
            
            // Bias correction
            T bias_correction1 = T(1) - std::pow(beta1_, static_cast<T>(t_));
            T bias_correction2 = T(1) - std::pow(beta2_, static_cast<T>(t_));
            
            // Update parameters
            #pragma omp parallel for
            for (size_t i = 0; i < weights.size(); ++i) {
                T m_hat = m.data()[i] / bias_correction1;
                T v_hat = v.data()[i] / bias_correction2;
                weights.data()[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    
        void update(Tensor<T, 1>& biases, const Tensor<T, 1>& gradients) override {
            void* biases_ptr = static_cast<void*>(&biases);
            
            // Initialize buffers if needed
            if (bias_m_.find(biases_ptr) == bias_m_.end()) {
                bias_m_[biases_ptr] = Tensor<T, 1>(biases.size());
                bias_v_[biases_ptr] = Tensor<T, 1>(biases.size());
                bias_m_[biases_ptr].fill(T(0));
                bias_v_[biases_ptr].fill(T(0));
            }
            
            Tensor<T, 1>& m = bias_m_[biases_ptr];
            Tensor<T, 1>& v = bias_v_[biases_ptr];
            
            // Update moments and biases (similar to weights)
            #pragma omp parallel for
            for (size_t i = 0; i < biases.size(); ++i) {
                T grad = gradients.data()[i];
                m.data()[i] = beta1_ * m.data()[i] + (T(1) - beta1_) * grad;
                v.data()[i] = beta2_ * v.data()[i] + (T(1) - beta2_) * grad * grad;
            }
            
            T bias_correction1 = T(1) - std::pow(beta1_, static_cast<T>(t_));
            T bias_correction2 = T(1) - std::pow(beta2_, static_cast<T>(t_));
            
            #pragma omp parallel for
            for (size_t i = 0; i < biases.size(); ++i) {
                T m_hat = m.data()[i] / bias_correction1;
                T v_hat = v.data()[i] / bias_correction2;
                biases.data()[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    
        void reset() override {
            weight_m_.clear();
            weight_v_.clear();
            bias_m_.clear();
            bias_v_.clear();
            t_ = 0;
        }
    };
    
    } // namespace ml::optimizers
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                           6. LOSS FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    namespace ml::loss {
    
    template<FloatingPoint T>
    class LossFunctions {
    public:
        // Mean Squared Error
        static T mse_loss(const Matrix<T>& predictions, const Matrix<T>& targets) {
            assert(predictions.shape() == targets.shape());
            
            T sum = T(0);
            #pragma omp parallel for reduction(+:sum)
            for (size_t i = 0; i < predictions.size(); ++i) {
                T diff = predictions.data()[i] - targets.data()[i];
                sum += diff * diff;
            }
            
            return sum / static_cast<T>(predictions.size());
        }
    
        static Matrix<T> mse_gradient(const Matrix<T>& predictions, const Matrix<T>& targets) {
            assert(predictions.shape() == targets.shape());
            
            Matrix<T> gradient(predictions.rows(), predictions.cols());
            T scale = T(2) / static_cast<T>(predictions.size());
            
            #pragma omp parallel for
            for (size_t i = 0; i < predictions.size(); ++i) {
                gradient.data()[i] = scale * (predictions.data()[i] - targets.data()[i]);
            }
            
            return gradient;
        }
    
        // Cross-Entropy Loss
        static T cross_entropy_loss(const Matrix<T>& predictions, const Matrix<T>& targets) {
            assert(predictions.shape() == targets.shape());
            
            T sum = T(0);
            const T epsilon = T(1e-15); // For numerical stability
            
            #pragma omp parallel for reduction(+:sum)
            for (size_t i = 0; i < predictions.size(); ++i) {
                T pred = std::max(std::min(predictions.data()[i], T(1) - epsilon), epsilon);
                sum -= targets.data()[i] * std::log(pred);
            }
            
            return sum / static_cast<T>(predictions.rows());
        }
    
        static Matrix<T> cross_entropy_gradient(const Matrix<T>& predictions, const Matrix<T>& targets) {
            assert(predictions.shape() == targets.shape());
            
            Matrix<T> gradient(predictions.rows(), predictions.cols());
            const T epsilon = T(1e-15);
            T scale = T(1) / static_cast<T>(predictions.rows());
            
            #pragma omp parallel for
            for (size_t i = 0; i < predictions.size(); ++i) {
                T pred = std::max(std::min(predictions.data()[i], T(1) - epsilon), epsilon);
                gradient.data()[i] = -scale * targets.data()[i] / pred;
            }
            
            return gradient;
        }
    
        // Binary Cross-Entropy
        static T binary_cross_entropy_loss(const Matrix<T>& predictions, const Matrix<T>& targets) {
            assert(predictions.shape() == targets.shape());
            
            T sum = T(0);
            const T epsilon = T(1e-15);
            
            #pragma omp parallel for reduction(+:sum)
            for (size_t i = 0; i < predictions.size(); ++i) {
                T pred = std::max(std::min(predictions.data()[i], T(1) - epsilon), epsilon);
                T target = targets.data()[i];
                sum -= target * std::log(pred) + (T(1) - target) * std::log(T(1) - pred);
            }
            
            return sum / static_cast<T>(predictions.rows());
        }
    };
    
    } // namespace ml::loss
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                           7. GPU COMPUTING WITH CUDA
    // ═══════════════════════════════════════════════════════════════════════════════
    
    #ifdef __CUDA_ARCH__
    namespace ml::gpu {
    
    // CUDA kernel for matrix multiplication
    __global__ void matrix_multiply_kernel(const float* A, const float* B, float* C,
                                         int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    
    // CUDA kernel for element-wise operations
    __global__ void element_wise_add_kernel(const float* A, const float* B, float* C, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            C[idx] = A[idx] + B[idx];
        }
    }
    
    // CUDA kernel for ReLU activation
    __global__ void relu_kernel(const float* input, float* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = fmaxf(input[idx], 0.0f);
        }
    }
    
    // GPU Tensor class with CUDA integration
    template<FloatingPoint T>
    class GPUTensor {
    private:
        T* d_data_;
        std::array<size_t, 2> shape_;
        size_t size_;
        
    public:
        GPUTensor(size_t rows, size_t cols) : shape_{rows, cols}, size_(rows * cols) {
            cudaMalloc(&d_data_, size_ * sizeof(T));
        }
        
        ~GPUTensor() {
            if (d_data_) {
                cudaFree(d_data_);
            }
        }
        
        // Copy from host
        void copy_from_host(const T* h_data) {
            cudaMemcpy(d_data_, h_data, size_ * sizeof(T), cudaMemcpyHostToDevice);
        }
        
        // Copy to host
        void copy_to_host(T* h_data) const {
            cudaMemcpy(h_data, d_data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
        }
        
        // GPU matrix multiplication
        GPUTensor multiply(const GPUTensor& other) const {
            assert(shape_[1] == other.shape_[0]);
            
            GPUTensor result(shape_[0], other.shape_[1]);
            
            if constexpr (std::is_same_v<T, float>) {
                // Use cuBLAS for optimized matrix multiplication
                cublasHandle_t handle;
                cublasCreate(&handle);
                
                const float alpha = 1.0f, beta = 0.0f;
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           other.shape_[1], shape_[0], shape_[1],
                           &alpha, other.d_data_, other.shape_[1],
                           d_data_, shape_[1],
                           &beta, result.d_data_, other.shape_[1]);
                
                cublasDestroy(handle);
            }
            
            return result;
        }
        
        // Element-wise addition
        GPUTensor& operator+=(const GPUTensor& other) {
            assert(shape_ == other.shape_);
            
            int block_size = 256;
            int grid_size = (size_ + block_size - 1) / block_size;
            
            if constexpr (std::is_same_v<T, float>) {
                element_wise_add_kernel<<<grid_size, block_size>>>(
                    d_data_, other.d_data_, d_data_, size_);
            }
            
            cudaDeviceSynchronize();
            return *this;
        }
        
        // Apply ReLU activation
        void relu() {
            int block_size = 256;
            int grid_size = (size_ + block_size - 1) / block_size;
            
            if constexpr (std::is_same_v<T, float>) {
                relu_kernel<<<grid_size, block_size>>>(d_data_, d_data_, size_);
            }
            
            cudaDeviceSynchronize();
        }
        
        T* data() { return d_data_; }
        const T* data() const { return d_data_; }
        const std::array<size_t, 2>& shape() const { return shape_; }
        size_t size() const { return size_; }
    };
    
    } // namespace ml::gpu
    #endif
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                           8. CLASSICAL ML ALGORITHMS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    namespace ml::algorithms {
    
    // Linear Regression with analytical solution
    template<FloatingPoint T>
    class LinearRegression {
    private:
        Matrix<T> weights_;
        bool fitted_;
        
    public:
        LinearRegression() : fitted_(false) {}
        
        void fit(const Matrix<T>& X, const Matrix<T>& y) {
            assert(X.rows() == y.rows());
            
            // Add bias term (column of ones)
            Matrix<T> X_bias(X.rows(), X.cols() + 1);
            
            // Copy X and add bias column
            #pragma omp parallel for
            for (size_t i = 0; i < X.rows(); ++i) {
                X_bias(i, 0) = T(1); // bias term
                for (size_t j = 0; j < X.cols(); ++j) {
                    X_bias(i, j + 1) = X(i, j);
                }
            }
            
            // Normal equation: weights = (X^T * X)^(-1) * X^T * y
            Matrix<T> X_T = X_bias.transpose();
            Matrix<T> XTX = X_T.multiply(X_bias);
            
            // For simplicity, assume XTX is invertible
            // In production, use SVD or other robust methods
            Matrix<T> XTy = X_T.multiply(y);
            
            // This would require matrix inversion implementation
            // weights_ = inverse(XTX) * XTy;
            
            fitted_ = true;
        }
        
        Matrix<T> predict(const Matrix<T>& X) const {
            assert(fitted_);
            
            // Add bias term and multiply with weights
            Matrix<T> X_bias(X.rows(), X.cols() + 1);
            
            #pragma omp parallel for
            for (size_t i = 0; i < X.rows(); ++i) {
                X_bias(i, 0) = T(1);
                for (size_t j = 0; j < X.cols(); ++j) {
                    X_bias(i, j + 1) = X(i, j);
                }
            }
            
            return X_bias.multiply(weights_);
        }
    };
    
    // K-Means clustering with parallel implementation
    template<FloatingPoint T>
    class KMeans {
    private:
        size_t k_;
        size_t max_iters_;
        T tolerance_;
        Matrix<T> centroids_;
        std::vector<size_t> labels_;
        bool fitted_;
        
    public:
        KMeans(size_t k, size_t max_iters = 100, T tolerance = T(1e-6))
            : k_(k), max_iters_(max_iters), tolerance_(tolerance), fitted_(false) {}
        
        void fit(const Matrix<T>& X) {
            const size_t n_samples = X.rows();
            const size_t n_features = X.cols();
            
            // Initialize centroids randomly
            centroids_ = Matrix<T>(k_, n_features);
            centroids_.random_uniform(T(-1), T(1));
            
            labels_.resize(n_samples);
            
            for (size_t iter = 0; iter < max_iters_; ++iter) {
                Matrix<T> old_centroids = centroids_;
                
                // Assign points to nearest centroid
                #pragma omp parallel for
                for (size_t i = 0; i < n_samples; ++i) {
                    T min_distance = std::numeric_limits<T>::max();
                    size_t best_centroid = 0;
                    
                    for (size_t j = 0; j < k_; ++j) {
                        T distance = T(0);
                        for (size_t f = 0; f < n_features; ++f) {
                            T diff = X(i, f) - centroids_(j, f);
                            distance += diff * diff;
                        }
                        
                        if (distance < min_distance) {
                            min_distance = distance;
                            best_centroid = j;
                        }
                    }
                    
                    labels_[i] = best_centroid;
                }
                
                // Update centroids
                std::vector<size_t> counts(k_, 0);
                centroids_.fill(T(0));
                
                for (size_t i = 0; i < n_samples; ++i) {
                    size_t cluster = labels_[i];
                    counts[cluster]++;
                    
                    for (size_t f = 0; f < n_features; ++f) {
                        centroids_(cluster, f) += X(i, f);
                    }
                }
                
                // Normalize centroids
                for (size_t j = 0; j < k_; ++j) {
                    if (counts[j] > 0) {
                        for (size_t f = 0; f < n_features; ++f) {
                            centroids_(j, f) /= static_cast<T>(counts[j]);
                        }
                    }
                }
                
                // Check convergence
                T change = T(0);
                for (size_t i = 0; i < centroids_.size(); ++i) {
                    T diff = centroids_.data()[i] - old_centroids.data()[i];
                    change += diff * diff;
                }
                
                if (std::sqrt(change) < tolerance_) {
                    break;
                }
            }
            
            fitted_ = true;
        }
        
        std::vector<size_t> predict(const Matrix<T>& X) const {
            assert(fitted_);
            
            std::vector<size_t> predictions(X.rows());
            
            #pragma omp parallel for
            for (size_t i = 0; i < X.rows(); ++i) {
                T min_distance = std::numeric_limits<T>::max();
                size_t best_centroid = 0;
                
                for (size_t j = 0; j < k_; ++j) {
                    T distance = T(0);
                    for (size_t f = 0; f < X.cols(); ++f) {
                        T diff = X(i, f) - centroids_(j, f);
                        distance += diff * diff;
                    }
                    
                    if (distance < min_distance) {
                        min_distance = distance;
                        best_centroid = j;
                    }
                }
                
                predictions[i] = best_centroid;
            }
            
            return predictions;
        }
        
        const Matrix<T>& centroids() const { return centroids_; }
        const std::vector<size_t>& labels() const { return labels_; }
    };
    
    } // namespace ml::algorithms
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                           9. PERFORMANCE UTILITIES AND PROFILING
    // ═══════════════════════════════════════════════════════════════════════════════
    
    namespace ml::utils {
    
    // High-resolution timer for benchmarking
    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_time_;
        
    public:
        void start() {
            start_time_ = std::chrono::high_resolution_clock::now();
        }
        
        double elapsed_ms() const {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time_);
            return duration.count() / 1000.0;
        }
        
        double elapsed_seconds() const {
            return elapsed_ms() / 1000.0;
        }
    };
    
    // Memory usage profiler
    class MemoryProfiler {
    private:
        size_t peak_memory_;
        size_t current_memory_;
        
    public:
        MemoryProfiler() : peak_memory_(0), current_memory_(0) {}
        
        void allocate(size_t bytes) {
            current_memory_ += bytes;
            peak_memory_ = std::max(peak_memory_, current_memory_);
        }
        
        void deallocate(size_t bytes) {
            current_memory_ = (bytes > current_memory_) ? 0 : current_memory_ - bytes;
        }
        
        size_t peak_memory_mb() const { return peak_memory_ / (1024 * 1024); }
        size_t current_memory_mb() const { return current_memory_ / (1024 * 1024); }
    };
    
    // FLOPS counter for performance analysis
    class FLOPSCounter {
    private:
        size_t total_ops_;
        Timer timer_;
        
    public:
        FLOPSCounter() : total_ops_(0) {}
        
        void start_timing() { timer_.start(); }
        
        void add_ops(size_t ops) { total_ops_ += ops; }
        
        double gflops() const {
            double seconds = timer_.elapsed_seconds();
            return (seconds > 0) ? (total_ops_ / 1e9) / seconds : 0.0;
        }
        
        void reset() { total_ops_ = 0; }
    };
    
    } // namespace ml::utils