# PYTHON AI/ML - Artificial Intelligence and Machine Learning Reference - by Richard Rembert

# Python has become the dominant language for AI, ML, and Data Science
# This reference covers the essential libraries, frameworks, and patterns for modern AI development

# SETUP AND INSTALLATION

# Essential AI/ML packages installation
# pip install numpy pandas matplotlib seaborn scikit-learn
# pip install tensorflow keras torch torchvision
# pip install jupyter jupyterlab
# pip install openai anthropic langchain
# pip install transformers datasets accelerate
# pip install plotly streamlit gradio

# Create virtual environment for AI projects
# python -m venv ai_env
# source ai_env/bin/activate  # On Windows: ai_env\Scripts\activate
# pip install -r requirements.txt

# Jupyter notebook setup
# pip install jupyter
# jupyter notebook  # Start Jupyter
# jupyter lab       # Start JupyterLab (modern interface)

# GPU support (if available)
# pip install tensorflow-gpu  # For TensorFlow
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For PyTorch with CUDA


# NUMPY - NUMERICAL COMPUTING FOUNDATION

import numpy as np

# Array creation
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Array creation functions
zeros = np.zeros((3, 4))          # 3x4 array of zeros
ones = np.ones((2, 3))            # 2x3 array of ones
identity = np.eye(3)              # 3x3 identity matrix
random_arr = np.random.random((2, 3))  # Random values 0-1
range_arr = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)   # 5 equally spaced values from 0 to 1

# Array properties
print(f"Shape: {arr_2d.shape}")      # (2, 3)
print(f"Size: {arr_2d.size}")        # 6
print(f"Dimensions: {arr_2d.ndim}")  # 2
print(f"Data type: {arr_2d.dtype}")  # int64

# Array operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
addition = a + b        # [5, 7, 9]
subtraction = a - b     # [-3, -3, -3]
multiplication = a * b  # [4, 10, 18]
division = a / b        # [0.25, 0.4, 0.5]
power = a ** 2          # [1, 4, 9]

# Mathematical functions
sqrt_a = np.sqrt(a)     # Square root
exp_a = np.exp(a)       # Exponential
log_a = np.log(a)       # Natural logarithm
sin_a = np.sin(a)       # Sine

# Array indexing and slicing
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# Basic indexing
first_row = matrix[0]           # [1, 2, 3, 4]
first_element = matrix[0, 0]    # 1
last_element = matrix[-1, -1]   # 12

# Slicing
submatrix = matrix[0:2, 1:3]    # [[2, 3], [6, 7]]
every_other = matrix[::2, ::2]  # [[1, 3], [9, 11]]

# Boolean indexing
mask = matrix > 5
filtered = matrix[mask]         # [6, 7, 8, 9, 10, 11, 12]

# Array reshaping and manipulation
reshaped = matrix.reshape(4, 3)  # Reshape to 4x3
flattened = matrix.flatten()     # 1D array
transposed = matrix.T            # Transpose

# Statistical operations
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mean = np.mean(data)            # 5.5
median = np.median(data)        # 5.5
std = np.std(data)              # Standard deviation
var = np.var(data)              # Variance
min_val = np.min(data)          # 1
max_val = np.max(data)          # 10
sum_val = np.sum(data)          # 55

# Linear algebra operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
dot_product = np.dot(A, B)      # Matrix multiplication
matmul = A @ B                  # Alternative syntax (Python 3.5+)

# Matrix operations
determinant = np.linalg.det(A)  # Determinant
inverse = np.linalg.inv(A)      # Inverse
eigenvals, eigenvecs = np.linalg.eig(A)  # Eigenvalues and eigenvectors