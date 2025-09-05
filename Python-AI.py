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


# PANDAS - DATA MANIPULATION AND ANALYSIS

import pandas as pd

# Series creation
series = pd.Series([1, 2, 3, 4, 5])
named_series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

# DataFrame creation
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Tokyo', 'Paris'],
    'Salary': [50000, 60000, 70000, 55000]
})

# Reading data from files
# df = pd.read_csv('data.csv')
# df = pd.read_excel('data.xlsx')
# df = pd.read_json('data.json')
# df = pd.read_sql('SELECT * FROM table', connection)

# DataFrame information
print(df.info())                # Data types and null counts
print(df.describe())            # Statistical summary
print(df.head())                # First 5 rows
print(df.tail())                # Last 5 rows
print(df.shape)                 # (rows, columns)
print(df.columns)               # Column names
print(df.dtypes)                # Data types

# Data selection and indexing
names = df['Name']              # Select column
subset = df[['Name', 'Age']]    # Select multiple columns
first_row = df.iloc[0]          # Select by position
alice_data = df.loc[df['Name'] == 'Alice']  # Select by condition

# Filtering data
high_earners = df[df['Salary'] > 55000]
young_people = df[df['Age'] < 30]
complex_filter = df[(df['Age'] > 25) & (df['Salary'] > 50000)]

# Data manipulation
df['Salary_K'] = df['Salary'] / 1000  # Create new column
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Adult')

# Sorting
sorted_by_age = df.sort_values('Age')
sorted_multi = df.sort_values(['City', 'Age'], ascending=[True, False])

# Grouping and aggregation
age_stats = df.groupby('City')['Age'].mean()
salary_stats = df.groupby('City').agg({
    'Salary': ['mean', 'min', 'max'],
    'Age': 'mean'
})

# Handling missing data
# df.isnull()                   # Check for null values
# df.dropna()                   # Remove rows with null values
# df.fillna(0)                  # Fill null values with 0
# df.fillna(df.mean())          # Fill with mean values

# Data transformation
# One-hot encoding
city_encoded = pd.get_dummies(df['City'])

# Date handling
dates = pd.date_range('2024-01-01', periods=10, freq='D')
date_df = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(10)
})
date_df['year'] = date_df['date'].dt.year
date_df['month'] = date_df['date'].dt.month
date_df['weekday'] = date_df['date'].dt.day_name()

# Merging and joining DataFrames
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})

inner_join = pd.merge(df1, df2, on='key', how='inner')  # Inner join
left_join = pd.merge(df1, df2, on='key', how='left')    # Left join
outer_join = pd.merge(df1, df2, on='key', how='outer')  # Outer join


# MATPLOTLIB - DATA VISUALIZATION

import matplotlib.pyplot as plt

# Basic plotting
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.show()

# Multiple plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Line plot
axes[0, 0].plot(x, y)
axes[0, 0].set_title('Line Plot')

# Scatter plot
axes[0, 1].scatter(df['Age'], df['Salary'])
axes[0, 1].set_title('Age vs Salary')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Salary')

# Histogram
axes[1, 0].hist(df['Age'], bins=10, alpha=0.7)
axes[1, 0].set_title('Age Distribution')

# Bar plot
city_counts = df['City'].value_counts()
axes[1, 1].bar(city_counts.index, city_counts.values)
axes[1, 1].set_title('City Distribution')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Advanced plotting features
plt.figure(figsize=(10, 6))

# Multiple series
plt.plot(x, np.sin(x), label='sin(x)', linewidth=2)
plt.plot(x, np.cos(x), label='cos(x)', linewidth=2, linestyle='--')
plt.plot(x, np.tan(x), label='tan(x)', linewidth=2, linestyle=':', alpha=0.7)

plt.xlim(0, 2*np.pi)
plt.ylim(-2, 2)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Trigonometric Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# SEABORN - STATISTICAL VISUALIZATION

import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Create sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'value': np.random.randint(1, 100, 100)
})

# Correlation heatmap
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Histogram with KDE
sns.histplot(data['x'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution with KDE')

# Box plot
sns.boxplot(x='category', y='value', data=data, ax=axes[0, 1])
axes[0, 1].set_title('Box Plot by Category')

# Violin plot
sns.violinplot(x='category', y='value', data=data, ax=axes[1, 0])
axes[1, 0].set_title('Violin Plot by Category')

# Scatter plot with regression
sns.scatterplot(x='x', y='y', hue='category', data=data, ax=axes[1, 1])
sns.regplot(x='x', y='y', data=data, scatter=False, ax=axes[1, 1])
axes[1, 1].set_title('Scatter Plot with Regression')

plt.tight_layout()
plt.show()

# Pair plot for exploring relationships
# sns.pairplot(df, hue='City')
# plt.show()