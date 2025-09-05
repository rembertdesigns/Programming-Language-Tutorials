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


# SCIKIT-LEARN - MACHINE LEARNING

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification, make_regression, load_iris, load_boston

# Data preprocessing
# Load sample dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification Example
# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

results = {}
for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Cross-validation
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\n{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Regression Example
# Generate regression dataset
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Scale features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Train regression models
reg_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in reg_models.items():
    model.fit(X_train_reg_scaled, y_train_reg)
    y_pred_reg = model.predict(X_test_reg_scaled)
    
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    print(f"\n{name} Regression Results:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

# Clustering Example
# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_train_scaled)

# Visualize clusters (if 2D)
if X_train_scaled.shape[1] >= 2:
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=cluster_labels, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    plt.legend()
    plt.colorbar(scatter)
    plt.show()


# DEEP LEARNING WITH TENSORFLOW/KERAS

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical

# Check TensorFlow version and GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Neural Network for MNIST digit classification
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Train model
history = model.fit(x_train, y_train,
                   batch_size=128,
                   epochs=10,
                   validation_split=0.1,
                   verbose=1)

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Convolutional Neural Network for image classification
def create_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# For CIFAR-10 dataset
# (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()
# cnn_model = create_cnn_model((32, 32, 3), 10)

# Transfer Learning Example
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom top layers
transfer_model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # Adjust for your number of classes
])

transfer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Custom training loop example
@tf.function
def train_step(x, y, model, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# plot_training_history(history)


# PYTORCH - ALTERNATIVE DEEP LEARNING FRAMEWORK

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Check PyTorch setup
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model instance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN(input_size=784, hidden_size=128, num_classes=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(x_train.reshape(-1, 784))
y_tensor = torch.LongTensor(np.argmax(y_train, axis=1))

# Create data loader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training loop
def train_pytorch_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

# train_pytorch_model(model, dataloader, criterion, optimizer)


# HUGGING FACE TRANSFORMERS - PRE-TRAINED MODELS

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertModel
)

# Text classification pipeline
classifier = pipeline("sentiment-analysis")
results = classifier(["I love this product!", "This is terrible.", "It's okay."])
for result in results:
    print(f"Text: {result}")

# Named Entity Recognition
ner = pipeline("ner", aggregation_strategy="simple")
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
entities = ner(text)
for entity in entities:
    print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.4f}")

# Text Generation
generator = pipeline("text-generation", model="gpt2")
prompts = ["The future of artificial intelligence is"]
generated = generator(prompts, max_length=50, num_return_sequences=2)
for gen in generated:
    print(f"Generated: {gen['generated_text']}")

# Question Answering
qa_pipeline = pipeline("question-answering")
context = """
Machine learning is a subset of artificial intelligence that focuses on algorithms 
that can learn and make decisions from data without being explicitly programmed.
"""
question = "What is machine learning?"
answer = qa_pipeline(question=question, context=context)
print(f"Answer: {answer['answer']}, Score: {answer['score']:.4f}")

# Working with BERT for embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts):
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**encoded)
        # Use the [CLS] token embedding as sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
    return embeddings

texts = ["I love machine learning", "Natural language processing is fascinating"]
embeddings = get_bert_embeddings(texts)
print(f"Embedding shape: {embeddings.shape}")

# Fine-tuning example (simplified)
def fine_tune_classifier(train_texts, train_labels, model_name="bert-base-uncased"):
    from transformers import Trainer, TrainingArguments
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Tokenize training data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
    
    train_dataset = Dataset(train_encodings, train_labels)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    return model, tokenizer


# LLM INTEGRATION - OPENAI AND ANTHROPIC APIs

# OpenAI API integration
import openai
import os

# Set up OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

def chat_with_gpt(messages, model="gpt-3.5-turbo", temperature=0.7):
    """
    Chat with OpenAI's GPT models
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

# Example usage
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Explain machine learning in simple terms."}
# ]
# response = chat_with_gpt(messages)
# print(response)

# Anthropic Claude API integration
import anthropic

# Set up Anthropic API key
# anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def chat_with_claude(message, model="claude-3-sonnet-20240229", max_tokens=1000):
    """
    Chat with Anthropic's Claude models
    """
    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": message}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return None

# Example usage
# response = chat_with_claude("Explain the difference between supervised and unsupervised learning.")
# print(response)

# Function calling with OpenAI
def get_weather(location):
    """Mock function to get weather data"""
    return f"The weather in {location} is sunny with 75°F"

def function_calling_example():
    """
    Example of using OpenAI function calling
    """
    functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "What's the weather like in New York?"}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=functions,
            function_call="auto"
        )
        
        if response.choices[0].message.get("function_call"):
            function_name = response.choices[0].message["function_call"]["name"]
            function_args = eval(response.choices[0].message["function_call"]["arguments"])
            
            if function_name == "get_weather":
                result = get_weather(function_args["location"])
                return result
                
    except Exception as e:
        print(f"Error with function calling: {e}")
        return None


# LANGCHAIN - LLM APPLICATION FRAMEWORK

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.document_loaders import TextLoader, PDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Basic LLM setup
# llm = OpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
# chat_model = ChatOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Prompt templates
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a brief explanation about {topic} for beginners."
)

# LLM Chain
# chain = LLMChain(llm=llm, prompt=prompt_template)
# result = chain.run("machine learning")

# Chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specializing in {domain}."),
    ("human", "{question}")
])

# Memory for conversations
memory = ConversationBufferMemory()

# Sequential chains
def create_sequential_chain():
    """
    Create a sequential chain for complex tasks
    """
    # First chain: Generate a topic outline
    outline_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Create an outline for a tutorial about {topic}."
    )
    # outline_chain = LLMChain(llm=llm, prompt=outline_prompt)
    
    # Second chain: Write content based on outline
    content_prompt = PromptTemplate(
        input_variables=["outline"],
        template="Write detailed content based on this outline:\n{outline}"
    )
    # content_chain = LLMChain(llm=llm, prompt=content_prompt)
    
    # Combine chains
    # sequential_chain = SimpleSequentialChain(chains=[outline_chain, content_chain])
    # return sequential_chain

# Document processing and Q&A
def create_document_qa_system(file_path):
    """
    Create a question-answering system for documents
    """
    # Load documents
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Create Q&A chain
    from langchain.chains import RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa

# Custom tools for agents
def calculator_tool(expression):
    """Simple calculator tool"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

def search_tool(query):
    """Mock search tool"""
    return f"Search results for: {query}"

# Create agent with tools
def create_agent():
    """
    Create an agent with custom tools
    """
    tools = [
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Useful for mathematical calculations"
        ),
        Tool(
            name="Search",
            func=search_tool,
            description="Useful for searching information"
        )
    ]
    
    # agent = initialize_agent(
    #     tools,
    #     llm,
    #     agent="zero-shot-react-description",
    #     verbose=True
    # )
    # return agent


# COMPUTER VISION

import cv2
from PIL import Image
import face_recognition

# OpenCV basics
def process_image_opencv(image_path):
    """
    Basic image processing with OpenCV
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    return result, gray, edges

# Face recognition
def detect_faces(image_path):
    """
    Detect and recognize faces in an image
    """
    # Load image
    image = face_recognition.load_image_file(image_path)
    
    # Find face locations
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    print(f"Found {len(face_locations)} face(s) in the image")
    
    return face_locations, face_encodings

# Object detection with YOLO (requires additional setup)
def setup_yolo_detection():
    """
    Setup YOLO for object detection
    Note: Requires downloading YOLO weights and config files
    """
    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # Load class names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, classes

def detect_objects(image_path, net, classes):
    """
    Detect objects in an image using YOLO
    """
    # Load image
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    
    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward()
    
    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids


# NATURAL LANGUAGE PROCESSING

import nltk
import spacy
from textblob import TextBlob
from wordcloud import WordCloud

# Download required NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# Basic text preprocessing
def preprocess_text(text):
    """
    Basic text preprocessing pipeline
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    import re
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

# Sentiment analysis
def analyze_sentiment(text):
    """
    Analyze sentiment using TextBlob and NLTK
    """
    # TextBlob sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # NLTK VADER sentiment
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(text)
    
    return {
        'textblob_polarity': polarity,
        'textblob_subjectivity': subjectivity,
        'vader_scores': vader_scores
    }

# Named Entity Recognition with spaCy
def extract_entities(text):
    """
    Extract named entities using spaCy
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
    return entities

# Topic modeling with LDA
def topic_modeling(documents, num_topics=5):
    """
    Perform topic modeling using LDA
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    
    # Vectorize documents
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Display topics
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        topics.append(top_words)
    
    return topics, lda, vectorizer

# Word cloud generation
def create_wordcloud(text, width=800, height=400):
    """
    Create a word cloud from text
    """
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
    return wordcloud