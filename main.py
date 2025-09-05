# PYTHON - Comprehensive Programming Language - by Richard Rembert

# Comments start with # and will show code below with examples

# Python is a high-level, interpreted programming language known for its simple syntax
# and readability, making it ideal for beginners and powerful for advanced applications


# SETUP AND BASICS

# Install Python from python.org or use package managers
# python --version  # Check Python version
# pip install package_name  # Install packages
# pip freeze  # List installed packages
# pip freeze > requirements.txt  # Save dependencies
# pip install -r requirements.txt  # Install from requirements file

# Run Python files
# python filename.py
# python3 filename.py  # On systems with both Python 2 and 3

# Interactive Python shell
# python  # Start interactive mode
# exit()  # Exit interactive mode

# First Python script people use 
print("Hello, World!") 


# PRIMITIVE (FUNDAMENTAL) DATA TYPES

# String (str): Represents sequences of characters
name = "John"
message = 'Single quotes work too'
multiline = """This is a
multiline string"""
template = f"Hello, {name}!"  # f-string formatting

# Integer (int): Represents whole numbers
age = 25
big_number = 1_000_000  # Underscores for readability
binary = 0b1010  # Binary (equals 10)
octal = 0o12    # Octal (equals 10)
hexadecimal = 0xa  # Hexadecimal (equals 10)

# Float (float): Represents numbers with decimals
height = 1.75
scientific = 2.5e-4  # Scientific notation (0.00025)

# Boolean (bool): Represents True or False
is_student = True
is_employed = False

# None: Represents absence of value
result = None

# Print variable values 
print("Name:", name) 
print("Age:", age) 
print("Height:", height)
print("Is student?", is_student)

# TYPE CHECKING AND CONVERSION

# Check type of variables
print(type(name))     # <class 'str'>
print(type(age))      # <class 'int'>
print(type(height))   # <class 'float'>

# Type conversion
age_str = str(age)           # Convert to string
height_int = int(height)     # Convert to integer (truncates)
age_float = float(age)       # Convert to float
is_true = bool(1)           # Convert to boolean (1 = True, 0 = False)

# Safe type conversion with try/except
def safe_int_convert(value):
    try:
        return int(value)
    except ValueError:
        return None

number = safe_int_convert("123")  # Returns 123
invalid = safe_int_convert("abc")  # Returns None


# NON-PRIMITIVE (COMPOSITE) DATA TYPES

# List: Ordered, mutable collection of values
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", True, 3.14]  # Can contain different types

# List operations
fruits.append("orange")        # Add to end
fruits.insert(1, "grape")     # Insert at index
removed = fruits.pop()         # Remove and return last item
fruits.remove("banana")       # Remove specific item
length = len(fruits)          # Get length
first_fruit = fruits[0]       # Access by index
last_fruit = fruits[-1]       # Negative indexing
sliced = fruits[1:3]          # Slicing [start:end]

# Tuple: Ordered, immutable collection of values
coordinates = (3, 7)
person = ("Alice", 25, True)
single_item = (42,)  # Note the comma for single-item tuple

# Tuple operations
x, y = coordinates            # Tuple unpacking
name, age, status = person    # Multiple assignment

# Dictionary: Unordered collection of key-value pairs
person = {
    "name": "Alice",
    "age": 25,
    "is_student": True,
    "grades": [85, 90, 78]
}

# Dictionary operations
person["city"] = "New York"        # Add new key-value pair
name = person["name"]              # Access value by key
name = person.get("name", "Unknown")  # Safe access with default
del person["age"]                  # Delete key-value pair
keys = person.keys()               # Get all keys
values = person.values()           # Get all values
items = person.items()             # Get key-value pairs

# Set: Unordered collection of unique values
colors = {"red", "green", "blue"}
numbers_set = {1, 2, 3, 3, 4}  # Duplicates automatically removed

# Set operations
colors.add("yellow")              # Add element
colors.discard("red")             # Remove element (no error if not found)
colors.remove("green")            # Remove element (error if not found)
is_member = "blue" in colors      # Check membership
union = colors | {"purple", "orange"}     # Union
intersection = colors & {"blue", "yellow"}  # Intersection