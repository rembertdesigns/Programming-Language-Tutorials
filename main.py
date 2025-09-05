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

# OPERATORS

# Arithmetic operators
result = 10 + 3   # Addition (13)
result = 10 - 3   # Subtraction (7)
result = 10 * 3   # Multiplication (30)
result = 10 / 3   # Division (3.333...)
result = 10 // 3  # Floor division (3)
result = 10 % 3   # Modulo/remainder (1)
result = 10 ** 3  # Exponentiation (1000)

# Comparison operators
is_equal = 10 == 10        # Equal to (True)
not_equal = 10 != 5        # Not equal to (True)
greater = 10 > 5           # Greater than (True)
less = 10 < 5              # Less than (False)
greater_equal = 10 >= 10   # Greater than or equal (True)
less_equal = 10 <= 5       # Less than or equal (False)

# Logical operators
result = True and False    # Logical AND (False)
result = True or False     # Logical OR (True)
result = not True          # Logical NOT (False)

# Assignment operators
x = 10
x += 5    # Same as x = x + 5
x -= 3    # Same as x = x - 3
x *= 2    # Same as x = x * 2
x /= 4    # Same as x = x / 4

# Identity operators
a = [1, 2, 3]
b = a
c = [1, 2, 3]
print(a is b)      # True (same object)
print(a is c)      # False (different objects)
print(a == c)      # True (same content)

# Membership operators
fruits = ["apple", "banana", "cherry"]
print("apple" in fruits)      # True
print("grape" not in fruits)  # True

# CONTROL STRUCTURES

# If statements
age = 18

if age >= 18:
    print("You are an adult")
elif age >= 13:
    print("You are a teenager")
else:
    print("You are a child")

# Ternary operator (conditional expression)
status = "adult" if age >= 18 else "minor"

# Match statement (Python 3.10+)
def handle_status(status):
    match status:
        case "active":
            return "User is active"
        case "inactive":
            return "User is inactive"
        case "pending":
            return "User registration pending"
        case _:  # Default case
            return "Unknown status"
        
# LOOPS

# For loops
# Iterate over list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Iterate with index
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")

# Iterate over range
for i in range(5):          # 0 to 4
    print(i)

for i in range(1, 6):       # 1 to 5
    print(i)

for i in range(0, 10, 2):   # 0, 2, 4, 6, 8
    print(i)

# Iterate over dictionary
person = {"name": "Alice", "age": 25}
for key in person:
    print(key, person[key])

for key, value in person.items():
    print(f"{key}: {value}")

# While loops
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1

# Loop control statements
for i in range(10):
    if i == 3:
        continue  # Skip this iteration
    if i == 7:
        break     # Exit loop
    print(i)

# Loop with else clause
for i in range(5):
    print(i)
else:
    print("Loop completed normally")  # Runs if loop wasn't broken


# FUNCTIONS

# Basic function
def greet():
    print("Hello!")

greet()  # Call function

# Function with parameters
def greet_person(name):
    print(f"Hello, {name}!")

greet_person("Alice")

# Function with return value
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)

# Function with default parameters
def greet_with_title(name, title="Mr."):
    return f"Hello, {title} {name}!"

print(greet_with_title("Smith"))           # Uses default title
print(greet_with_title("Smith", "Dr."))    # Uses provided title

# Function with keyword arguments
def create_profile(name, age, city="Unknown", occupation="Student"):
    return {
        "name": name,
        "age": age,
        "city": city,
        "occupation": occupation
    }

profile = create_profile("Alice", 25, occupation="Engineer")

# Function with *args (variable positional arguments)
def sum_all(*numbers):
    return sum(numbers)

total = sum_all(1, 2, 3, 4, 5)

# Function with **kwargs (variable keyword arguments)
def create_user(**details):
    return details

user = create_user(name="Alice", age=25, city="New York")

# Lambda functions (anonymous functions)
square = lambda x: x ** 2
add = lambda a, b: a + b

numbers = [1, 2, 3, 4, 5]
squared = list(map(square, numbers))      # Apply function to each element
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))  # Filter elements

# Function decorators
def timer_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer_decorator
def slow_function():
    import time
    time.sleep(1)
    return "Done"

# Generator functions
def count_up_to(n):
    count = 1
    while count <= n:
        yield count
        count += 1

for number in count_up_to(5):
    print(number)


# CLASSES AND OBJECTS

# Basic class
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old"
    
    def have_birthday(self):
        self.age += 1

# Create objects
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

print(person1.introduce())
person1.have_birthday()
print(f"Alice is now {person1.age}")

# Class with class variables and methods
class BankAccount:
    bank_name = "Python Bank"  # Class variable
    accounts_created = 0       # Class variable
    
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        BankAccount.accounts_created += 1
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            return True
        return False
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False
    
    @classmethod
    def get_bank_info(cls):
        return f"Bank: {cls.bank_name}, Accounts: {cls.accounts_created}"
    
    @staticmethod
    def validate_amount(amount):
        return amount > 0
    
    def __str__(self):
        return f"Account({self.owner}: ${self.balance})"

account = BankAccount("Alice", 1000)
account.deposit(500)
print(account)  # Uses __str__ method

# Inheritance
class SavingsAccount(BankAccount):
    def __init__(self, owner, balance=0, interest_rate=0.02):
        super().__init__(owner, balance)  # Call parent constructor
        self.interest_rate = interest_rate
    
    def add_interest(self):
        interest = self.balance * self.interest_rate
        self.balance += interest
        return interest

savings = SavingsAccount("Bob", 1000, 0.05)
interest_earned = savings.add_interest()

# Property decorators
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9

temp = Temperature(25)
print(temp.fahrenheit)  # 77.0
temp.fahrenheit = 86
print(temp.celsius)     # 30.0


# FILE HANDLING

# Reading files
# Read entire file
with open("example.txt", "r") as file:
    content = file.read()

# Read line by line
with open("example.txt", "r") as file:
    for line in file:
        print(line.strip())  # strip() removes newline characters

# Read all lines into a list
with open("example.txt", "r") as file:
    lines = file.readlines()

# Writing files
# Write to file (overwrites existing content)
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is a new line")

# Append to file
with open("output.txt", "a") as file:
    file.write("\nThis line is appended")

# Write multiple lines
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open("output.txt", "w") as file:
    file.writelines(lines)

# Working with CSV files
import csv

# Write CSV
data = [
    ["Name", "Age", "City"],
    ["Alice", 25, "New York"],
    ["Bob", 30, "San Francisco"]
]

with open("people.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Read CSV
with open("people.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# CSV with dictionaries
with open("people.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(f"{row['Name']} is {row['Age']} years old")

# Working with JSON
import json

# Write JSON
data = {"name": "Alice", "age": 25, "city": "New York"}
with open("person.json", "w") as file:
    json.dump(data, file, indent=2)

# Read JSON
with open("person.json", "r") as file:
    person = json.load(file)
    print(person["name"])

# JSON string conversion
json_string = json.dumps(data)  # Convert to JSON string
parsed_data = json.loads(json_string)  # Parse JSON string


# ERROR HANDLING

# Basic try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
try:
    number = int(input("Enter a number: "))
    result = 10 / number
except ValueError:
    print("Invalid input - not a number")
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Catch multiple exceptions together
try:
    # Some risky code
    pass
except (ValueError, TypeError) as e:
    print(f"Error occurred: {e}")

# Catch all exceptions
try:
    # Some risky code
    pass
except Exception as e:
    print(f"Unexpected error: {e}")

# Finally block (always executes)
try:
    file = open("somefile.txt", "r")
    # Process file
except FileNotFoundError:
    print("File not found")
finally:
    # This always runs
    print("Cleanup code here")

# Else block (runs if no exception occurred)
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Division error")
else:
    print(f"Result: {result}")  # Only runs if no exception

# Raising exceptions
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age seems unrealistic")
    return True

# Custom exceptions
class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def risky_function():
    raise CustomError("Something went wrong in my function")


# LIST COMPREHENSIONS AND GENERATORS

# List comprehensions
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]                    # [1, 4, 9, 16, 25]
even_squares = [x**2 for x in numbers if x % 2 == 0] # [4, 16]

# Nested list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]   # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Dictionary comprehension
words = ["hello", "world", "python"]
word_lengths = {word: len(word) for word in words}   # {'hello': 5, 'world': 5, 'python': 6}

# Set comprehension
numbers = [1, 2, 2, 3, 3, 4, 5]
unique_squares = {x**2 for x in numbers}             # {1, 4, 9, 16, 25}

# Generator expressions (memory efficient)
numbers_gen = (x**2 for x in range(1000000))  # Doesn't create list immediately
first_five = [next(numbers_gen) for _ in range(5)]

# Generator functions
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
first_ten_fib = [next(fib) for _ in range(10)]


# ADVANCED FEATURES

# Context managers
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# Use custom context manager
with FileManager("test.txt", "w") as f:
    f.write("Hello from context manager")

# Using contextlib
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"Elapsed time: {end - start:.4f} seconds")

with timer():
    # Some time-consuming operation
    sum(range(1000000))

# Multiple inheritance and method resolution order
class A:
    def method(self):
        print("Method from A")

class B:
    def method(self):
        print("Method from B")

class C(A, B):  # Multiple inheritance
    pass

c = C()
c.method()  # Prints "Method from A" (left-to-right MRO)
print(C.__mro__)  # Shows method resolution order

# Magic methods (dunder methods)
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __len__(self):
        return int((self.x**2 + self.y**2)**0.5)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

v1 = Vector(2, 3)
v2 = Vector(1, 4)
v3 = v1 + v2  # Uses __add__
print(v3)     # Uses __str__


# MODULES AND PACKAGES

# Importing modules
import math
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json as js  # Alias

# Using imported modules
print(math.pi)
print(os.getcwd())  # Current working directory
now = datetime.now()

# Creating your own module (save as mymodule.py)
"""
# mymodule.py
def greet(name):
    return f"Hello, {name}!"

PI = 3.14159

class Calculator:
    @staticmethod
    def add(a, b):
        return a + b
"""

# Using your module
# from mymodule import greet, PI, Calculator

# Module search path
import sys
print(sys.path)  # Shows where Python looks for modules

# Installing packages with pip
# pip install requests
# pip install pandas numpy matplotlib


# USEFUL BUILT-IN FUNCTIONS

# String methods
text = "  Hello, World!  "
print(text.strip())           # Remove whitespace
print(text.lower())           # Convert to lowercase
print(text.upper())           # Convert to uppercase
print(text.replace("World", "Python"))  # Replace substring
print(text.split(","))        # Split into list
print("Hello" in text)        # Check if substring exists

# String formatting
name = "Alice"
age = 25
print("Name: %s, Age: %d" % (name, age))        # Old style
print("Name: {}, Age: {}".format(name, age))    # New style
print(f"Name: {name}, Age: {age}")              # f-strings (recommended)

# Math operations
import math
print(abs(-5))          # Absolute value
print(round(3.14159, 2)) # Round to 2 decimal places
print(max([1, 5, 3]))   # Maximum value
print(min([1, 5, 3]))   # Minimum value
print(sum([1, 2, 3]))   # Sum of values
print(math.ceil(3.2))   # Ceiling (4)
print(math.floor(3.8))  # Floor (3)
print(math.sqrt(16))    # Square root (4.0)

# Working with iterables
numbers = [1, 2, 3, 4, 5]
print(len(numbers))                    # Length
print(any([True, False, False]))       # True if any element is True
print(all([True, True, False]))        # True if all elements are True
print(list(reversed(numbers)))         # Reverse iterable
print(sorted([3, 1, 4, 1, 5]))        # Sort iterable
print(list(enumerate(numbers)))        # Add indices: [(0, 1), (1, 2), ...]
print(list(zip([1, 2, 3], ['a', 'b', 'c'])))  # Combine iterables

# Type checking
print(isinstance(42, int))      # True
print(isinstance("hello", str)) # True
print(hasattr([], 'append'))    # True


# REGULAR EXPRESSIONS

import re

# Basic pattern matching
text = "The phone number is 123-456-7890"
pattern = r"\d{3}-\d{3}-\d{4}"
match = re.search(pattern, text)
if match:
    print(f"Found phone number: {match.group()}")

# Common regex patterns
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
url_pattern = r"https?://[^\s]+"
word_pattern = r"\b\w+\b"

# Regex functions
text = "Contact us at info@example.com or support@test.org"
emails = re.findall(email_pattern, text)  # Find all matches
text_replaced = re.sub(email_pattern, "[EMAIL]", text)  # Replace matches

# Compile regex for better performance
compiled_pattern = re.compile(email_pattern)
matches = compiled_pattern.findall(text)


# DATE AND TIME

from datetime import datetime, date, time, timedelta

# Current date and time
now = datetime.now()
today = date.today()
current_time = datetime.now().time()

# Creating specific dates
birthday = date(1990, 5, 15)
meeting = datetime(2024, 12, 25, 14, 30)  # Dec 25, 2024 at 2:30 PM

# Formatting dates
formatted = now.strftime("%Y-%m-%d %H:%M:%S")  # 2024-01-15 14:30:45
print(formatted)

# Parsing date strings
date_string = "2024-01-15"
parsed_date = datetime.strptime(date_string, "%Y-%m-%d")

# Date arithmetic
tomorrow = today + timedelta(days=1)
next_week = today + timedelta(weeks=1)
last_month = today - timedelta(days=30)

# Working with timezones
from datetime import timezone
utc_now = datetime.now(timezone.utc)


# WORKING WITH APIs AND HTTP

import requests  # pip install requests

# GET request
response = requests.get("https://api.github.com/users/octocat")
if response.status_code == 200:
    user_data = response.json()
    print(user_data["name"])

# POST request
data = {"name": "John", "email": "john@example.com"}
response = requests.post("https://api.example.com/users", json=data)

# Request with headers
headers = {"Authorization": "Bearer your-token-here"}
response = requests.get("https://api.example.com/data", headers=headers)

# Error handling with requests
try:
    response = requests.get("https://api.example.com/data", timeout=5)
    response.raise_for_status()  # Raises exception for bad status codes
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")


# TESTING

import unittest

# Basic unit test
class TestCalculator(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(2 + 2, 4)
    
    def test_division(self):
        self.assertEqual(10 / 2, 5)
        with self.assertRaises(ZeroDivisionError):
            10 / 0
    
    def test_list_operations(self):
        test_list = [1, 2, 3]
        self.assertIn(2, test_list)
        self.assertEqual(len(test_list), 3)

# Run tests
if __name__ == "__main__":
    unittest.main()

# Using pytest (pip install pytest)
# pytest is more popular and easier to use
def test_simple_addition():
    assert 2 + 2 == 4

def test_string_operations():
    text = "hello world"
    assert "world" in text
    assert text.upper() == "HELLO WORLD"


# VIRTUAL ENVIRONMENTS

# Create virtual environment
# python -m venv myenv

# Activate virtual environment
# On Windows: myenv\Scripts\activate
# On macOS/Linux: source myenv/bin/activate

# Deactivate virtual environment
# deactivate

# Install packages in virtual environment
# pip install package_name

# Freeze dependencies
# pip freeze > requirements.txt

# Install from requirements
# pip install -r requirements.txt


# BEST PRACTICES

# 1. Use meaningful variable names
user_age = 25  # Good
a = 25         # Bad

# 2. Follow PEP 8 style guide
def calculate_area(length, width):  # Good: snake_case for functions
    return length * width

# 3. Use docstrings for documentation
def calculate_compound_interest(principal, rate, time):
    """
    Calculate compound interest.
    
    Args:
        principal (float): Initial amount
        rate (float): Interest rate (as decimal)
        time (int): Time period in years
    
    Returns:
        float: Final amount after compound interest
    """
    return principal * (1 + rate) ** time

# 4. Handle errors appropriately
def safe_divide(a, b):
    """Safely divide two numbers."""
    try:
        return a / b
    except ZeroDivisionError:
        return None
    except TypeError:
        raise TypeError("Both arguments must be numbers")

# 5. Use list comprehensions for simple operations
# Good
squares = [x**2 for x in range(10)]

# Less efficient
squares = []
for x in range(10):
    squares.append(x**2)

# 6. Use constants for magic numbers
PI = 3.14159
DAYS_IN_WEEK = 7
MAX_ATTEMPTS = 3

# 7. Keep functions small and focused
def get_user_input():
    """Get and validate user input."""
    while True:
        try:
            return int(input("Enter a number: "))
        except ValueError:
            print("Please enter a valid number")

# 8. Use type hints (Python 3.5+)
def greet_user(name: str, age: int) -> str:
    """Greet a user with their name and age."""
    return f"Hello {name}, you are {age} years old"

from typing import List, Dict, Optional

def process_names(names: List[str]) -> Dict[str, int]:
    """Process a list of names and return their lengths."""
    return {name: len(name) for name in names}