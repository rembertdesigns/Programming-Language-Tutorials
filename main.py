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