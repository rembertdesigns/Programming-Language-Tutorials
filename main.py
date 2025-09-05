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