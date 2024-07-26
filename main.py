# Python  - Beginners Guide - by Richard Rembert
# Comments start with # & will show code below with examples

# First Python script people use 
print("Hello, World!") 


# Primitive (Fundamental) Data Types:
# Primitive data types represent simple values. These data types are the most basic and essential units used to store and manipulate information in a program. They translate directly into low-level machine code.

# Primitive data types include:
String (str): Represents sequences of characters. Should be enclosed in quotes. Example: "Hello, Python!"
Integer (int): Represents whole numbers without decimals. Example: 42
Float (float): Represents numbers with decimals. Example: 3.14
Boolean (bool): Represents either True or False.

# String Example 
name = "John"
# Integer Example
age = 25 
# Float Example 
height = 1.75 
# Boolean Example
is_student = True
# Print variable values 
print("Name:", name) 
print("Age:", age) 
print("Height:", height)
print("Is student?", is_student)

#Output
Name: John
Age: 25
Height: 1.75
Is student? True


# Non-Primitive (Composite) Data Types:
# Non-primitive data types are structures that can hold multiple values and are composed of other data types, including both primitive and other composite types. Unlike primitive data types, non-primitive types allow for more complex and structured representations of data.

# Non-primitive data types include:
List (list): Represents an ordered and mutable collection of values. Example: fruits = ["apple", "banana", "cherry"]
Tuple (tuple): Represents an ordered and immutable collection of values. Example: coordinates = (3, 7)
Dictionary (dict): Represents an unordered collection of key-value pairs. Example: person = {"name": "Alice", "age": 25, "is_student": True}

# List Example
fruits = ["apple", "banana", "cherry"]
print("List Example:", fruits)
# Tuple Example
coordinates = (3, 7)
print("Tuple Example:", coordinates)
# Dictionary Example
person = {"name": "Alice", "age": 25, "is_student": True}
print("Dictionary Example:", person)

# Output
List Example: ['apple', 'banana', 'cherry']
Tuple Example: (3, 7) 
Dictionary Example: {'name': 'Alice', 'age': 25, 'is_student': True}
