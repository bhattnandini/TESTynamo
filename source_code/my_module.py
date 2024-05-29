def is_not_integer(num):
    return not isinstance(num, int)

def is_not_float(num):
    return not isinstance(num, float)

# Assumptions
# a is an integer 
# b is a floating point number
def add(a, b):
    if is_not_integer(a):
        raise ValueError("A must be an integer!")
    
    if is_not_float(b):
        raise ValueError("B must be a floating point number!")
    
    return a + b

# a is a floating point number greater than b
# b is an integer
def subtract(a, b):
    if is_not_float(a):
        raise ValueError("A must be a floating point number!")
    
    if is_not_integer(b):
        raise ValueError("B must be an integer!")
   
    return a - b

# a is a non-negative floating point number
# b is a floating point number
# The product of a and b must be in the range [-1000, 1000]
def multiply(a, b):
    if is_not_float(a):
        raise ValueError("A must be a floating point number!")
    
    if is_not_float(b):
        raise ValueError("B must be a floating point number!")
    
    if a < 0:
        raise ValueError("A cannot be negative!")
    
    if a*b < -1000 or a*b > 1000:
        raise ValueError("Product out-of-range!")
    
    return a * b

# a is a floating point number greater than b
# b is a floating point number except 0
def divide(a, b):
    if is_not_float(a):
        raise ValueError("A must be a floating point number!")
    
    if is_not_float(b):
        raise ValueError("B must be a floating point number!")
    
    if a < b:
        raise ValueError("A cannot be lesser than B!")

    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b