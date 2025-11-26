# test_math.py

import pytest

# --- Functions to be tested (Your original code) ---

def add(a, b):
    """Returns the sum of two numbers."""
    return a + b

def subtract(a, b):
    """Returns the difference between two numbers."""
    return a - b

def multiply(a, b):
    """Returns the product of two numbers."""
    return a * b

def divide(a, b):
    """Divides a by b, handles division by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b

# --- Pytest Test Functions ---

def test_add_positive():
    """Test addition of two positive integers."""
    assert add(5, 3) == 8

def test_subtract_simple():
    """Test basic subtraction."""
    assert subtract(10, 5) == 5
    assert subtract(5, 10) == -5

def test_multiply_negative():
    """Test multiplication involving negative numbers."""
    assert multiply(-2, 4) == -8
    assert multiply(-5, -5) == 25

# Example using 'parametrize' for multiple division tests
@pytest.mark.parametrize("numerator, denominator, expected", [
    (10, 2, 5.0),
    (1, 4, 0.25),
    (-8, 2, -4.0),
])
def test_divide_valid(numerator, denominator, expected):
    """Test division with various valid inputs."""
    assert divide(numerator, denominator) == expected

def test_divide_by_zero_error():
    """Test that division by zero raises a ValueError."""
    # Use pytest.raises to assert that a specific exception occurs
    with pytest.raises(ValueError) as excinfo:
        divide(10, 0)
    # Optionally, check the error message
    assert "Cannot divide by zero!" in str(excinfo.value)