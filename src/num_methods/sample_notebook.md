
# General Numerical Methods

## Introduction

Numerical methods are essential in solving mathematical problems that are difficult or impossible to tackle analytically. These methods approximate solutions to problems, ranging from finding roots of equations, integrating functions, differentiating functions, to error analysis and the implementation of adaptive algorithms. This notebook will guide you through core numerical methods, illustrating theoretical concepts and their practical applications using Python. 

## Root-Finding Techniques

### Theory

Root-finding algorithms are numerical methods for estimating solutions to an equation $f(x) = 0$, where $f(x)$ is a continuous function. Common methods include:

1. **Bisection Method**: This is a simple and robust method based on repeatedly bisecting an interval and choosing a subinterval in which a root exists. It requires a continuous function and an initial interval $[a, b]$ where $f(a)$ and $f(b)$ have opposite signs.

2. **Newton's Method**: An iterative approach that uses the first derivative of $f$ to approximate roots. Starting from an initial guess $x_0$, a sequence is defined by:
   $$ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} $$
   This method converges rapidly near roots but may fail if the derivative is zero or undefined.

3. **Secant Method**: Similar to Newton’s method but does not require the computation of derivatives. It approximates the derivative using a difference quotient:
   $$ x_{n+1} = x_n - f(x_n) \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})} $$

### Examples

Here is a basic implementation of the bisection method in Python:

```python
def bisection_method(f, a, b, tol=1e-10, max_iter=1000):
    if f(a) * f(b) > 0:
        raise ValueError("Function values at the endpoints must be of opposite sign.")
    
    for _ in range(max_iter):
        mid = (a + b) / 2.0
        if abs(f(mid)) < tol or (b - a) / 2.0 < tol:
            return mid
        elif f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    return mid

# Real Application Example
import math

# Finding root of f(x) = x^2 - 4, which is 2
root = bisection_method(lambda x: x**2 - 4, 1, 3)
print(f"Approximate Root: {root}")
```

## Numerical Integration

### Theory

Numerical integration involves approximating the integral of a function. Two common methods include:

1. **Trapezoidal Rule**: Approximates the region under the graph of a function as a series of trapezoids. 
   $$ \int_a^b f(x) \, dx \approx \frac{b-a}{2} (f(a) + f(b)) $$

2. **Simpson's Rule**: Uses parabolic segments instead of linear ones for better accuracy.
   $$ \int_a^b f(x) \, dx \approx \frac{b-a}{6} (f(a) + 4f((a+b)/2) + f(b)) $$

### Examples

Below is an implementation of the trapezoidal rule using Python:

```python
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return s * h

# Real Application Example
import numpy as np

# Integrating f(x) = x^2 from 0 to 1
integral_value = trapezoidal_rule(lambda x: x**2, 0, 1, 1000)
print(f"Approximate Integral: {integral_value}")
```

## Numerical Differentiation

### Theory

Numerical differentiation concerns the approximation of derivatives of functions. The simplest approach is to use finite differences:

1. **Forward Difference**: 
   $$ f'(x) \approx \frac{f(x+h) - f(x)}{h} $$

2. **Central Difference**: 
   $$ f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} $$

These methods are highly sensitive to the choice of $h$.

### Examples

Implementing forward difference in Python:

```python
def forward_difference(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

# Real Application Example
# Derivative of f(x) = sin(x) at x = 0
derivative = forward_difference(np.sin, 0)
print(f"Approximate Derivative: {derivative}")
```

## Error Analysis

### Theory

Error analysis is vital in numerical methods to assess the accuracy and stability of approximations. Key components include:

1. **Truncation Error**: The difference between the exact mathematical solution and the approximation derived from a numerical method. Expressed typically in terms of the step size, like $O(h)$.

2. **Round-off Error**: Arises due to the finite precision of numerical computation, often from floating-point arithmetic.

Understanding and minimizing these errors are crucial for reliable numerical computation.

### Examples

Let's analyze the error in numerical integration using Python:

```python
def error_analysis_trapezoidal(actual, estimated):
    return abs(actual - estimated)

# Real Application Example
# Actual value of integral of f(x) = x^2 from 0 to 1 is 1/3
actual_value = 1/3
estimated_value = trapezoidal_rule(lambda x: x**2, 0, 1, 1000)
error = error_analysis_trapezoidal(actual_value, estimated_value)
print(f"Error in Numerical Integration: {error}")
```

## Adaptive Algorithms

### Theory

Adaptive algorithms dynamically adjust parameters (like step sizes in integrations) to optimize computational efforts while balancing accuracy. 

1. **Adaptive integration**: Adjusts the step size according to the function's behavior, refining areas needing more precision. 

2. **Adaptive step size in ODE solvers**: Chooses step sizes based on error estimates at each step, as seen in Runge-Kutta Fehlberg methods.

These methods improve both efficiency and accuracy.

### Examples

Here we explore how to use an adaptive method with the integral calculation:

```python
from scipy.integrate import quad

# Real Application Example
# Integrating using an adaptive method for f(x) = x^2
integral_value, error_estimate = quad(lambda x: x**2, 0, 1)
print(f"Adaptive Integral Result: {integral_value} with error estimate {error_estimate}")
```

Through this notebook, you've encountered some of the foundational aspects of numerical methods, providing you with the analytical understanding and computational tools necessary to apply these techniques effectively in various scientific and engineering domains.