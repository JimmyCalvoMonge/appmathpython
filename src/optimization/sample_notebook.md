```markdown
# Optimization in Mathematics and Computer Science

## Introduction

Optimization is a crucial area in both mathematics and computer science dedicated to finding the best solution from all feasible solutions. It involves maximizing or minimizing a function by systematically choosing the values of variables from a permissible set, which is critical in numerous real-world applications.

The purpose of this notebook is to deliver a comprehensive overview of different optimization approaches, explain their theoretical underpinnings, and provide practical demonstrations using Python programming language.

## Linear Programming

### Theory

Linear Programming (LP) is a mathematical technique for determining the best outcome, such as maximum profit or lowest cost, in a mathematical model whose requirements are represented by linear relationships. A linear programming problem is usually expressed in canonical form:

\[
\begin{align*}
\text{Optimize:} & \quad \mathbf{c}^T \mathbf{x} \\
\text{Subject to:} & \quad A \mathbf{x} \leq \mathbf{b}, \\
& \quad \mathbf{x} \geq 0.
\end{align*}
\]

Where:
- \(\mathbf{c}\) represents cost coefficients,
- \(\mathbf{x}\) denotes decision variables,
- \(A\) is a matrix representing equation coefficients,
- \(\mathbf{b}\) is the constraints vector.

### Examples

Let's solve a simple LP problem in Python using the library `scipy.optimize`.

**Problem Statement:**
Maximize \(z = 3x + 2y\) subject to:  
1. \(2x + y \leq 18\)
2. \(2x + 3y \leq 42\)
3. \(3x + y \leq 24\)
4. \(x, y \geq 0\)

```python
from scipy.optimize import linprog

c = [-3, -2]  # Coefficients for the objective function (negated for maximization)
A = [[2, 1], [2, 3], [3, 1]]
b = [18, 42, 24]
x_bounds = (0, None)
y_bounds = (0, None)

result = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds], method='highs')

print('Optimal value:', -result.fun)
print('x values:', result.x)
```

## Nonlinear Programming

### Theory

Nonlinear Programming (NLP) involves optimizing a nonlinear objective function subject to nonlinear constraints. An NLP problem is of the form:

\[
\begin{align*}
\text{minimize:} & \quad f(\mathbf{x}) \\
\text{subject to:} & \quad g_i(\mathbf{x}) \leq b_i, \quad \text{for } i = 1, \ldots, m \\
& \quad h_j(\mathbf{x}) = c_j, \quad \text{for } j = 1, \ldots, p \\
& \quad \mathbf{x} \in \Omega.
\end{align*}
\]

Where \(f, g_i, h_j\) are nonlinear functions.

### Examples

We demonstrate solving an NLP problem using `scipy.optimize.minimize`.

**Problem Statement:**
Minimize \(f(x, y) = x^2 + y^2\) subject to the constraint \(x + y = 1\).

```python
from scipy.optimize import minimize

def objective(x):
    return x[0] ** 2 + x[1] ** 2

def constraint(x):
    return x[0] + x[1] - 1

x0 = [0.5, 0.5]
cons = {'type': 'eq', 'fun': constraint}

solution = minimize(objective, x0, constraints=[cons], method='SLSQP')
print('Optimal solution:', solution.x)
```

## Convex Optimization

### Theory

Convex Optimization is a subfield of optimization where the objective function \((\text{if minimizing})\) or concave \((\text{if maximizing})\) and the feasible region are both convex sets. These types of problems are important because convex problems are often easier to solve than non-convex problems, due to the property that any local minimum is also a global minimum.

### Examples

Suppose we want to minimize a convex quadratic function. We can use libraries like `cvxopt`.

**Problem Statement:**
Minimize \(f(x) = \frac{1}{2}x^T Q x + c^T x\)

```python
from cvxopt import matrix, solvers

Q = matrix([[2.0, 0.0], [0.0, 2.0]])
p = matrix([1.0, 1.0])  # Linear term
G = matrix([[-1.0, 0.0], [0.0, -1.0]])
h = matrix([0.0, 0.0])

sol = solvers.qp(Q, p, G, h)
print('Optimal solution:', sol['x'])
```

## Integer Programming

### Theory

Integer Programming (IP) is optimization where some or all variables are constrained to take on integer values. This type of programming is essential for problems where the solution must be discrete, such as scheduling, allocation, and resource distribution.

### Examples

We utilize `PuLP`, an open-source linear programming library in Python to solve an integer programming problem.

**Problem Statement:**
Maximize \(z = x + 2y\) subject to:
1. \(x + y \leq 5\)
2. \(x, y \in \text{Integers} \geq 0\)

```python
from pulp import LpMaximize, LpProblem, LpVariable

model = LpProblem(name="integer-program", sense=LpMaximize)

x = LpVariable(name="x", lowBound=0, cat='Integer')
y = LpVariable(name="y", lowBound=0, cat='Integer')

model += (x + 2 * y)
model += (x + y <= 5)

model.solve()

print(f"Optimal x: {x.varValue}")
print(f"Optimal y: {y.varValue}")
```

## Gradient-Based Methods

### Theory

Gradient-Based Methods are iterative optimization algorithms leveraging the gradient of the function to find the minimum or maximum. These methods include Gradient Descent and its numerous variants, suitable for large-scale optimization problems.

### Examples

The following demonstrates basic gradient descent for minimizing a simple quadratic function.

**Problem Statement:**
Minimize \(f(x) = x^2\)

```python
def gradient_descent(grad, x0, learning_rate=0.1, tolerance=1e-6, max_iterations=1000):
    x = x0
    for i in range(max_iterations):
        x_new = x - learning_rate * grad(x)
        if abs(x_new - x) < tolerance:
            break
        x = x_new
    return x

gradient = lambda x: 2 * x
optimal_x = gradient_descent(gradient, x0=10.0)
print('Optimal x:', optimal_x)
```

This comprehensive guide provides a foundation for understanding various optimization methods and showcases practical implementations in Python. By exploring these sections, one gains crucial insights into how optimization techniques are employed to solve complex real-world problems.