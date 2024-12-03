# Operations Research

## Introduction

Operations Research (OR) is a discipline that applies advanced analytical methods to help make better decisions. It uses mathematical models, statistical analyses, and algorithms to solve complex problems often encountered in business, engineering, and other fields. The core goal of OR is to maximize efficiency, minimize costs, and optimize resource use.

This notebook will cover key topics in Operations Research, including Optimization Models, Queuing Theory, Simulation Methods, Supply Chain Optimization, and Decision Analysis.

## Optimization Models

### Theory

Optimization models are mathematical representations crafted to find the best solution, usually the maximization or minimization of some objective function, within given constraints. These models can be classified into various types, such as:

- **Linear Programming (LP):** where the objective function and constraints are linear.
- **Integer Programming (IP):** where solutions are required to be whole numbers.
- **Non-linear Programming (NLP):** where the objective function or constraints are non-linear.

Mathematically, a basic linear programming problem can be represented as:
  
\[ 
\text{Maximize } \quad c^T x 
\]

\[ 
\text{Subject to } \quad Ax \leq b 
\]

\[ 
x \geq 0
\]

Where \( x \) is the vector of decision variables, \( c \) is the coefficient vector, and \( A \) and \( b \) represent the constraint coefficients and limits.

### Examples

``` python
from scipy.optimize import linprog

# Coefficients of the objective function
c = [-1, -2]  # Maximizing x + 2y, negate for linprog

# Coefficients of the inequality constraints
A = [[2, 1],
     [1, 2],
     [1, -1]]

# Right-hand side values of the inequalities
b = [20, 20, 0]

# Using scipy's linprog function to solve the LP problem
result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None))

print("Optimal value:", -result.fun)
print("Optimal solution:", result.x)
```

In this example, we employ the `linprog` method from the SciPy library to solve a linear programming problem. The method seeks to optimize the objective function subject to linear inequalities.

#### Real Applications

Optimization models are used extensively in supply chain management, finance, and production planning to determine the best allocation of resources, optimal scheduling, and cost minimization.

## Queuing Theory

### Theory

Queuing Theory is the mathematical study of waiting lines or queues. It focuses on the analysis and modeling of the process where customers wait for service. Key variables include:

- **Arrival Rate (\(\lambda\)):** the rate at which customers arrive into the system.
- **Service Rate (\(\mu\)):** the rate at which customers are served.

The basic queue model, M/M/1 (single server), assumes Poisson arrivals, exponential service times, and a first-come-first-serve discipline:

\[
L = \frac{\lambda}{\mu - \lambda}
\]

Where \( L \) is the average number of customers in the system.

### Examples

``` python
import math

# Parameters
arrival_rate = 2  # Customers per hour
service_rate = 3  # Customers served per hour

# Average number of customers in the system
L = arrival_rate / (service_rate - arrival_rate)

print("Average number of customers in the system:", L)
```

This simplistic implementation calculates the average number of customers in the system for a given arrival and service rate using the M/M/1 model.

#### Real Applications

Queuing theory is widely applied in sectors such as telecommunications, traffic engineering, and service centers to improve efficiency and reduce waiting times.

## Simulation Methods

### Theory

Simulation Methods involve creating mathematical models to replicate real-world processes, providing insights into system behavior under varying conditions. They are beneficial when analytical solutions are impracticable.

Monte Carlo Simulations are a typical example that involves using random sampling to compute results and understand the impact of risk and uncertainty in prediction and forecasting models.

### Examples

``` python
import numpy as np

# Simulate a simple process, e.g., the outcome of a die roll
n_simulations = 10000
dice_rolls = np.random.randint(1, 7, n_simulations)

# Calculate probabilities
prob_of_six = np.sum(dice_rolls == 6) / n_simulations

print("Probability of rolling a six:", prob_of_six)
```

In this code snippet, a Monte Carlo simulation is used to estimate the probability of rolling a six on a fair die.

#### Real Applications

Simulation methods are extensively used in financial modeling, manufacturing, project management, and risk analysis to anticipate outcomes and inform strategic decisions.

## Supply Chain Optimization

### Theory

Supply Chain Optimization seeks to improve efficiency and effectiveness across the entire supply chain process, optimizing inventory levels, distribution, and transportation. Models in this area often include multi-echelon optimization and network design.

A typical optimization problem might involve minimizing costs associated with the supply chain network while meeting demand requirements efficiently:

\[
\text{Minimize } \quad \sum \text{Variable and fixed costs}
\]

### Examples

``` python
# Example not provided as involves complex supply chain libraries or frameworks
# such as Google OR-Tools, PuLP, or specialized platforms.
```

Operation and optimization of complex supply chains often require advanced modeling and commercial solvers.

#### Real Applications

This approach is utilized by retailers, wholesalers, and manufacturers alike to streamline processes, reduce costs, and enhance service quality across the entire supply chain.

## Decision Analysis

### Theory

Decision Analysis encompasses skills, tools, and processes to make informed decisions. This often involves structuring decision problems, assessing risks, and evaluating multiple alternatives through decision trees or other models.

A Decision Tree visually represents decisions and their possible consequences, including chance event outcomes.

### Examples

``` python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train decision tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Test the model
accuracy = dt.score(X_test, y_test)
print("Accuracy of the Decision Tree:", accuracy)
```

This example shows the use of a Decision Tree for a classification task using the Iris dataset, illustrating choices and their outcomes.

#### Real Applications

Decision Analysis is used in many fields including finance, healthcare, and supply chain management to make critical choices with quantified uncertainties and trade-offs.

By understanding and applying these concepts, practitioners of Operations Research can solve complex problems efficiently and effectively, improving both strategic and operational decision-making.