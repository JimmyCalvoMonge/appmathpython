# Numerical Linear Algebra

## Introduction

Numerical Linear Algebra is a vital field in computational science that studies algorithms for performing linear algebra computations, often with a focus on efficiency and numerical stability. These techniques are essential when dealing with large datasets or complex mathematical models across a wide array of applications, from machine learning to scientific simulations.

In this notebook, we will delve into five key areas of Numerical Linear Algebra: Matrix Decompositions, Iterative Solvers, Eigenvalue Problems, Sparse Matrix Techniques, and Stability in Computations. Each section will offer comprehensive mathematical explanations, Python code examples, and real-world applications.

## Matrix Decompositions

### Theory

Matrix decomposition is a fundamental numerical strategy that expresses a matrix as a product of simpler matrices, enabling easy matrix operations like solving systems of linear equations. The most common decompositions include:

1. **LU Decomposition**: Splits a matrix \( A \) into a lower triangular matrix \( L \) and an upper triangular matrix \( U \). It's primarily used for solving linear equations, matrix inversion, and determining matrix determinants.

   \[
   A = LU
   \]

2. **QR Decomposition**: Decomposes a matrix into an orthogonal matrix \( Q \) and an upper triangular matrix \( R \). It's crucial in solving linear least squares problems and computing eigenvalues.

   \[
   A = QR
   \]

3. **Singular Value Decomposition (SVD)**: Generalizes the diagonalization process to any \( m \times n \) matrix, breaking it down into two orthogonal matrices \( U \) and \( V^T \), and a diagonal matrix \( \Sigma \).

   \[
   A = U \Sigma V^T
   \]

### Examples

Let's explore these decompositions using Python:

```python
import numpy as np
from scipy.linalg import lu, qr, svd

# Sample Matrix
A = np.array([[4, 3], [6, 3]])

# LU Decomposition
P, L, U = lu(A)
print("L:\n", L)
print("U:\n", U)

# QR Decomposition
Q, R = qr(A)
print("Q:\n", Q)
print("R:\n", R)

# SVD
U, S, VT = svd(A)
print("U:\n", U)
print("Singular values:\n", S)
print("VT:\n", VT)
```

### Real Applications

- **LU Decomposition** is widely used in optimizing processes, such as in logistic regression and other linear modeling techniques.
- **QR Decomposition** finds application in numerical solutions of linear equations and eigenvalue computations.
- **SVD** is extensively used in data science for dimensionality reduction and noise reduction in signal processing.

## Iterative Solvers

### Theory

Iterative solvers are techniques for finding numerical solutions of linear systems, especially useful when dealing with large, sparse matrices. Unlike direct solvers, iterative solvers improve an initial guess to generate a sequence of approximations converging toward the solution. Key methods include:

- **Jacobi Method** and **Gauss-Seidel Method**: Simple yet powerful, suitable for diagonally dominant matrices.
- **Conjugate Gradient Method**: Specifically designed for solving large systems with a symmetric positive definite matrix.

### Examples

Below is an example of the Jacobi Method:

```python
def jacobi(A, b, x0, tol=1e-10, max_iterations=100):
    D = np.diag(A)
    R = A - np.diagflat(D)

    x = x0
    for i in range(max_iterations):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return x

# Example usage
A = np.array([[10, -1, 2, 0],
              [-1, 11, -1, 3],
              [2, -1, 10, -1],
              [0, 3, -1, 8]])
b = np.array([6, 25, -11, 15])
x0 = np.zeros_like(b)

solution = jacobi(A, b, x0)
print("Solution:", solution)
```

### Real Applications

- **Iterative Solvers** are essential in large-scale simulations, such as finite element analysis used in engineering for calculating stresses in structures.

## Eigenvalue Problems

### Theory

Eigenvalue problems involve determining the characteristic values (eigenvalues) and characteristic vectors (eigenvectors) of a matrix. These concepts are crucial in understanding matrix transformations and are widely applied in qualitative dynamics analysis, quantum mechanics, and stability analysis.

For a square matrix \( A \) and a non-zero vector \( v \):

\[
Av = \lambda v
\]

where \( \lambda \) are the eigenvalues and \( v \) are the eigenvectors.

### Examples

Finding eigenvalues and eigenvectors in Python using NumPy:

```python
# Define a matrix
A = np.array([[3, 1], [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

### Real Applications

- Eigenvalues and eigenvectors are instrumental in **Principal Component Analysis (PCA)** for reducing data dimensions.
- They play a role in stability analyses in control systems.

## Sparse Matrix Techniques

### Theory

Sparse matrices are matrices predominantly filled with zeroes. Utilizing their properties can significantly save memory and computational time. Key techniques include storing matrices using formats like Compressed Sparse Row (CSR) and using algorithms that exploit sparsity to speed up computations.

### Examples

Below is an example of creating and manipulating a sparse matrix using SciPy:

```python
from scipy.sparse import csr_matrix

# Create a sparse matrix
A = csr_matrix([[1, 0, 0], [0, 0, 2], [0, 3, 0]])
print("Sparse matrix:\n", A)

# Perform matrix operations
B = A @ A.T  # Matrix multiplication
print("Result of multiplication:\n", B.toarray())
```

### Real Applications

- Sparse matrix techniques are widely applied in solving **large graph algorithms**, such as in social network analysis.
- They play a crucial role in **compressed sensing** and **large-scale machine learning models**.

## Stability in Computations

### Theory

Stability in numerical computations addresses how errors propagate through an algorithm. A stable algorithm produces results that have bounded errors relative to the data input, even when subjected to small perturbations in input data or arithmetic round-off.

When dealing with stability concerns, it is crucial to analyze:

- **Condition Number**: A measure of a matrix's sensitivity to input changes.
- **Error Propagation**: Understanding how initial errors affect the final results is key to selecting numerical methods.

### Examples

Assessing the condition number using NumPy:

```python
# Create a matrix
A = np.array([[1, 2], [3, 4]])

# Calculate the condition number
cond_number = np.linalg.cond(A)
print("Condition Number:", cond_number)
```

### Real Applications

- In numerical weather prediction, ensuring stability helps in achieving reliable forecasts.
- Stability analysis is fundamental in designing robust **financial modeling algorithms**.

With this overview, we have explored the foundational concepts and applications of Numerical Linear Algebra, emphasizing both the theoretical backbone and practical examples. These techniques form the cornerstone of many scientific and engineering computations in diverse disciplines.