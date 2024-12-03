```markdown
# Applications to Biology and Epidemiology

## Introduction
Biology and epidemiology are fields that are inherently data-rich and quantitatively intensive. Mathematical models and computational approaches have become crucial in understanding complex biological systems and disease dynamics. In this notebook, we will explore how mathematics and computer programming intersect with biology and epidemiology through the following areas: Population Dynamics, Disease Modeling, Bioinformatics Algorithms, Statistical Genetics, and Computational Neuroscience.

## Population Dynamics

### Theory
Population dynamics is the study of how populations of organisms change over time and space. The foundational models in population dynamics include the exponential and logistic growth models, which are mathematically represented as follows:

1. **Exponential Growth Model**

   The exponential growth model assumes unlimited resources and is represented by:
   \[
   \frac{dN}{dt} = rN
   \]
   where:
   - \( N \) is the population size,
   - \( r \) is the intrinsic growth rate.

   The solution to this differential equation is:
   \[
   N(t) = N_0 e^{rt}
   \]
   where \( N_0 \) is the initial population size.

2. **Logistic Growth Model**

   The logistic growth model accounts for resource limitations:
   \[
   \frac{dN}{dt} = rN \left(1 - \frac{N}{K}\right)
   \]
   where:
   - \( K \) is the carrying capacity of the environment.

   The solution is given by:
   \[
   N(t) = \frac{K}{1 + \left(\frac{K-N_0}{N_0}\right) e^{-rt}}
   \]

### Examples
Let's simulate population growth using these models with Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters for simulation
r = 0.1  # intrinsic growth rate
K = 1000  # carrying capacity
N0 = 10  # initial population
t = np.linspace(0, 100, num=1000)

# Exponential Growth
N_exp = N0 * np.exp(r * t)

# Logistic Growth
N_log = (K * N0 * np.exp(r * t)) / (K + N0 * (np.exp(r * t) - 1))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, N_exp, label='Exponential Growth')
plt.plot(t, N_log, label='Logistic Growth', linestyle='--')
plt.title('Population Dynamics')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.legend()
plt.grid(True)
plt.show()
```

## Disease Modeling

### Theory
In epidemiology, disease modeling is essential for understanding the spread of infectious diseases. One classical model is the SIR model, which divides the population into three compartments: Susceptible (S), Infected (I), and Recovered (R). The model is defined by the following set of differential equations:

\[
\frac{dS}{dt} = -\beta SI
\]
\[
\frac{dI}{dt} = \beta SI - \gamma I
\]
\[
\frac{dR}{dt} = \gamma I
\]

where:
- \( \beta \) is the transmission rate,
- \( \gamma \) is the recovery rate.

### Examples
Here's a simulation of the SIR model using Python.

```python
from scipy.integrate import odeint

# Parameters
beta = 0.3  # transmission rate
gamma = 0.1  # recovery rate

# Initial conditions
S0 = 0.99  # initial susceptible proportion
I0 = 0.01  # initial infected proportion
R0 = 0.0   # initial recovered proportion
y0 = S0, I0, R0

# Time points for simulation
t = np.linspace(0, 160, 160)

# SIR model differential equations
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return dS_dt, dI_dt, dR_dt

# Integrate the SIR equations over the time grid
solution = odeint(sir_model, y0, t, args=(beta, gamma))
S, I, R = solution.T

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.title('SIR Model Simulation')
plt.xlabel('Time')
plt.ylabel('Proportion of Population')
plt.legend()
plt.grid(True)
plt.show()
```

## Bioinformatics Algorithms

### Theory
Bioinformatics involves the application of algorithms to biological data, particularly in the context of genomics and proteomics. One common algorithmic challenge in bioinformatics is sequence alignment, crucial for comparing DNA, RNA, or protein sequences.

**Needleman-Wunsch Algorithm**: A dynamic programming algorithm for global sequence alignment. It constructs a matrix to find the optimal alignment between sequences.

### Examples
Implementing a simple version of the Needleman-Wunsch algorithm in Python.

```python
import numpy as np

def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-1):
    # Initialize scoring matrix
    n, m = len(seq1), len(seq2)
    score_matrix = np.zeros((n+1, m+1), dtype=int)

    # Initialize first row and column
    for i in range(n+1):
        score_matrix[i][0] = i * gap
    for j in range(m+1):
        score_matrix[0][j] = j * gap

    # Fill the scoring matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            match_score = score_matrix[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            delete = score_matrix[i-1][j] + gap
            insert = score_matrix[i][j-1] + gap
            score_matrix[i][j] = max(match_score, delete, insert)

    # Traceback
    i, j = n, m
    aligned_seq1, aligned_seq2 = "", ""
    while i > 0 and j > 0:
        current_score = score_matrix[i][j]
        if current_score == score_matrix[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch):
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            i -= 1
            j -= 1
        elif current_score == score_matrix[i-1][j] + gap:
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = "-" + aligned_seq2
            i -= 1
        else:
            aligned_seq1 = "-" + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            j -= 1

    return aligned_seq1, aligned_seq2

# Example sequences
seq1 = "GATTACA"
seq2 = "GCATGCU"

# Align sequences
alignment = needleman_wunsch(seq1, seq2)
print("Aligned Sequences:")
print(alignment[0])
print(alignment[1])
```

## Statistical Genetics

### Theory
Statistical genetics involves the study of the genetic basis of complex traits using statistical methods. A common task is Genome-Wide Association Studies (GWAS), which identify genetic variants associated with traits. GWAS involves testing the association between a large number of genetic markers (SNPs) and phenotypes.

The statistical model often used is a simple linear regression:
\[ y = X\beta + \epsilon \]
where \( y \) is the phenotype, \( X \) is the genotype matrix, \( \beta \) is the vector of effect sizes, and \( \epsilon \) is the error term.

### Examples
Performing a simple linear regression in Python to mimic a GWAS analysis.

```python
import pandas as pd
import statsmodels.api as sm

# Simulated data
np.random.seed(0)
n_individuals = 100
n_snps = 1000
X = np.random.choice([0, 1, 2], size=(n_individuals, n_snps))
effect_sizes = np.random.randn(n_snps)
y = X @ effect_sizes + np.random.randn(n_individuals) * 0.5

# Selecting a single SNP for demonstration
snp_index = 0
X_snp = X[:, snp_index]

# Adding constant to the model
X_snp = sm.add_constant(X_snp)

# Fitting linear regression
model = sm.OLS(y, X_snp)
results = model.fit()

print("SNP Index:", snp_index)
print(results.summary())
```

## Computational Neuroscience

### Theory
Computational neuroscience seeks to understand how the brain computes information. A fundamental model is the Hodgkin-Huxley model, which describes how action potentials in neurons are initiated and propagated. It uses differential equations to model the ionic currents across the neuron membrane.

The Hodgkin-Huxley model involves:
\[ C_m \frac{dV}{dt} = I_{ext} - (g_{Na} m^3 h (V - V_{Na}) + g_{K} n^4 (V - V_{K}) + g_L (V - V_L)) \]

### Examples
Simulating the Hodgkin-Huxley model for action potential generation.

```python
import numpy as np
import matplotlib.pyplot as plt

# Model parameters
Cm = 1.0  # membrane capacitance, in uF/cm^2
g_Na = 120.0  # maximum conducances, in mS/cm^2
g_K = 36.0
g_L = 0.3
E_Na = 50.0  # Nernst reversal potentials, in mV
E_K = -77.0
E_L = -54.387
I_ext = 10.0  # external current, in uA/cm^2

# Time parameters
dt = 0.01  # time step
t = np.arange(0.0, 50.0, dt)

# Variables
V = np.zeros_like(t)  # membrane potential

# Initial conditions
V[0] = -65  # initial membrane potential

# Hodgkin-Huxley equations
def alpha_n(v): return 0.01*(v+55)/(1-np.exp(-(v+55)/10))
def beta_n(v): return 0.125*np.exp(-(v+65)/80)
def alpha_m(v): return 0.1*(v+40)/(1-np.exp(-(v+40)/10))
def beta_m(v): return 4.0*np.exp(-(v+65)/18)
def alpha_h(v): return 0.07*np.exp(-(v+65)/20)
def beta_h(v): return 1/(1+np.exp(-(v+35)/10))

# Initialize gating variables
n = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))
m = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
h = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))

for i in range(1, len(t)):
    # ionic currents
    I_Na = g_Na * m**3 * h * (V[i-1] - E_Na)
    I_K = g_K * n**4 * (V[i-1] - E_K)
    I_L = g_L * (V[i-1] - E_L)

    # Membrane potential update
    V[i] = V[i-1] + dt * (I_ext - (I_Na + I_K + I_L)) / Cm

    # Gating variables update
    n = n + dt * (alpha_n(V[i-1]) * (1.0 - n) - beta_n(V[i-1]) * n)
    m = m + dt * (alpha_m(V[i-1]) * (1.0 - m) - beta_m(V[i-1]) * m)
    h = h + dt * (alpha_h(V[i-1]) * (1.0 - h) - beta_h(V[i-1]) * h)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, V)
plt.title('Hodgkin-Huxley Neuron Model')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.grid(True)
plt.show()
```
```