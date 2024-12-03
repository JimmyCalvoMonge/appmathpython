# Bayesian Analysis

## Introduction

Bayesian analysis is a statistical method that applies probability to statistical problems, using Bayes' Theorem as a foundational principle. It provides a comprehensive framework for statistical inference, where probabilities express a degree of belief in an event. Throughout this notebook, we will explore different components of Bayesian analysis, supported by theoretical explanations and practical examples using Python.

## Bayes' Theorem Basics

### Theory

Bayes' Theorem is a fundamental concept in probability theory and statistics, providing a method to update the probability estimate for a hypothesis as new evidence is available. The theorem is expressed mathematically as:

\[ P(H | E) = \frac{P(E | H) \cdot P(H)}{P(E)} \]

Where:
- \( P(H | E) \) is the posterior probability, the probability of hypothesis \( H \) given the evidence \( E \).
- \( P(E | H) \) is the likelihood, the probability of evidence \( E \) given that hypothesis \( H \) is true.
- \( P(H) \) is the prior probability, the initial probability of hypothesis \( H \).
- \( P(E) \) is the marginal likelihood, the probability of the evidence.

Bayes' Theorem essentially balances the prior belief with the likelihood of the current evidence to produce a posterior belief.

### Examples

Let's consider a simple example of Bayesian analysis:

Suppose a medical test is used to detect a rare disease occurring in 0.1% of the population. The test correctly identifies a diseased individual 99% of the time and correctly identifies a non-diseased individual 95% of the time. What is the probability of having the disease if a person has tested positive?

```python
# Defining the probabilities
P_disease = 0.001  # Prior probability of the disease
P_positive_given_disease = 0.99  # Probability of testing positive given the disease
P_positive_given_no_disease = 0.05  # Probability of testing positive without the disease

# Applying Bayes' theorem
P_no_disease = 1 - P_disease
P_positive = (P_positive_given_disease * P_disease) + (P_positive_given_no_disease * P_no_disease)
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print(f"The probability of having the disease if the test is positive is {P_disease_given_positive:.3f}")
```

## Priors and Posteriors

### Theory

In Bayesian analysis, the prior and posterior distributions are essential. They represent our beliefs about the parameter before and after observing the data, respectively.

- **Prior (\( P(H) \))**: It reflects the initial belief about a parameter's value before any data is observed.
- **Posterior (\( P(H | E) \))**: It updates the prior based on new evidence (data) through Bayes' Theorem.

Selecting appropriate prior distributions and understanding their influence on the posterior is critical in Bayesian analysis, leading to more informed and robust inference.

### Examples

Consider you are estimating the probability of a coin landing heads using Bayesian updating:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Define prior as a uniform distribution
prior_alpha, prior_beta = 1, 1

# Simulate observing 5 heads and 3 tails
observed_heads = 5
observed_tails = 3

# Update priors with observed data
posterior_alpha = prior_alpha + observed_heads
posterior_beta = prior_beta + observed_tails

# Define the posterior distribution
x = np.linspace(0, 1, 100)
prior = beta.pdf(x, prior_alpha, prior_beta)
posterior = beta.pdf(x, posterior_alpha, posterior_beta)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(x, prior, label='Prior', linestyle='--')
plt.plot(x, posterior, label='Posterior')
plt.title('Prior and Posterior Distributions')
plt.xlabel('Probability of Heads')
plt.ylabel('Density')
plt.legend()
plt.show()
```

## Conjugate Priors

### Theory

In Bayesian statistics, a conjugate prior for a likelihood function is a prior that, when combined with a likelihood function, gives a posterior distribution in the same family. This conjugacy simplifies computation greatly, especially in closed-form analytical solutions.

Some common examples include:
- Binomial likelihood with Beta conjugate prior.
- Gaussian likelihood with Gaussian conjugate prior.
- Poisson likelihood with Gamma conjugate prior.

### Examples

Let's see an example with a Gaussian conjugate prior:

```python
from scipy.stats import norm

# Prior parameters
prior_mean = 0
prior_std = 1

# Observations (data)
observations = [1.2, -0.7, 0.3, 0.9, -0.2]

# Updating using conjugate prior
posterior_mean = (prior_mean / prior_std**2 + np.sum(observations)) / (1 / prior_std**2 + len(observations))
posterior_std = np.sqrt(1 / (1 / prior_std**2 + len(observations)))

# Plotting
x = np.linspace(-3, 3, 100)
prior_pdf = norm.pdf(x, prior_mean, prior_std)
posterior_pdf = norm.pdf(x, posterior_mean, posterior_std)

plt.figure(figsize=(8, 4))
plt.plot(x, prior_pdf, label='Prior', linestyle='--')
plt.plot(x, posterior_pdf, label='Posterior')
plt.title('Gaussian Conjugate Prior Example')
plt.xlabel('Mean value')
plt.ylabel('Density')
plt.legend()
plt.show()
```

## Markov Chain Monte Carlo (MCMC)

### Theory

Markov Chain Monte Carlo (MCMC) is a class of algorithms for sampling from probability distributions, commonly used in Bayesian computation. By constructing a Markov chain, whose equilibrium distribution is the target distribution, MCMC allows us to approximate the posterior distribution even when it cannot be analytically derived.

Popular MCMC methods include:
- Metropolis-Hastings Algorithm
- Gibbs Sampling

### Examples

Let's conduct a simple MCMC simulation using the Metropolis algorithm.

```python
import pymc3 as pm

# Define model assuming we are estimating the mean of a distribution
with pm.Model() as model:
    # Prior distribution
    mean = pm.Normal('mean', mu=0, sigma=1)
    
    # Likelihood with observed data
    likelihood = pm.Normal('obs', mu=mean, sigma=1, observed=[1, 2, 3, 2.5, 3.5])
    
    # Inference using Metropolis
    step = pm.Metropolis()
    trace = pm.sample(1000, step=step, random_seed=42)

# Plot the results
pm.traceplot(trace)
plt.show()
```

## Bayesian Model Selection

### Theory

Bayesian model selection involves choosing between models using the Bayesian framework. It often requires the computation of the marginal likelihood or evidence, a challenging task due to the need to integrate over all parameters. However, metrics like the Bayes Factor, which is the ratio of the marginal likelihoods of two models, can be used to perform model comparison objectively.

### Examples

An example can be shown by comparing two models with different priors using PyMC3:

```python
import arviz as az

# Model 1 with prior mean 0
with pm.Model() as model_1:
    mean1 = pm.Normal('mean', mu=0, sigma=1)
    likelihood1 = pm.Normal('obs', mu=mean1, sigma=1, observed=[1, 2, 3, 2.5, 3.5])
    trace1 = pm.sample(1000, random_seed=42)

# Model 2 with prior mean 1
with pm.Model() as model_2:
    mean2 = pm.Normal('mean', mu=1, sigma=1)
    likelihood2 = pm.Normal('obs', mu=mean2, sigma=1, observed=[1, 2, 3, 2.5, 3.5])
    trace2 = pm.sample(1000, random_seed=42)

# Compare models
comp = az.compare({'model_1': trace1, 'model_2': trace2})
az.plot_compare(comp)
plt.show()
```

By the method described above, the model selection is facilitated by computing and comparing the approximated marginal likelihoods, thus providing an informed way to choose the best model based on the data.

This notebook has explored multiple facets of Bayesian Analysis, demonstrating its powerful utility in making probabilistic inferences and driving model selection with real-world data.