```markdown
# Stochastic Processes

## Introduction

Stochastic processes are mathematical objects used to describe systems that evolve over time in a way that incorporates randomness. They are foundational in various fields such as finance, economics, physics, and engineering, offering a framework to model the seemingly random dynamics occurring in these fields. This notebook explores several key types of stochastic processes, examining their underlying mathematics, coding them in Python, and exploring their real-world applications.

## Markov Chains

### Theory

A Markov Chain is a stochastic process characterized by the Markov property, which asserts that the future state of the process depends only on the present state and not on the sequence of events that preceded it. Formally, if \(X_t\) represents the state of the process at time \(t\), a Markov Chain must satisfy the condition:

\[ 
P(X_{t+1} = x | X_t = x_t, X_{t-1} = x_{t-1}, \ldots, X_0 = x_0) = P(X_{t+1} = x | X_t = x_t)
\]

A Markov Chain is fully defined by its state space, the transition probabilities between states (often presented in a transition matrix), and its initial state distribution.

### Examples

Let's model a simple Markov Chain in Python:

```python
import numpy as np

# Define state space and transition matrix
states = ['Rainy', 'Sunny']
transition_matrix = [[0.6, 0.4],   # Probabilities of transitioning from Rainy
                     [0.2, 0.8]]   # Probabilities of transitioning from Sunny

# Initial state probabilities
initial_state = [0.5, 0.5]

def next_state(current_state, transition_matrix):
    return np.random.choice(states, replace=True, p=transition_matrix[current_state])

# Simulating Markov Chain
num_steps = 10
current_state = np.random.choice(len(states), p=initial_state)
chain = [states[current_state]]

for _ in range(num_steps):
    current_state = next_state(current_state, transition_matrix)
    chain.append(states[current_state])

print("Sequence of states: ", chain)
```

#### Real Applications:
- **Weather Modeling**: Often used to predict weather conditions where future weather depends only on the current state.
- **Financial Markets**: Used for pricing derivatives where the price's future depends only on its present state.

## Poisson Processes

### Theory

A Poisson process is a stochastic process that models a series of events occurring randomly over time. The defining feature of a Poisson process is that the events occur with a known average rate \(\lambda\) and are independent of the time since the last event. Mathematically, the probability of observing \(k\) events in a time interval \(t\) is given by the Poisson distribution:

\[ 
P(N(t) = k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}
\]

where \(N(t)\) represents the number of events that have occurred by time \(t\).

### Examples

In Python, simulating a Poisson process can be done by determining the time at which each event occurs:

```python
import numpy as np

# Average rate of events
lambda_rate = 3

# Simulate the time between events
inter_event_times = np.random.exponential(1/lambda_rate, 100)
event_times = np.cumsum(inter_event_times)

import matplotlib.pyplot as plt
plt.hist(event_times, bins=10, edgecolor='black', alpha=0.7)
plt.title('Histogram of Event Times')
plt.show()
```

#### Real Applications:
- **Queueing Theory**: Models customer arrivals at a service center.
- **Telecommunications**: Used in modeling the arrival of packets/data in networks.

## Brownian Motion

### Theory

Brownian Motion, also known as Wiener process, is a continuous-time stochastic process used to model random movements such as the path of a molecule in a liquid. It serves as the basis for the mathematical filtration in stochastic calculus. It has the properties that \(W(0) = 0\), has continuous paths, independent increments, and normally distributed increments with mean zero.

Mathematically, for any \(0 \leq s < t\), the increment \(W(t) - W(s) \sim \mathcal{N}(0, t-s)\).

### Examples

Simulating a Brownian Motion trajectory in Python involves generating normally distributed increments:

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0  # total time
N = 1000  # number of steps
dt = T/N  # time increment
t = np.linspace(0, T, N)
dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=N)  # Brownian increments
W = np.cumsum(dW)  # cumulative sum gives Brownian path

plt.plot(t, W)
plt.title('Brownian Motion Path')
plt.show()
```

#### Real Applications:
- **Stock Prices**: Used in the Black-Scholes model for option pricing.
- **Physics**: Describes particle diffusion processes.

## Stochastic Differential Equations

### Theory

Stochastic Differential Equations (SDEs) are differential equations in which one or more of the terms is a stochastic process, most commonly driven by Brownian motion. A standard form SDE is given by:

\[ 
dX_t = \mu(X_t, t)dt + \sigma(X_t, t)dW_t
\]

where \(dX_t\) is the infinitesimal change in the process, \(\mu(X_t, t)\) is the drift coefficient, and \(\sigma(X_t, t)\) is the diffusion coefficient, with \(dW_t\) representing the stochastic innovation.

### Examples

Numerical solutions to SDEs can be approximated using methods akin to the Euler method in deterministic calculus:

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu, sigma = 0.1, 0.3
T, N = 1.0, 1000
dt = T/N
t = np.linspace(0, T, N)
X = np.zeros(N)
X[0] = 1  # initial value

dW = np.random.normal(0, np.sqrt(dt), size=N)
for i in range(1, N):
    X[i] = X[i-1] + mu*X[i-1]*dt + sigma*X[i-1]*dW[i-1]

plt.plot(t, X)
plt.title('Simulation of a Geometric Brownian Motion (SDE)')
plt.show()
```

#### Real Applications:
- **Finance**: Geometric Brownian Motion for modeling stock prices.
- **Biology**: Describes population dynamics under random environmental influences.

## Applications of Stochastic Models

### Theory

The practical applications of stochastic models are vast. The ability to incorporate inherent randomness and time-dependent variability is crucial across various fields. Stochastic models enable predictions and decisions in environments where uncertainty is prevalent.

### Examples

Here, we illustrate the use of stochastic models in real-world applications:

- **Finance**: The Black-Scholes model, a cornerstone of financial engineering, utilizes stochastic calculus to price options by modeling stock price movements as a stochastic process.
  
  ```python
  from scipy.stats import norm
  import numpy as np

  # Black-Scholes parameters
  S = 100  # asset price
  K = 100  # option strike price
  T = 1    # time to maturity
  r = 0.05 # risk-free rate
  sigma = 0.2 # volatility

  # Black-Scholes formula for a European call option
  def black_scholes(S, K, T, r, sigma):
      d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
      d2 = d1 - sigma * np.sqrt(T)
      call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
      return call_price

  call_option_price = black_scholes(S, K, T, r, sigma)
  print(f"European call option price: {call_option_price:.2f}")
  ```

- **Biological Systems**: Stochastic processes model gene expressions that inherently bear random fluctuations.

- **Queueing Theory in Operations Management**: Stochastic models accurately forecast customer service length and wait times.

Each application relies on the core idea of integrating prospective randomness into system dynamics, providing better comprehension and more precise forecasts in uncertain environments.
```
