```markdown
# General Probability

## Introduction

Probability is a fundamental concept in statistics, mathematics, and many science disciplines. It provides a quantitative description of the likely occurrence of an event. Probability theory forms the backbone of statistical inference and is widely applied in fields such as finance, healthcare, and engineering. This notebook will guide you through essential concepts of probability, providing theoretical insights, Python implementations, and real-world applications.

## Basic Probability Rules

### Theory

Probability, in its essence, is the measure of the likelihood that an event will occur. It ranges from 0 (the event will not occur) to 1 (the event will occur). Basic probability rules form the foundation for more complex probability concepts. These include:

1. **Addition Rule**: For any two mutually exclusive events \(A\) and \(B\),
   \[
   P(A \cup B) = P(A) + P(B)
   \]
   
2. **Multiplication Rule**: For two independent events \(A\) and \(B\),
   \[
   P(A \cap B) = P(A) \times P(B)
   \]

3. **Complementary Rule**: The probability of an event not occurring is:
   \[
   P(A^c) = 1 - P(A)
   \]

4. **Total Probability Rule**: For disjoint events \(B_1, B_2, \ldots, B_n\) covering all possible outcomes,
   \[
   P(A) = \sum_{i=1}^n P(A \cap B_i)
   \]

### Examples

Let's apply these rules using Python.

```python
# Import necessary library
from itertools import product

# Define basics
outcome_space = list(product(['H', 'T'], repeat=3))  # Example: coin tossing thrice

# Probability of getting at least one head
event_A = [outcome for outcome in outcome_space if 'H' in outcome]
P_A = len(event_A) / len(outcome_space)
print("Probability of getting at least one head:", P_A)

# Probability of getting all heads (independent events)
P_B = (1/2) * (1/2) * (1/2)
print("Probability of getting all heads:", P_B)

# Complementary Rule - Probability of not getting all heads
P_not_B = 1 - P_B
print("Probability of not getting all heads:", P_not_B)
```

## Conditional Probability

### Theory

Conditional probability is the probability of an event occurring given that another event has occurred. The conditional probability of \(A\) given \(B\) is denoted by \(P(A|B)\) and is defined as:

\[
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad \text{provided } P(B) > 0
\]

Bayes' theorem, a fundamental theorem in probability, relates conditional probabilities and is stated as:

\[
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

### Examples

To illustrate conditional probability, consider a scenario of drawing cards from a deck.

```python
# Probability of drawing an Ace given the card is a face card
# Considering a simplified deck of example

total_face_cards = 12
total_aces = 4
total_cards = 52

# Probability of a card being a face card
P_face = total_face_cards / total_cards

# Conditional probability of it being an Ace if it's a face card
# Assumption: Only considering face cards that aren't aces for this illustration
P_ace_given_face = 0

print("Probability of drawing an Ace given the card is a face card:", P_ace_given_face)
```

## Random Variables

### Theory

A random variable is a numerical description of the outcome of an experiment. It can be discrete or continuous. A discrete random variable has a finite or countably infinite set of values, whereas a continuous random variable has infinitely many possible values.

1. **Discrete Random Variable**: Characterized by a probability mass function (PMF) \(P(X = x_i)\).

2. **Continuous Random Variable**: Characterized by a probability density function (PDF) that integrates to 1 over the space for possible values of the random variable.

### Examples

Consider a die roll, which is a discrete random variable example:

```python
import numpy as np

# Situation: Rolling a fair six-sided die
dice_sides = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6

# PMF display
import matplotlib.pyplot as plt

plt.stem(dice_sides, probabilities)
plt.title('Probability Mass Function of a Fair Die')
plt.xlabel('Dice Sides')
plt.ylabel('Probability')
plt.show()
```

## Probability Distributions

### Theory

Probability Distribution is a function that describes the likelihood of obtaining the possible values that a random variable can assume. Two key types are:

1. **Binomial Distribution**: Discrete distribution with parameters \(n\) and \(p\), the number of trials and success probability, respectively.

2. **Normal Distribution**: Continuous distribution with parameters \(\mu\) (mean) and \(\sigma^2\) (variance), often used due to the Central Limit Theorem.

### Examples

Binomial and Normal distribution examples in Python:

```python
# Binomial Distribution
from scipy.stats import binom

# Parameters
n = 10  # number of trials
p = 0.5  # probability of success on each trial

# Creating the binomial random variable X
X = binom(n, p)

# Probability of getting exactly 5 successes
prob_5 = X.pmf(5)
print("Probability of getting exactly 5 successes in 10 trials:", prob_5)

# Normal Distribution
from scipy.stats import norm

# Normal distribution with mean 0, std deviation 1
mu, sigma = 0, 1

# Probability of X being between -1 and 1
prob_between = norm.cdf(1, mu, sigma) - norm.cdf(-1, mu, sigma)
print("Probability of X being between -1 and 1 in a standard normal distribution:", prob_between)
```

## Law of Large Numbers

### Theory

The Law of Large Numbers (LLN) states that as the number of trials of a random process increases, the average of the results obtained from the process will almost surely converge to the expected value. It is divided into the **Weak Law** and the **Strong Law**, focusing on convergence in probability and almost sure convergence, respectively.

### Examples

Consider coin tosses to observe LLN:

```python
import numpy as np

np.random.seed(42)
coin_tosses = np.random.choice([0, 1], size=10000)  # Simulating 0 (Tails) or 1 (Heads)

# Cumulative averages
cumulative_averages = np.cumsum(coin_tosses) / (np.arange(1, 10001))

# Plotting
plt.plot(cumulative_averages)
plt.axhline(y=0.5, color='r', linestyle='-')
plt.title('Law of Large Numbers: Coin Toss')
plt.xlabel('Number of Tosses')
plt.ylabel('Cumulative Average of Heads')
plt.show()
```

The red line represents the theoretical probability of obtaining heads, demonstrating convergence as the number of trials increases.

```