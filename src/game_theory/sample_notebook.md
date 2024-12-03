# Game Theory

## Introduction

Game Theory is a branch of mathematics that studies strategic interactions among rational decision-makers. It is widely used in several fields including economics, political science, psychology, and computer science, offering a framework for understanding various competitive situations where the outcome for each participant depends on the choices of all involved.

In this notebook, we will explore several key concepts in game theory, including Nash Equilibria, Zero-Sum Games, Cooperative Games, Evolutionary Game Theory, and Applications in Economics. Each section will present theoretical foundations, examples, and practical applications.

## Nash Equilibria

### Theory

Nash Equilibrium is a fundamental concept in game theory, formulated by John Nash. It describes a situation where no player can benefit by changing their strategy unilaterally. In formal terms, a strategy profile is a Nash Equilibrium if, for each player, their choice is optimal given the strategies chosen by all other players. 

Mathematically, for a game represented by a set of players \(N\), strategies \(s_i\) for player \(i\), and a payoff function \(u_i(s)\):

A strategy profile \(s^* = (s_1^*, s_2^*, ..., s_n^*)\) is a Nash Equilibrium if:
\[ u_i(s_i^*, s_{-i}^*) \geq u_i(s_i, s_{-i}^*) \]
for all players \(i\) and all possible strategies \(s_i\).

### Examples

In Python, we can use the library `nashpy` to calculate Nash Equilibria for simple games such as the Prisoner's Dilemma:

```python
import nashpy as nash
import numpy as np

# Define payoff matrices for two players
A = np.array([[3, 3], [0, 6]])
B = np.array([[3, 0], [3, 6]])

# Create the game
prisoners_dilemma = nash.Game(A, B)

# Compute Nash equilibria
equilibria = list(prisoners_dilemma.support_enumeration())
print("Nash Equilibria:", equilibria)
```

**Application:** In economics, Nash Equilibrium is used to model how firms in an oligopoly choose production quantities that balance their profits given other firms’ outputs.

## Zero-Sum Games

### Theory

A Zero-Sum Game is a situation where one player's gain or loss is exactly balanced by the losses or gains of other participants. In such games, the total benefit available is fixed, and no cooperation is possible between the players. 

Mathematically, for a zero-sum game with payoff matrix \(A\), the sum of the payoffs for all players is zero for every possible outcome:
\[ \sum u_i(s_i) = 0 \]

### Examples

Let's consider the famous game, rock-paper-scissors, which is a zero-sum game. We will use Python to simulate this:

```python
import nashpy as nash
import numpy as np

# Define the payoff matrix
payoff_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])

# Create the game
rps_game = nash.Game(payoff_matrix)

# Compute Nash equilibria
equilibria = list(rps_game.support_enumeration())
print("Nash Equilibria:", equilibria)
```

**Application:** Zero-Sum Games are an essential concept in military strategy, where the gains of one side correspond to the losses of the other.

## Cooperative Games

### Theory

Cooperative Game Theory deals with scenarios where players can form coalitions and make binding commitments to improve their outcomes. A major focus is on how the total payoffs can be distributed among players in ways that reflect their contributions to the coalition.

One important concept here is the Shapley Value, which distributes payoffs based on each player's marginal contribution to any possible coalition:
\[ \phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (n-|S|-1)!}{n!} (v(S \cup \{i\}) - v(S)) \]

### Examples

Here's an example of how we calculate the Shapley Value in a cooperative game:

```python
from itertools import permutations

def calculate_shapley_value(vals, num_players):
    shapley_values = [0] * num_players
    
    for player in range(num_players):
        for perm in permutations(range(num_players)):
            index = perm.index(player)
            subset = perm[:index]
            complete_set = set(perm[:index + 1])
            
            marginal_value = vals[complete_set] - vals[set(subset)]
            shapley_values[player] += marginal_value
    
    fact = float(factorial(num_players))
    return [v/fact for v in shapley_values]

# Example for a 3-player game
vals = {frozenset([0]): 1, frozenset([1]): 2, frozenset([2]): 3, 
        frozenset([0,1]): 3, frozenset([1,2]): 5, frozenset([0,2]): 4, 
        frozenset([0,1,2]): 6}

shapley_values = calculate_shapley_value(vals, 3)
print("Shapley Values:", shapley_values)
```

**Application:** Cooperative Games are often used to understand business mergers, where companies can form alliances to increase market power and share profits based on their individual contributions.

## Evolutionary Game Theory

### Theory

Evolutionary Game Theory examines the strategic interaction in populations of agents who follow simple inheritance rules rather than attempts to maximize utility in the traditional sense. It introduces concepts such as evolutionary stability and replicator dynamics. 

An Evolutionarily Stable Strategy (ESS) is one that, if adopted by a population, cannot be invaded by any alternative strategy. Formally, a strategy \(s^*\) is ESS if:
\[ u(s^*,s^*) > u(s,s^*) \quad \text{or} \quad [u(s^*,s^*) = u(s,s^*) \text{ and } u(s^*,s) > u(s,s)] \]

### Examples

Let's simulate a Hawk-Dove game, which is a common model of conflict in evolutionary biology:

```python
import numpy as np

# Definition of the payoff matrix
payoff_matrix = np.array([[2, 0], [3, 1]])

def replicator_dynamics(payoff_matrix, x, time_steps):
    proportions = np.asarray(x)
    num_strategies = payoff_matrix.shape[0]
    
    for t in range(time_steps):
        expected_payoff = proportions @ payoff_matrix @ proportions.T
        fitness_increase = (payoff_matrix @ proportions.T) - expected_payoff
        proportions += proportions * fitness_increase
        
        # Keep proportions normalized
        proportions /= proportions.sum()
    
    return proportions

# Initial proportions of Hawk and Dove
initial_proportions = np.array([0.5, 0.5])
evolution = replicator_dynamics(payoff_matrix, initial_proportions, 100)
print("Final Strategy Distribution:", evolution)
```

**Application:** Evolutionary Game Theory is applied in understanding natural selection and social behaviors within populations in biology.

## Applications in Economics

### Theory

Game Theory provides powerful tools for economists to analyze situations where agents have to consider the reactions of other agents. It helps to predict outcomes in various scenarios such as auctions, pricing competition, and bargaining.

### Examples

`Auction Theory`: Auctions are quintessential examples where game theory is applied to predict and strategize participant bids:

```python
# Example of simple bidding strategy
import numpy as np

def second_price_auction(private_values):
    highest_bid = max(private_values)
    second_highest_bid = sorted(private_values)[-2]
    return second_highest_bid

bidders = np.array([100, 150, 120])
winning_price = second_price_auction(bidders)
print("Winning Price in the Auction:", winning_price)
```

**Application:** Game Theory is critical in designing mechanisms for markets such as spectrum auctions or financial exchanges where firms need to consider strategic bids to maximize profits and optimize allocations.

This notebook has provided a brief overview of several core concepts in Game Theory, each illustrated with examples and applications highlighting their practical utility. This framework serves as the backbone for decision-making processes in competitive and cooperative environments across a multitude of disciplines.