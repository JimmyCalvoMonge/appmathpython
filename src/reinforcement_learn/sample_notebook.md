# Reinforcement Learning

## Introduction

Reinforcement Learning (RL) is a foundational area of machine learning where an agent learns to make decisions by interacting with an environment to achieve a goal. RL is distinguished by how it mimics the trial-and-error learning process humans and animals use to acquire new skills. Given its roots in behavioral psychology and decision theory, RL is crucial for developing autonomous systems that evolve and adapt over time.

At its core, RL involves:

- **Agents**: The learner or decision-maker.
- **Environment**: The external system with which the agent interacts.
- **Actions**: All possible moves the agent can take.
- **States**: Current situations returned by the environment.
- **Rewards**: Feedback from the environment for each action.

This notebook explores various methodologies within RL, demonstrating their theory, common applications, and practical implementations in Python.

## Markov Decision Processes

### Theory

Markov Decision Processes (MDPs) provide a mathematical framework for modeling decision-making wherein outcomes are partly random and partly under the control of a decision-maker.

**MDP Components:**

- **States (S):** A finite set representing the situations the agent can encounter.
- **Actions (A):** A finite set of possible actions the agent can take.
- **Transition Model (T):** Probability distribution over states guiding state transition after taking an action.
- **Reward Function (R):** Immediate reward received after transitioning from one state to another.
- **Policy (π):** A strategy that specifies the action that the agent takes in each state.

The objective in an MDP is to find a policy π that maximizes the expected sum of rewards (also known as the return).

### Examples

Let's consider a simple grid world where an agent can move in four directions: up, down, left, and right until it reaches a goal or falls into a pit.

```python
import numpy as np
import random

class GridWorld:
    def __init__(self, grid, start, terminal_states, rewards):
        self.grid = grid
        self.agent_position = start
        self.terminal_states = terminal_states
        self.rewards = rewards

    def step(self, action):
        if action == "up" and self.agent_position[0] > 0:
            self.agent_position[0] -= 1
        elif action == "down" and self.agent_position[0] < len(self.grid)-1:
            self.agent_position[0] += 1
        elif action == "left" and self.agent_position[1] > 0:
            self.agent_position[1] -= 1
        elif action == "right" and self.agent_position[1] < len(self.grid[0])-1:
            self.agent_position[1] += 1

        reward = self.rewards.get(tuple(self.agent_position), 0)
        done = tuple(self.agent_position) in self.terminal_states
        return tuple(self.agent_position), reward, done

# Define the grid, starting point, terminal states, and rewards
grid = np.zeros((4, 4))
start = [0, 0]
terminal_states = [(3, 3), (3, 1)]
rewards = {(3, 3): 1, (3, 1): -1}

# Initialize and interact with the environment
env = GridWorld(grid, start, terminal_states, rewards)

# Sample interaction
action = random.choice(["up", "down", "left", "right"])
state, reward, done = env.step(action)
print(f"Action: {action}, New State: {state}, Reward: {reward}, Done: {done}")
```

## Q-Learning

### Theory

Q-Learning is an off-policy, model-free RL algorithm to learn the value of an action in a particular state. It seeks to find the best policy by learning an action-value function, \( Q(s, a) \), representing the expected utility of taking a given action \( a \) in a given state \( s \).

**Q-Learning Update Rule**:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]
\]

where:
- \( \alpha \) is the learning rate.
- \( \gamma \) is the discount factor.
- \( r \) is the reward observed after executing action \( a \).

### Examples

```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.99, exploration_min=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(range(self.action_size))
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)

# Usage:
num_states = 4 * 4  # Example state size for a 4x4 grid
num_actions = 4  # Up, down, left, right
agent = QLearningAgent(num_states, num_actions)

# Simulate interactions with the environment
state = 0  # Initial state
next_state = 1  # Example state transition
reward = 0  # Sample reward

action = agent.choose_action(state)
agent.update(state, action, reward, next_state)
```

## Policy Gradient Methods

### Theory

Policy Gradient Methods directly learn the policy which maps states to actions without needing a value function. Policies are usually parameterized with a neural network, and learning involves optimizing these parameters to maximize expected return.

**Objective Function**:

\[
J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
\]

**Gradient Ascent**:
The parameters \( \theta \) are updated using the gradient ascent:

\[
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
\]

### Examples

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(state_size, action_size)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class PolicyGradientAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def choose_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy_network(state)
        action = torch.multinomial(probs, 1).item()
        return action

    def update(self, rewards, log_probs):
        discounts = np.array([0.99**i for i in range(len(rewards))])
        returns = np.array([sum(rewards[i:] * discounts[:len(rewards[i:])]) for i in range(len(rewards))])
        returns = torch.FloatTensor(returns)

        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)
        loss = torch.cat(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Simulating an environment interaction
state_size = 4
action_size = 2
agent = PolicyGradientAgent(state_size, action_size)

state = np.random.rand(state_size)  # Example state
action = agent.choose_action(state)

# Assume we collected rewards and log_probs during an episode
rewards = [1, 0, -1]
log_probs = [torch.log(torch.tensor(0.5)), torch.log(torch.tensor(0.3)), torch.log(torch.tensor(0.8))]
agent.update(rewards, log_probs)
```

## Deep Reinforcement Learning

### Theory

Deep Reinforcement Learning (Deep RL) combines neural networks with reinforcement learning principles to create end-to-end systems where the agent learns from high-dimensional inputs. The incorporation of deep learning enables RL to tackle problems with large state spaces, such as visual inputs from games.

A popular Deep RL algorithm is Deep Q-Networks (DQN), which approximates the Q-function using a neural network:

- **Experience Replay**: Stores past experiences and samples a mini-batch for training to break the correlation between consecutive samples.
- **Target Network**: A separate network to calculate target Q-values that is periodically updated to stabilize learning.

### Examples

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=64, gamma=0.99, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def choose_action(self, state):
        if random.uniform(0, 1) < 0.1:  # Exploration vs Explotation
            return random.choice(range(self.action_size))
        state = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(self.model(state)).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            target = reward
            if not done:
                target += self.gamma * torch.amax(self.target_model(next_state)).item()

            q_values = self.model(state)
            q_value = q_values[0][action]

            loss = nn.functional.mse_loss(q_value, torch.tensor(target))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Usage in a hypothetical environment
state_size = 4
action_size = 2
agent = DQNAgent(state_size, action_size)

# Remember past experiences ...
agent.remember([1, 0, 0, 1], 1, 1, [1, 1, 0, 0], False)

# ... and learn from them
agent.replay()
```

## Applications in Game AI

### Theory

Reinforcement Learning has extensive applications in Game AI, where it’s used to develop strategies that outperform traditional rule-based methods. Notable successes include AlphaGo's defeat of the world champion in the game of Go, using strategies that combined RL with deep neural networks and tree search techniques.

In video games, RL systems can learn to play directly from pixels, surpassing human-level performance in several cases. This success is exemplified by breakthroughs such as DeepMind's accomplishment in Atari games using the DQN algorithm.

### Examples

The use of RL in Game AI generally involves training an agent using extensive simulated game play. Here's a very simplified setup for a Tic-Tac-Toe game:

```python
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_winner = None

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        # Check if the combination leads to a win
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for condition in win_conditions:
            if all([self.board[i] == letter for i in condition]):
                return True
        return False

# Simple example to use RL for Tic-Tac-Toe
game = TicTacToe()
agent_x = QLearningAgent(9, 9)
agent_o = QLearningAgent(9, 9)  # Agent for playing 'O'

# Each agent learns to play the game through repeated simulations
for episode in range(10000):
    current_player = 'X'

    while not game.current_winner:
        available = game.available_moves()
        if current_player == 'X':
            action = agent_x.choose_action(len(available))
            game.make_move(action, current_player)
            current_player = 'O'
        else:
            action = agent_o.choose_action(len(available))
            game.make_move(action, current_player)
            current_player = 'X'
    
    # Update the agents based on game outcomes...

print("Training complete!")
```

In these applications, RL algorithms adapt through self-play, ultimately creating complex AI models that behave optimally or creatively in game environments.