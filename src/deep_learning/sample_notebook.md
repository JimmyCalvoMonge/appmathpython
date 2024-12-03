# Deep Learning

## Introduction

Deep Learning is a subfield of machine learning characterized by the use of artificial neural networks with multiple layers. These neural networks attempt to model high-level abstractions in data, achieving remarkable feats in image recognition, speech processing, and various other domains. This notebook explores the core architectures and techniques utilized in deep learning, providing both theoretical insights and practical examples.

---

## Neural Network Architectures

### Theory

Neural Networks, the foundation of deep learning, consist of layers of neurons or nodes, where each node in a layer is connected to every node in the preceding layer. The three primary types of layers are input, hidden, and output layers. 

A simple feedforward neural network can be mathematically represented as:

\[ a^{(l)} = f(W^{(l)}a^{(l-1)} + b^{(l)}) \]

Here:
- \( a^{(l)} \) represents the activations in layer \( l \).
- \( W^{(l)} \) is the weight matrix for layer \( l \).
- \( b^{(l)} \) is the bias for layer \( l \).
- \( f \) is the activation function, typically non-linear such as sigmoid, ReLU, or tanh.

The training of neural networks involves optimizing weights using algorithms like backpropagation, minimizing the error in predictions.

### Examples

```python
import numpy as np
from sklearn.neural_network import MLPClassifier

# Example: Simple Neural Network on a Classification Problem
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=(100,))

mlp = MLPClassifier(hidden_layer_sizes=(5, 2), activation='relu', max_iter=500)
mlp.fit(X, y)

predictions = mlp.predict(X)
print("Predictions:", predictions)
```
*Real-world Application*: Neural networks are extensively used in credit scoring systems for predicting borrower defaults based on historical data.

---

## Convolutional Neural Networks

### Theory

Convolutional Neural Networks (CNNs) are specialized neural networks designed for processing grid-like data with spatial hierarchies, commonly used in image recognition tasks. 

Key components include:
- **Convolutional Layers**: Perform convolutions, using filters to capture local patterns. 
  \[ (I * K)(x, y) = \sum_m\sum_n I(m, n)K(x-m, y-n) \]

- **Pooling Layers**: Reduce dimensionality and control overfitting by taking the maximum or average value in a particular region.
- **Fully Connected Layers**: Conclude the network to classify the input based on the learned features.

### Examples

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Example: Construct a Simple CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
*Real-world Application*: CNNs power advanced applications like autonomous vehicles, where they help in identifying pedestrians, obstacles, and road signs.

---

## Recurrent Neural Networks

### Theory

Recurrent Neural Networks (RNNs) are networks with loops allowing information to persist. They are especially powerful in sequence prediction problems. 

The architecture of an RNN features:
- **Hidden States**: Maintained over time to capture temporal dependencies.
  \[ h_t = f(W_h h_{t-1} + W_x x_t + b) \]

Challenges include dealing with vanishing or exploding gradients, often mitigated with Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) cells.

### Examples

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Example: Simple RNN using LSTM for sequence prediction
model = Sequential([
    LSTM(50, input_shape=(10, 1)),  # 10 timesteps, 1 feature
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
```
*Real-world Application*: RNNs are vital in natural language processing tasks such as machine translation and sentiment analysis.

---

## Optimization in Deep Learning

### Theory

Optimization in deep learning involves finding the best parameters that minimize a given loss function. Common strategies include:
- **Stochastic Gradient Descent (SGD)**: Updates parameters incrementally, \( \theta = \theta - \alpha \nabla J(\theta) \).
- **Adam Optimizer**: An adaptive learning rate method that computes individual learning rates for different parameters.
- Regularization techniques such as L2 regularization and dropout help prevent overfitting.

### Examples

```python
from keras.optimizers import Adam

# Example: Compile a model with Adam optimizer
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```
*Real-world Application*: Optimized neural networks improve reliability and performance of services like spam filters in email applications.

---

## Generative Adversarial Networks

### Theory

Generative Adversarial Networks (GANs) consist of two neural networks—the generator and the discriminator—competing against each other in a zero-sum game. The generator tries to generate realistic data to fool the discriminator, which aims to distinguish between real and fake data.

The loss for each can be modeled as:

\[ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] \]

Here, \( D \) and \( G \) represent discriminator and generator respectively.

### Examples

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Example: Simplified GAN structure
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 784)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return x

generator = Generator()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)

# Further training loops and discriminator implementation are needed for a full GAN setup.
```
*Real-world Application*: GANs are instrumental in applications like creating high-quality images and deepfake videos, transforming sectors from entertainment to e-commerce.

These sections provide a comprehensive overview of deep learning, demonstrating theoretical concepts with practical implementations in Python. Understanding and mastering these components will unlock new possibilities in building intelligent, data-driven applications.