```markdown
# Classification Models in Machine Learning

## Introduction

Classification models are an integral part of machine learning used to categorize data into predefined classes. Unlike regression models which predict continuous outputs, classification models focus on predicting discrete outcomes. These models have wide applications across different fields, from medical diagnosis to spam detection. In this notebook, we will delve into some of the popular classification models, exploring their theoretical underpinnings, potential applications, and practical implementation in Python.

## Decision Trees

### Theory

Decision trees are a non-parametric supervised learning method used for classification (and regression). The fundamental idea is to split the data into subsets based on the value of input features, constructing a tree with decision nodes and leaf nodes. Each internal node represents a "test" on an attribute, each branch a result of the test, and each leaf node a class label.

The decision tree can be represented mathematically through concepts like entropy or Gini impurity. The goal is to maximize information gain, defined as:

\[
IG(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \cdot Entropy(S_v)
\]

Where \(Entropy(S)\) is calculated as:

\[
Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
\]

### Examples

Here is a simple Python example using scikit-learn to classify a dataset with a decision tree:

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = tree_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

**Real Applications:** Decision trees are used in credit scoring, diagnosis of diseases, and even recommendation systems.

## Support Vector Machines

### Theory

Support Vector Machines (SVM) are powerful classification techniques that aim to find the hyperplane that best separates different classes in a feature space. The vector points lying on the hyperplane are called support vectors. For a linearly separable dataset, the decision function is:

\[
f(x) = \text{sign}(w \cdot x + b)
\]

where \(w\) is the weight vector and \(b\) is the bias. In nonlinear cases, kernel functions transform the feature space, enabling hyperplane separation.

### Examples

Using SVM with a Radial Basis Function (RBF) kernel in Python:

```python
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM model
svm_clf = SVC(kernel='rbf', gamma='auto')
svm_clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = svm_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

**Real Applications:** SVMs are used in image recognition, text categorization, and bioinformatics.

## Neural Networks for Classification

### Theory

Neural Networks are computational models inspired by the human brain, consisting of interconnected layers of nodes called neurons. Each neuron computes a weighted sum of its inputs, applies an activation function, and passes the result to the next layer. The network is trained through optimization techniques like backpropagation to minimize a loss function.

For a binary classification problem, the neural network might use the sigmoid activation function, modeled as:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

For multiple classes, softmax is commonly used in the output layer to map scores to probabilities.

### Examples

Building a simple neural network using Keras for classification:

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset and preprocessing
iris = load_iris()
X = iris.data
y = to_categorical(iris.target)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define neural network model
model = Sequential([
    Dense(10, input_shape=(4,), activation='relu'),
    Dense(3, activation='softmax')
])

# Compile and train model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {accuracy}")
```

**Real Applications:** Neural networks are widely used in image and voice recognition, natural language processing, and financial forecasting.

## Ensemble Methods

### Theory

Ensemble Methods are techniques that combine multiple models to improve overall performance. The main idea is that multiple weak models, when combined, produce a more accurate prediction. Two popular ensemble methods are **Bagging** (Bootstrap Aggregating) and **Boosting**.

**Bagging:** Each model is trained on a random subset of the data. Predictions from these models are combined, usually by voting.

**Boosting:** Models are added sequentially, each correcting the errors of its predecessor. Popular boosting algorithms include AdaBoost and Gradient Boosting.

### Examples

Here is how you can implement Random Forest (a bagging method) and AdaBoost using scikit-learn:

```python
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_accuracy = rf_clf.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_accuracy}")

# AdaBoost
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_clf.fit(X_train, y_train)
ada_accuracy = ada_clf.score(X_test, y_test)
print(f"AdaBoost Accuracy: {ada_accuracy}")
```

**Real Applications:** Ensemble methods excel in competition solutions, fraud detection, and in many winning algorithms in platforms like Kaggle.

## Performance Metrics

### Theory

Performance metrics are crucial for evaluating classification model effectiveness. Here are key metrics:

- **Accuracy:** Ratio of correctly predicted observations to total observations.
  
  \[
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  \]

- **Precision:** Ratio of correctly predicted positive observations to total predicted positives.

  \[
  \text{Precision} = \frac{TP}{TP + FP}
  \]

- **Recall (Sensitivity):** Ratio of correctly predicted positive observations to all actual positives.

  \[
  \text{Recall} = \frac{TP}{TP + FN}
  \]

- **F1 Score:** Harmonic mean of Precision and Recall, useful when the class distribution is uneven.

  \[
  \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  \]

### Examples

To compute these metrics using scikit-learn:

```python
from sklearn.metrics import classification_report, confusion_matrix

# Assuming y_test and y_pred from previous model predictions
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
```

In real-world scenarios, selecting the appropriate metric is critical and often depends on the importance of false positives versus false negatives within a specific domain.

---

This notebook provided an overview of various classification methods, touching upon theory, practical implementation, and real-world applications. Understanding these models' strengths and limitations is fundamental to their successful application in industry.
```