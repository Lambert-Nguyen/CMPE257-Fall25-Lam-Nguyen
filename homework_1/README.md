# CMPE 257 HW1: Neural Network from Scratch

**Author:** Lam Nguyen  
**Date:** October 2025

## Overview

This project implements a 2-layer neural network from scratch using only NumPy, fulfilling the following requirements:

- **Part 1:** Forward and backward propagation with gradient computation
- **Part 2:** Gradient descent training (supports batch, stochastic, and mini-batch modes)

### Network Architecture

- **Input layer:** 2 units (d⁽⁰⁾ = 2)
- **Hidden layer:** m units (d⁽¹⁾ = m, configurable)
- **Output layer:** 1 unit (d⁽²⁾ = 1)
- **Activation function:** tanh
- **Loss function:** Squared error

## Requirements

- Python 3.7+
- NumPy

### Installation

```bash
pip install numpy
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Examples

Simply run the script to see two test cases:

```bash
python Task2.py
```

This will execute:
1. **XOR Problem with Batch Gradient Descent** - Uses all samples at once
2. **XOR Problem with Stochastic Gradient Descent** - Uses one sample at a time

### Using the Neural Network Class

#### Basic Example

```python
import numpy as np
from Task2 import NeuralNetwork

# Create training data
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
y = np.array([[0, 1, 1, 0]])

# Initialize network with 4 hidden units
nn = NeuralNetwork(hidden_size=4, random_seed=42)

# Train with batch gradient descent
nn.gradient_descent(X, y, epochs=5000, learning_rate=0.5)

# Make predictions
predictions = nn.predict(X)
print("Predictions:", predictions.flatten())
```

#### Stochastic Gradient Descent

```python
# Train with stochastic gradient descent (batch_size=1)
nn = NeuralNetwork(hidden_size=4)
nn.gradient_descent(X, y, epochs=3000, learning_rate=0.5, batch_size=1)
```

#### Mini-Batch Gradient Descent

```python
# Train with mini-batch gradient descent (batch_size=2)
nn = NeuralNetwork(hidden_size=4)
nn.gradient_descent(X, y, epochs=3000, learning_rate=0.5, batch_size=2)
```

### Method Documentation

#### `forward_propagation(X)`
Performs forward pass through the network.
- **Input:** X - shape (2, n_samples)
- **Returns:** predictions, cache

#### `back_propagation(y, cache)`
Computes gradients using backpropagation.
- **Input:** y - true labels, cache - from forward pass
- **Returns:** Dictionary of gradients {dW1, db1, dW2, db2}

#### `gradient_descent(X, y, epochs, learning_rate, batch_size, verbose)`
Trains the network using gradient descent.
- **X:** Training data (2, n_samples) or (n_samples, 2)
- **y:** Labels (1, n_samples) or (n_samples, 1)
- **epochs:** Number of training iterations (default: 1000)
- **learning_rate:** Step size (default: 0.1)
- **batch_size:** Samples per batch (default: None = all samples)
- **verbose:** Print progress (default: True)

#### `predict(X)`
Makes predictions on new data.
- **Input:** X - shape (2, n_samples) or (n_samples, 2)
- **Returns:** predictions - shape (1, n_samples)

## Expected Output

When running `python Task2.py`, you should see:

```
==================================================
Testing on XOR Problem
==================================================

Input X:
[[0 0 1 1]
 [0 1 0 1]]

Target y:
[[0 1 1 0]]

--- Training with Batch Gradient Descent ---
Epoch 0/5000, Loss: 0.243297
Epoch 100/5000, Loss: 0.081173
...
Epoch 4999/5000, Loss: 0.000028

--- Predictions ---
Raw predictions: [1.83e-04 9.89e-01 9.89e-01 2.95e-04]
True values:     [0 1 1 0]
Rounded:         [0. 1. 1. 0.]
```

The network successfully learns the XOR function, which is non-linearly separable and requires a hidden layer to solve.

## Implementation Details

### Forward Propagation (Part 1)
```
Layer 1: z₁ = W₁X + b₁,  a₁ = tanh(z₁)
Layer 2: z₂ = W₂a₁ + b₂, a₂ = tanh(z₂)
```

### Backward Propagation (Part 1)
```
Output layer:  δ₂ = (a₂ - y) ⊙ (1 - a₂²)
               dW₂ = δ₂a₁ᵀ/n,  db₂ = sum(δ₂)/n

Hidden layer:  δ₁ = (W₂ᵀδ₂) ⊙ (1 - a₁²)
               dW₁ = δ₁Xᵀ/n,   db₁ = sum(δ₁)/n
```

### Gradient Descent (Part 2)
```
θ_new = θ_old - α × ∇θ
```

Where α is the learning rate and ∇θ are the computed gradients.

## Customization

### Different Hidden Layer Sizes

```python
# Try different numbers of hidden units
nn = NeuralNetwork(hidden_size=10)
```

### Different Learning Rates

```python
# Adjust learning rate for faster/slower convergence
nn.gradient_descent(X, y, epochs=5000, learning_rate=0.1)
```

### Your Own Data

```python
# Create your own dataset
X_custom = np.array([[x1_1, x1_2, x1_3, ...],
                     [x2_1, x2_2, x2_3, ...]])
y_custom = np.array([[y1, y2, y3, ...]])

nn = NeuralNetwork(hidden_size=5)
nn.gradient_descent(X_custom, y_custom, epochs=2000, learning_rate=0.3)
predictions = nn.predict(X_custom)
```

## Files

- `Task2.py` - Main implementation (this file)
- `neural_network.py` - Extended version with additional features
- `examples.py` - Additional examples and utilities
- `requirements.txt` - Python dependencies

## Notes

- The network uses **Xavier/Glorot initialization** for better convergence
- Gradients are computed using the **chain rule** of calculus
- The implementation supports **any number of samples** for flexible training
- **Batch size** controls the gradient descent variant:
  - `None` → Batch GD (uses all samples)
  - `1` → Stochastic GD (uses 1 sample)
  - `k` → Mini-batch GD (uses k samples)

## Troubleshooting

### Network not converging?
- Try increasing the number of hidden units
- Adjust the learning rate (try 0.1, 0.5, or 1.0)
- Increase the number of epochs

### Loss increasing?
- Learning rate might be too high, reduce it
- Check your data format (should be shape (2, n_samples))

### Predictions stuck at same value?
- Network might be in a local minimum
- Try different random seed
- Increase hidden units or learning rate

## License

MIT License - Free for educational use

