"""
CMPE 257 HW1: Neural Network Implementation from Scratch
Author: Lam Nguyen
Date: October 2025

Part 1: Forward and backward propagation for 2-layer neural network
Part 2: Gradient descent training (supports stochastic gradient descent)
"""

import numpy as np


class NeuralNetwork:
    """
    2-layer neural network with:
    - 2 input units (d^(0) = 2)
    - m hidden units (d^(1) = m)
    - 1 output unit (d^(2) = 1)
    - tanh activation function
    - squared error loss
    """
    
    def __init__(self, hidden_size=3, random_seed=42):
        """
        Initialize network weights.
        
        Args:
            hidden_size: Number of hidden units (m)
            random_seed: Seed for reproducibility
        """
        np.random.seed(random_seed)
        
        # Network architecture
        self.input_size = 2
        self.hidden_size = hidden_size
        self.output_size = 1
        
        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(hidden_size, 2) * np.sqrt(1.0 / 2)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(1, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, 1))
    
    def tanh(self, z):
        """Tanh activation function."""
        return np.tanh(z)
    
    def tanh_derivative(self, a):
        """Derivative of tanh: 1 - tanh^2(z)."""
        return 1 - np.power(a, 2)
    
    def forward_propagation(self, X):
        """
        Part 1: Forward propagation through the network.
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Layer 1: Input -> Hidden
        z1 = np.dot(self.W1, X) + self.b1
        a1 = self.tanh(z1)
        
        # Layer 2: Hidden -> Output
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.tanh(z2)
        
        # Cache for backpropagation
        cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        
        return a2, cache
    
    def compute_loss(self, predictions, y):
        """
        Compute squared error loss: L = (1/2n) * sum((predictions - y)^2)
        """
        n_samples = y.shape[1]
        loss = (1.0 / (2 * n_samples)) * np.sum(np.power(predictions - y, 2))
        return loss
    
    def back_propagation(self, y, cache):
        """
        Part 1: Backward propagation to compute gradients.
        Uses chain rule:
        - Output layer: delta2 = (a2 - y) * (1 - a2^2)
        - Hidden layer: delta1 = (W2^T * delta2) * (1 - a1^2)
        """
        X = cache['X']
        a1 = cache['a1']
        a2 = cache['a2']
        n_samples = X.shape[1]
        
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Backward pass - Output layer
        delta2 = (a2 - y) * self.tanh_derivative(a2)
        dW2 = (1.0 / n_samples) * np.dot(delta2, a1.T)
        db2 = (1.0 / n_samples) * np.sum(delta2, axis=1, keepdims=True)
        
        # Backward pass - Hidden layer
        delta1 = np.dot(self.W2.T, delta2) * self.tanh_derivative(a1)
        dW1 = (1.0 / n_samples) * np.dot(delta1, X.T)
        db1 = (1.0 / n_samples) * np.sum(delta1, axis=1, keepdims=True)
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def gradient_descent(self, X, y, epochs=1000, learning_rate=0.1, 
                        batch_size=None, verbose=True):
        """
        Part 2: Train network using gradient descent.
        
        Supports:
        - Batch gradient descent: batch_size=None (all samples)
        - Stochastic gradient descent: batch_size=1 (one sample)
        - Mini-batch gradient descent: batch_size=k (k samples)
        """
        # Ensure correct shape: (features, samples)
        if X.shape[0] != 2:
            X = X.T
        if y.shape[0] != 1:
            y = y.T
        
        n_samples = X.shape[1]
        
        # Default to batch gradient descent
        if batch_size is None:
            batch_size = n_samples
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data for stochastic/mini-batch
            if batch_size < n_samples:
                indices = np.random.permutation(n_samples)
                X = X[:, indices]
                y = y[:, indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Process batches
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X[:, i:end_idx]
                y_batch = y[:, i:end_idx]
                
                # Forward propagation
                predictions, cache = self.forward_propagation(X_batch)
                
                # Compute loss
                loss = self.compute_loss(predictions, y_batch)
                epoch_loss += loss
                
                # Backward propagation
                gradients = self.back_propagation(y_batch, cache)
                
                # Update weights
                self.W2 -= learning_rate * gradients['dW2']
                self.b2 -= learning_rate * gradients['db2']
                self.W1 -= learning_rate * gradients['dW1']
                self.b1 -= learning_rate * gradients['db1']
                
                n_batches += 1
            
            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
    
    def predict(self, X):
        """Make predictions on new data."""
        if X.shape[0] != 2:
            X = X.T
        predictions, _ = self.forward_propagation(X)
        return predictions


# ==================== Example Usage ====================

def test_xor():
    """Test on XOR problem (requires hidden layer to solve)."""
    print("\n" + "="*50)
    print("Testing on XOR Problem")
    print("="*50 + "\n")
    
    # XOR dataset
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    
    print("Input X:")
    print(X)
    print("\nTarget y:")
    print(y)
    
    # Create and train network
    print("\n--- Training with Batch Gradient Descent ---")
    nn = NeuralNetwork(hidden_size=4, random_seed=42)
    nn.gradient_descent(X, y, epochs=5000, learning_rate=0.5)
    
    # Test predictions
    print("\n--- Predictions ---")
    predictions = nn.predict(X)
    print("Raw predictions:", predictions.flatten())
    print("True values:    ", y.flatten())
    print("Rounded:        ", np.round(predictions).flatten())


def test_stochastic_gradient_descent():
    """Demonstrate stochastic gradient descent (batch_size=1)."""
    print("\n" + "="*50)
    print("Testing Stochastic Gradient Descent")
    print("="*50 + "\n")
    
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    
    nn = NeuralNetwork(hidden_size=4, random_seed=42)
    nn.gradient_descent(X, y, epochs=3000, learning_rate=0.5, 
                       batch_size=1, verbose=True)
    
    predictions = nn.predict(X)
    print("\nPredictions:", predictions.flatten())
    print("Rounded:    ", np.round(predictions).flatten())


if __name__ == "__main__":
    test_xor()
    test_stochastic_gradient_descent()
