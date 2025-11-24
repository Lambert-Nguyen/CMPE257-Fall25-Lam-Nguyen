"""
CMPE 257 HW1: Neural Network Implementation from Scratch
Author: Your Name
Date: October 2024

This module implements a 2-layer neural network with:
- 2 input units
- m hidden units (configurable)
- 1 output unit
- tanh activation function
- Squared error loss
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


class NeuralNetwork:
    """
    A 2-layer neural network with tanh activation and squared error loss.
    
    Architecture:
    - Input layer: input_size neurons
    - Hidden layer: hidden_size neurons with tanh activation
    - Output layer: output_size neurons with tanh activation
    
    Parameters:
    -----------
    input_size : int
        Number of input features (d^(0))
    hidden_size : int
        Number of hidden units (d^(1) = m)
    output_size : int
        Number of output units (d^(2))
    random_seed : int, optional
        Seed for reproducibility
    """
    
    def __init__(self, input_size: int = 2, hidden_size: int = 3, 
                 output_size: int = 1, random_seed: int = 42):
        np.random.seed(random_seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with small random values from normal distribution
        # Xavier/Glorot initialization: scale by sqrt(1/n_in)
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(1.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        
        # Store training history
        self.loss_history = []
        
    def tanh(self, z: np.ndarray) -> np.ndarray:
        """
        Hyperbolic tangent activation function.
        
        Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
        
        Parameters:
        -----------
        z : np.ndarray
            Input to activation function
            
        Returns:
        --------
        np.ndarray
            Output of tanh function, same shape as input
        """
        return np.tanh(z)
    
    def tanh_derivative(self, a: np.ndarray) -> np.ndarray:
        """
        Derivative of tanh function.
        
        Formula: d(tanh(z))/dz = 1 - tanh^2(z) = 1 - a^2
        where a = tanh(z)
        
        Parameters:
        -----------
        a : np.ndarray
            Output of tanh function (activation values)
            
        Returns:
        --------
        np.ndarray
            Derivative values, same shape as input
        """
        return 1 - np.power(a, 2)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Perform forward propagation through the network.
        
        Steps:
        1. Layer 1: z1 = W1 * X + b1, a1 = tanh(z1)
        2. Layer 2: z2 = W2 * a1 + b2, a2 = tanh(z2)
        
        Parameters:
        -----------
        X : np.ndarray
            Input data, shape (input_size, n_samples)
            
        Returns:
        --------
        predictions : np.ndarray
            Network output, shape (output_size, n_samples)
        cache : dict
            Dictionary containing intermediate values for backpropagation:
            - 'X': input data
            - 'z1': pre-activation values for hidden layer
            - 'a1': activation values for hidden layer
            - 'z2': pre-activation values for output layer
            - 'a2': activation values for output layer (predictions)
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Layer 1: Input to Hidden
        z1 = np.dot(self.W1, X) + self.b1  # Shape: (hidden_size, n_samples)
        a1 = self.tanh(z1)                  # Shape: (hidden_size, n_samples)
        
        # Layer 2: Hidden to Output
        z2 = np.dot(self.W2, a1) + self.b2  # Shape: (output_size, n_samples)
        a2 = self.tanh(z2)                   # Shape: (output_size, n_samples)
        
        # Cache for backpropagation
        cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2
        }
        
        return a2, cache
    
    def compute_loss(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """
        Compute squared error loss.
        
        Formula: L = (1/2n) * sum((predictions - y)^2)
        where n is the number of samples
        
        Parameters:
        -----------
        predictions : np.ndarray
            Network predictions, shape (output_size, n_samples)
        y : np.ndarray
            True labels, shape (output_size, n_samples)
            
        Returns:
        --------
        float
            Average squared error loss
        """
        n_samples = y.shape[1]
        loss = (1.0 / (2 * n_samples)) * np.sum(np.power(predictions - y, 2))
        return loss
    
    def backward(self, y: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform backward propagation to compute gradients.
        
        Uses chain rule to propagate error backward through network.
        
        For squared error loss L = (1/2) * (a2 - y)^2:
        
        Output layer (Layer 2):
        - delta2 = (a2 - y) * (1 - a2^2)
        - dW2 = delta2 * a1^T / n_samples
        - db2 = sum(delta2, axis=1) / n_samples
        
        Hidden layer (Layer 1):
        - delta1 = (W2^T * delta2) * (1 - a1^2)
        - dW1 = delta1 * X^T / n_samples
        - db1 = sum(delta1, axis=1) / n_samples
        
        Parameters:
        -----------
        y : np.ndarray
            True labels, shape (output_size, n_samples)
        cache : dict
            Cached values from forward propagation
            
        Returns:
        --------
        gradients : dict
            Dictionary containing gradients:
            - 'dW2': gradient of W2
            - 'db2': gradient of b2
            - 'dW1': gradient of W1
            - 'db1': gradient of b1
        """
        # Extract cached values
        X = cache['X']
        a1 = cache['a1']
        a2 = cache['a2']
        
        n_samples = X.shape[1]
        
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # ===== Backward pass for Layer 2 (Output layer) =====
        # Error at output: dL/da2 = (a2 - y)
        # Delta: dL/dz2 = dL/da2 * da2/dz2 = (a2 - y) * (1 - a2^2)
        delta2 = (a2 - y) * self.tanh_derivative(a2)  # Shape: (output_size, n_samples)
        
        # Gradients for W2 and b2
        dW2 = (1.0 / n_samples) * np.dot(delta2, a1.T)  # Shape: (output_size, hidden_size)
        db2 = (1.0 / n_samples) * np.sum(delta2, axis=1, keepdims=True)  # Shape: (output_size, 1)
        
        # ===== Backward pass for Layer 1 (Hidden layer) =====
        # Propagate error back: dL/da1 = W2^T * delta2
        # Delta: dL/dz1 = dL/da1 * da1/dz1 = (W2^T * delta2) * (1 - a1^2)
        delta1 = np.dot(self.W2.T, delta2) * self.tanh_derivative(a1)  # Shape: (hidden_size, n_samples)
        
        # Gradients for W1 and b1
        dW1 = (1.0 / n_samples) * np.dot(delta1, X.T)  # Shape: (hidden_size, input_size)
        db1 = (1.0 / n_samples) * np.sum(delta1, axis=1, keepdims=True)  # Shape: (hidden_size, 1)
        
        gradients = {
            'dW2': dW2,
            'db2': db2,
            'dW1': dW1,
            'db1': db1
        }
        
        return gradients
    
    def update_weights(self, gradients: Dict[str, np.ndarray], learning_rate: float):
        """
        Update network parameters using gradient descent.
        
        Update rule: theta_new = theta_old - learning_rate * gradient
        
        Parameters:
        -----------
        gradients : dict
            Dictionary of gradients computed by backward propagation
        learning_rate : float
            Step size for gradient descent
        """
        self.W2 -= learning_rate * gradients['dW2']
        self.b2 -= learning_rate * gradients['db2']
        self.W1 -= learning_rate * gradients['dW1']
        self.b1 -= learning_rate * gradients['db1']
    
    def train_step(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        """
        Perform one training step: forward pass, backward pass, update weights.
        
        Parameters:
        -----------
        X : np.ndarray
            Training data, shape (input_size, n_samples)
        y : np.ndarray
            Training labels, shape (output_size, n_samples)
        learning_rate : float
            Learning rate for gradient descent
            
        Returns:
        --------
        float
            Loss value for this step
        """
        # Forward propagation
        predictions, cache = self.forward(X)
        
        # Compute loss
        loss = self.compute_loss(predictions, y)
        
        # Backward propagation
        gradients = self.backward(y, cache)
        
        # Update weights
        self.update_weights(gradients, learning_rate)
        
        return loss
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              learning_rate: float = 0.1, batch_size: int = None, 
              verbose: bool = True, print_every: int = 100):
        """
        Train the neural network using gradient descent.
        
        Supports three modes:
        1. Batch gradient descent: batch_size = None (use all samples)
        2. Stochastic gradient descent: batch_size = 1 (use one sample)
        3. Mini-batch gradient descent: batch_size = k (use k samples)
        
        Parameters:
        -----------
        X : np.ndarray
            Training data, shape (input_size, n_samples) or (n_samples, input_size)
        y : np.ndarray
            Training labels, shape (output_size, n_samples) or (n_samples, output_size)
        epochs : int, default=1000
            Number of training epochs
        learning_rate : float, default=0.1
            Learning rate for gradient descent
        batch_size : int or None, default=None
            Size of mini-batches. If None, use full batch gradient descent.
            If 1, use stochastic gradient descent.
        verbose : bool, default=True
            Whether to print training progress
        print_every : int, default=100
            Print loss every this many epochs
        """
        # Ensure correct shape: (features, samples)
        if X.shape[0] != self.input_size:
            X = X.T
        if y.shape[0] != self.output_size:
            y = y.T
        
        n_samples = X.shape[1]
        
        # Default batch size is full batch
        if batch_size is None:
            batch_size = n_samples
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data for stochastic/mini-batch gradient descent
            if batch_size < n_samples:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[:, indices]
                y_shuffled = y[:, indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            epoch_loss = 0
            n_batches = 0
            
            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                # Get batch
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[:, i:end_idx]
                y_batch = y_shuffled[:, i:end_idx]
                
                # Training step
                loss = self.train_step(X_batch, y_batch, learning_rate)
                epoch_loss += loss
                n_batches += 1
            
            # Average loss for the epoch
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            # Print progress
            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data, shape (input_size, n_samples) or (n_samples, input_size)
            
        Returns:
        --------
        np.ndarray
            Predictions, shape (output_size, n_samples)
        """
        # Ensure correct shape
        if X.shape[0] != self.input_size:
            X = X.T
        
        predictions, _ = self.forward(X)
        return predictions
    
    def plot_loss_history(self):
        """Plot the training loss history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        plt.show()


def numerical_gradient_check(nn: NeuralNetwork, X: np.ndarray, y: np.ndarray, 
                             epsilon: float = 1e-7) -> Dict[str, float]:
    """
    Verify backpropagation implementation using numerical gradient approximation.
    
    Uses central difference: f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)
    
    Parameters:
    -----------
    nn : NeuralNetwork
        Neural network instance
    X : np.ndarray
        Input data
    y : np.ndarray
        Labels
    epsilon : float
        Small value for numerical approximation
        
    Returns:
    --------
    dict
        Relative errors for each parameter
    """
    # Get analytical gradients
    predictions, cache = nn.forward(X)
    analytical_grads = nn.backward(y, cache)
    
    # Check each parameter
    relative_errors = {}
    
    for param_name in ['W1', 'b1', 'W2', 'b2']:
        param = getattr(nn, param_name)
        grad_name = 'd' + param_name
        analytical_grad = analytical_grads[grad_name]
        
        numerical_grad = np.zeros_like(param)
        
        # Compute numerical gradient for each element
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_value = param[idx]
            
            # f(x + epsilon)
            param[idx] = old_value + epsilon
            pred_plus, _ = nn.forward(X)
            loss_plus = nn.compute_loss(pred_plus, y)
            
            # f(x - epsilon)
            param[idx] = old_value - epsilon
            pred_minus, _ = nn.forward(X)
            loss_minus = nn.compute_loss(pred_minus, y)
            
            # Central difference
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restore original value
            param[idx] = old_value
            it.iternext()
        
        # Compute relative error
        numerator = np.linalg.norm(analytical_grad - numerical_grad)
        denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
        relative_error = numerator / denominator if denominator != 0 else 0
        
        relative_errors[param_name] = relative_error
        
    return relative_errors


# =============== Example Usage ===============

def test_xor_problem():
    """
    Test the neural network on the XOR problem.
    
    XOR truth table:
    X1  X2  Y
    0   0   0
    0   1   1
    1   0   1
    1   1   0
    """
    print("\n" + "="*50)
    print("Testing Neural Network on XOR Problem")
    print("="*50 + "\n")
    
    # XOR dataset
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    
    print("Input data X:")
    print(X)
    print("\nTarget output y:")
    print(y)
    
    # Create and train network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, random_seed=42)
    
    print("\n--- Training Neural Network ---")
    nn.train(X, y, epochs=5000, learning_rate=0.5, verbose=True, print_every=1000)
    
    # Test predictions
    print("\n--- Final Predictions ---")
    predictions = nn.predict(X)
    print("Predictions:", predictions.flatten())
    print("True values:", y.flatten())
    print("\nRounded predictions:", np.round(predictions).flatten())
    
    # Plot loss
    nn.plot_loss_history()


def test_gradient_check():
    """Test the backpropagation implementation using numerical gradient checking."""
    print("\n" + "="*50)
    print("Gradient Checking")
    print("="*50 + "\n")
    
    # Small dataset for testing
    X = np.random.randn(2, 5)
    y = np.random.randn(1, 5)
    
    # Create network
    nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
    
    # Check gradients
    print("Computing gradients...")
    relative_errors = numerical_gradient_check(nn, X, y)
    
    print("\nRelative errors (should be < 1e-7 for correct implementation):")
    for param, error in relative_errors.items():
        status = "✓ PASS" if error < 1e-7 else "✗ FAIL"
        print(f"{param}: {error:.2e} {status}")


if __name__ == "__main__":
    # Run tests
    test_gradient_check()
    test_xor_problem()
