"""
CMPE 257 HW1: Example Usage and Visualizations
This file demonstrates how to use the neural network implementation
with various datasets and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork


def generate_circle_dataset(n_samples=200, noise=0.1, random_seed=42):
    """
    Generate a circular dataset for binary classification.
    Points inside the circle are class 0, points outside are class 1.
    """
    np.random.seed(random_seed)
    
    # Generate random points
    X = np.random.randn(2, n_samples)
    
    # Label based on distance from origin
    radius = np.sqrt(X[0]**2 + X[1]**2)
    y = (radius > 1.0).astype(float).reshape(1, -1)
    
    # Add noise to labels
    flip_indices = np.random.rand(n_samples) < noise
    y[0, flip_indices] = 1 - y[0, flip_indices]
    
    return X, y


def generate_spiral_dataset(n_samples=100, noise=0.2, random_seed=42):
    """Generate a two-spiral dataset."""
    np.random.seed(random_seed)
    
    n = n_samples // 2
    
    # Generate spiral 1
    theta1 = np.linspace(0, 4*np.pi, n)
    r1 = np.linspace(0, 1, n)
    x1 = r1 * np.cos(theta1) + np.random.randn(n) * noise
    y1 = r1 * np.sin(theta1) + np.random.randn(n) * noise
    
    # Generate spiral 2
    theta2 = np.linspace(0, 4*np.pi, n) + np.pi
    r2 = np.linspace(0, 1, n)
    x2 = r2 * np.cos(theta2) + np.random.randn(n) * noise
    y2 = r2 * np.sin(theta2) + np.random.randn(n) * noise
    
    # Combine
    X = np.vstack([np.hstack([x1, x2]), np.hstack([y1, y2])])
    y = np.hstack([np.zeros(n), np.ones(n)]).reshape(1, -1)
    
    return X, y


def plot_decision_boundary(nn, X, y, title="Decision Boundary"):
    """
    Plot decision boundary learned by the neural network.
    
    Parameters:
    -----------
    nn : NeuralNetwork
        Trained neural network
    X : np.ndarray
        Input data, shape (2, n_samples)
    y : np.ndarray
        Labels, shape (1, n_samples)
    title : str
        Plot title
    """
    # Create mesh
    h = 0.02  # Step size in mesh
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    mesh_input = np.c_[xx.ravel(), yy.ravel()].T
    Z = nn.predict(mesh_input)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=20, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.colorbar(label='Prediction')
    
    # Plot data points
    scatter = plt.scatter(X[0, :], X[1, :], c=y[0, :], 
                         cmap=plt.cm.RdYlBu, edgecolors='black', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.tight_layout()
    plt.show()


def plot_3d_surface(nn, X, y, title="Neural Network Output Surface"):
    """
    Plot 3D surface of neural network predictions.
    
    Parameters:
    -----------
    nn : NeuralNetwork
        Trained neural network
    X : np.ndarray
        Input data
    y : np.ndarray
        Labels
    title : str
        Plot title
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create mesh
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    # Predict on mesh
    mesh_input = np.c_[xx.ravel(), yy.ravel()].T
    Z = nn.predict(mesh_input)
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.7)
    
    # Plot data points
    ax.scatter(X[0, :], X[1, :], y[0, :], c=y[0, :], 
              cmap='RdYlBu', s=50, edgecolors='black')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Output')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def compare_learning_rates(X, y, learning_rates=[0.01, 0.1, 0.5, 1.0], 
                           epochs=2000, hidden_size=5):
    """
    Compare training with different learning rates.
    
    Parameters:
    -----------
    X : np.ndarray
        Training data
    y : np.ndarray
        Training labels
    learning_rates : list
        List of learning rates to try
    epochs : int
        Number of training epochs
    hidden_size : int
        Number of hidden units
    """
    plt.figure(figsize=(12, 8))
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        nn = NeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1)
        nn.train(X, y, epochs=epochs, learning_rate=lr, verbose=False)
        
        plt.plot(nn.loss_history, label=f'LR = {lr}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


def compare_hidden_sizes(X, y, hidden_sizes=[2, 5, 10, 20], 
                        epochs=2000, learning_rate=0.1):
    """
    Compare training with different numbers of hidden units.
    
    Parameters:
    -----------
    X : np.ndarray
        Training data
    y : np.ndarray
        Training labels
    hidden_sizes : list
        List of hidden layer sizes to try
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate
    """
    plt.figure(figsize=(12, 8))
    
    for h in hidden_sizes:
        print(f"\nTraining with {h} hidden units")
        nn = NeuralNetwork(input_size=2, hidden_size=h, output_size=1)
        nn.train(X, y, epochs=epochs, learning_rate=learning_rate, verbose=False)
        
        plt.plot(nn.loss_history, label=f'{h} hidden units')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Hidden Layer Sizes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def demonstrate_stochastic_vs_batch(X, y, epochs=500):
    """
    Compare batch, mini-batch, and stochastic gradient descent.
    """
    print("\n" + "="*60)
    print("Comparing Gradient Descent Variants")
    print("="*60)
    
    configs = [
        ("Batch GD", None),
        ("Mini-batch GD (batch_size=10)", 10),
        ("Stochastic GD (batch_size=1)", 1)
    ]
    
    plt.figure(figsize=(12, 8))
    
    for name, batch_size in configs:
        print(f"\nTraining with {name}")
        nn = NeuralNetwork(input_size=2, hidden_size=5, output_size=1, random_seed=42)
        nn.train(X, y, epochs=epochs, learning_rate=0.1, 
                batch_size=batch_size, verbose=False)
        
        plt.plot(nn.loss_history, label=name, alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Comparison of Gradient Descent Variants')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============== Main Examples ===============

def example_1_xor():
    """Example 1: XOR Problem"""
    print("\n" + "="*60)
    print("EXAMPLE 1: XOR Problem")
    print("="*60)
    
    # XOR dataset
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    
    # Train network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=5000, learning_rate=0.5, print_every=1000)
    
    # Predictions
    print("\n--- Predictions ---")
    predictions = nn.predict(X)
    for i in range(X.shape[1]):
        print(f"Input: [{X[0,i]}, {X[1,i]}] -> "
              f"Prediction: {predictions[0,i]:.4f}, "
              f"True: {y[0,i]}")
    
    # Plot decision boundary
    plot_decision_boundary(nn, X, y, "XOR Problem - Decision Boundary")
    nn.plot_loss_history()


def example_2_circles():
    """Example 2: Circular Dataset"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Circular Dataset")
    print("="*60)
    
    # Generate dataset
    X, y = generate_circle_dataset(n_samples=300, noise=0.05)
    
    # Train network
    nn = NeuralNetwork(input_size=2, hidden_size=8, output_size=1)
    nn.train(X, y, epochs=3000, learning_rate=0.1, print_every=500)
    
    # Visualize
    plot_decision_boundary(nn, X, y, "Circular Dataset - Decision Boundary")
    plot_3d_surface(nn, X, y, "Circular Dataset - Output Surface")
    nn.plot_loss_history()
    
    # Accuracy
    predictions = nn.predict(X)
    predicted_labels = (predictions > 0.5).astype(float)
    accuracy = np.mean(predicted_labels == y)
    print(f"\nAccuracy: {accuracy*100:.2f}%")


def example_3_spirals():
    """Example 3: Spiral Dataset (Challenging)"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Spiral Dataset")
    print("="*60)
    
    # Generate dataset
    X, y = generate_spiral_dataset(n_samples=200, noise=0.15)
    
    # Train network with more hidden units
    nn = NeuralNetwork(input_size=2, hidden_size=20, output_size=1)
    nn.train(X, y, epochs=5000, learning_rate=0.1, print_every=1000)
    
    # Visualize
    plot_decision_boundary(nn, X, y, "Spiral Dataset - Decision Boundary")
    nn.plot_loss_history()
    
    # Accuracy
    predictions = nn.predict(X)
    predicted_labels = (predictions > 0.5).astype(float)
    accuracy = np.mean(predicted_labels == y)
    print(f"\nAccuracy: {accuracy*100:.2f}%")


def example_4_hyperparameter_tuning():
    """Example 4: Hyperparameter Tuning"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Hyperparameter Tuning")
    print("="*60)
    
    # Generate dataset
    X, y = generate_circle_dataset(n_samples=300, noise=0.05)
    
    # Compare learning rates
    print("\n--- Comparing Learning Rates ---")
    compare_learning_rates(X, y)
    
    # Compare hidden sizes
    print("\n--- Comparing Hidden Layer Sizes ---")
    compare_hidden_sizes(X, y)


def example_5_gradient_descent_variants():
    """Example 5: Gradient Descent Variants"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Gradient Descent Variants")
    print("="*60)
    
    # Generate dataset
    X, y = generate_circle_dataset(n_samples=200, noise=0.05)
    
    # Compare variants
    demonstrate_stochastic_vs_batch(X, y)


def example_6_regression():
    """Example 6: Regression Problem"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Regression Problem")
    print("="*60)
    
    # Generate regression data
    np.random.seed(42)
    X = np.random.randn(2, 200) * 2
    y = np.sin(X[0, :]) + 0.5 * X[1, :] + np.random.randn(200) * 0.1
    y = y.reshape(1, -1)
    
    # Train network
    nn = NeuralNetwork(input_size=2, hidden_size=10, output_size=1)
    nn.train(X, y, epochs=3000, learning_rate=0.05, print_every=500)
    
    # Visualize
    plot_3d_surface(nn, X, y, "Regression Problem - Learned Function")
    nn.plot_loss_history()
    
    # Compute R-squared
    predictions = nn.predict(X)
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"\nR-squared: {r_squared:.4f}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CMPE 257 HW1: Neural Network Examples")
    print("="*60)
    
    # Run all examples
    # Uncomment the examples you want to run
    
    example_1_xor()
    # example_2_circles()
    # example_3_spirals()
    # example_4_hyperparameter_tuning()
    # example_5_gradient_descent_variants()
    # example_6_regression()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
