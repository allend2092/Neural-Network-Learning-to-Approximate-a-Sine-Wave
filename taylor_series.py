import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate sine wave data
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y_true = np.sin(X)

# Initialize weights and biases for a single hidden layer network
input_dim = 1
hidden_dim = 40
output_dim = 1
lr = 0.01
epochs = 2000

# Randomly initialize parameters
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1 / input_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1 / hidden_dim)
b2 = np.zeros((1, output_dim))

# Activation functions
def tanh(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1 - np.tanh(z) ** 2


# Store predictions over time for animation
history = []

# Training using manual backpropagation
for epoch in range(epochs):
    # Forward pass
    Z1 = X.dot(W1) + b1
    A1 = tanh(Z1)
    Z2 = A1.dot(W2) + b2
    y_pred = Z2

    # Loss (Mean Squared Error)
    loss = np.mean((y_pred - y_true) ** 2)

    # Backpropagation
    dZ2 = 2 * (y_pred - y_true) / len(X)
    dW2 = A1.T.dot(dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * tanh_deriv(Z1)
    dW1 = X.T.dot(dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update parameters
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    history.append(y_pred.copy())

# Animation setup
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot([], [], 'r-', linewidth=2, label='Network Prediction')
true_line, = ax.plot(X, y_true, 'b--', label='True Sine')
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Neural Network Learning to Approximate a Sine Wave")
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")
ax.legend()
text = ax.text(0.05, 0.90, '', transform=ax.transAxes)

def update(frame):
    y_pred = history[frame]
    line.set_data(X, y_pred)
    text.set_text(f"Epoch: {frame+1}")
    return line, text

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=50, blit=True)
plt.show()
