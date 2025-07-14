# Neural Network Learning to Approximate a Sine Wave

This project visualizes a simple, hand-crafted neural network learning to approximate the sine function using only **NumPy** and **Matplotlib** — no external deep learning frameworks like TensorFlow or PyTorch.

## 🎯 What This Demonstrates

- How a basic feedforward neural network learns through **forward and backward propagation**
- How neural networks can be trained to approximate **nonlinear functions** like `sin(x)`
- How the shape of the learned function **evolves over time** via gradient descent
- How animation can help reveal **what's going on inside the network** as it learns

---

## 🧠 Educational Objectives

This project helps you build intuition about:
- Neural network structure (input layer → hidden layer → output)
- Activation functions (here: `tanh`) and their nonlinear role
- The loss surface and convergence through **gradient descent**
- The effect of **input normalization**, **network capacity**, and **training duration** on the learned function
- How machine learning approximates functions without needing to know their exact form (i.e., no special-case sine math — just data)

---

## 🛠️ How It Works

### 1. **Data Generation**
- Inputs (`X`) range from \( 0 \) to \( 4\pi \)
- Targets (`y_true`) are computed as \( \sin(x) \)
- Inputs are **normalized** to \([-1, 1]\) to stabilize learning

### 2. **Network Architecture**
- 1 input neuron
- 1 hidden layer with 100 `tanh` neurons
- 1 output neuron (for scalar prediction)
- Weights are initialized using **Xavier initialization** for tanh

### 3. **Training Process**
- Manual implementation of forward pass
- Loss function: **Mean Squared Error (MSE)**
- Backpropagation updates weights and biases using gradients
- Predictions at each epoch are saved for later visualization

### 4. **Visualization**
- An animated plot shows how the network's output (red line) gradually learns to match the sine curve (blue dashed line)
- Animation is built using `matplotlib.animation.FuncAnimation`

---

## ▶️ Running the Code

Ensure you have the required packages:

```bash
pip install numpy matplotlib
```

Then run:

```bash
python neural_network_sine_wave.py
```

> A window will pop up showing a red line (the network’s prediction) slowly evolving to match the sine wave (blue dashed line) as training progresses.

---

## 🧪 Experiment Ideas

Try changing the following to deepen your understanding:

| Parameter       | Effect |
|----------------|--------|
| `hidden_dim`   | Increases/decreases model capacity |
| `epochs`       | Affects convergence quality |
| `lr` (learning rate) | Higher = faster learning but may overshoot |
| Input range    | Try `[0, 2π]`, `[0, 8π]`, or noisy sine data |
| Activation     | Replace `tanh` with `ReLU` or `sigmoid` and observe behavior |

---

## 💡 Inspiration

This project was inspired by:
- Classic neural net demos from scratch
- The connection between **Taylor series approximation** and **neural networks as universal function approximators**
- A desire to understand neural networks beyond the black-box API level

---

## 📖 License

MIT License — free to use, modify, and explore.

---

## 🙌 Acknowledgments

Thanks to the educational journey driven by curiosity, persistence, and the joy of *seeing machines learn*.
