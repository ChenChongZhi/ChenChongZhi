import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# It's good practice to set the backend at the beginning
import qibo
qibo.set_backend("pytorch")

from qibo import gates
from qiboml.models.encoding import PhaseEncoding
from qiboml.models.decoding import Expectation
from qiboml.interfaces.pytorch import QuantumModel
from qiboml.operations.differentiation import PSR

# --- 1. Define Target Function and Data ---
def target_function(x):
    return torch.sin(2 * x) * torch.cos(0.5 * x)

# Generate training data
x_train = torch.linspace(0, 2 * np.pi, 50).view(-1, 1)
y_train = target_function(x_train)

# --- 2. Define Quantum Circuit Components ---
nqubits = 1
nlayers = 5

# Encoding Layer: Encodes data via RX rotation
encoding_circ = PhaseEncoding(nqubits=nqubits, encoding_gate=gates.RX)

# Trainable Layer (Ansatz)
def trainable_layer():
    # Each call creates a new circuit with distinct trainable parameters
    circuit = qibo.Circuit(nqubits)
    # Using Parameter objects is more explicit for Qibo's PyTorch interface
    theta1 = qibo.gates.Parameter(trainable=True, initial_value=np.random.randn())
    theta2 = qibo.gates.Parameter(trainable=True, initial_value=np.random.randn())
    circuit.add(gates.RY(0, theta=theta1))
    circuit.add(gates.RZ(0, theta=theta2))
    return circuit

# Assemble the full circuit structure
# FIX 1: Initialize circuit_structure as an empty list
circuit_structure = []
for _ in range(nlayers):
    # The data is re-uploaded at each layer
    circuit_structure.extend([encoding_circ, trainable_layer()])

# --- 3. Build the QML Model with Qiboml ---
# Decoding: Measure expectation of Pauli Z
decoding_circ = Expectation(nqubits=nqubits)
# Differentiation: Use the hardware-compatible Parameter-Shift Rule
diff_rule = PSR()

# Assemble the model using the PyTorch interface
qml_model = QuantumModel(
    circuit_structure=circuit_structure,
    decoding=decoding_circ,
    differentiation=diff_rule,
)

# --- 4. Classical Training Loop ---
optimizer = optim.Adam(qml_model.parameters(), lr=0.1)
criterion = nn.MSELoss()
epochs = 150

# FIX 2: Initialize loss_history as an empty list
loss_history = []

print("Starting QML model training...")
for epoch in range(epochs):
    # Qiboml models often process one sample at a time
    # This list comprehension feeds each training point to the model
    y_pred = torch.stack([qml_model(x) for x in x_train]).squeeze()
    
    # Calculate the loss between predicted and actual values
    loss = criterion(y_pred, y_train.squeeze())
    
    # Standard PyTorch backpropagation and optimization steps
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
print("Training finished.")

# --- 5. Visualize Results ---
plt.figure(figsize=(12, 8))

# Plot the original function and the model's predictions
ax1 = plt.subplot(2, 1, 1)
ax1.plot(x_train.numpy(), y_train.numpy(), label='Target Function', linewidth=3, color='blue', alpha=0.7)
with torch.no_grad():
    predictions = torch.stack([qml_model(x) for x in x_train]).squeeze()
ax1.plot(x_train.numpy(), predictions.numpy(), label='QML Model Prediction', linestyle='--', marker='o', markersize=4, color='red')
ax1.set_title('QML Model Fit using Qibo and PyTorch', fontsize=14)
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend()
ax1.grid(True)

# Plot the training loss over epochs
ax2 = plt.subplot(2, 1, 2)
ax2.plot(range(epochs), loss_history, label='Training Loss', color='green')
ax2.set_title('Training Loss Over Epochs', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss')
ax2.legend()
ax2.grid(True)
ax2.set_yscale('log') # Log scale is often useful for viewing loss convergence

plt.tight_layout()
plt.show()



