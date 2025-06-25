import numpy as np
import matplotlib.pyplot as plt
from qibo import Circuit, gates, hamiltonians, models

# --- 1. Problem Definition: Portfolio Optimization ---
# We model an investor's decision for a portfolio of the "Magnificent Seven".
assets =
n_qubits = len(assets)

# Mock financial data (in a real scenario, this would be historical data)
mu = np.array([0.25, 0.22, 0.28, 0.32, 0.40, 0.27, 0.35])
sigma = np.array([0.08, 0.02, 0.01, 0.03, 0.04, 0.03, 0.02],
    [0.02, 0.07, 0.015, 0.025, 0.035, 0.02, 0.03],
    [0.01, 0.015, 0.09, 0.03, 0.025, 0.02, 0.01],
    [0.03, 0.025, 0.03, 0.10, 0.045, 0.03, 0.04],
    [0.04, 0.035, 0.025, 0.045, 0.12, 0.04, 0.05],
    [0.03, 0.02, 0.02, 0.03, 0.04, 0.11, 0.03],
    [0.02, 0.03, 0.01, 0.04, 0.05, 0.03, 0.10])

# Cognitive parameters
q = 0.5      # Risk appetite factor
budget = 3   # Desired number of assets
penalty = 10 # Penalty for violating the budget

# --- 2. QUBO and Hamiltonian Formulation ---
# Build the QUBO matrix from financial and cognitive parameters
qubo_matrix = np.zeros((n_qubits, n_qubits))
qubo_matrix += q * sigma
np.fill_diagonal(qubo_matrix, np.diag(qubo_matrix) - (1 - q) * mu)

# Add budget constraint penalty
for i in range(n_qubits):
    qubo_matrix[i, i] += penalty * (1 - 2 * budget)
    for j in range(i + 1, n_qubits):
        qubo_matrix[i, j] += 2 * penalty
        qubo_matrix[j, i] += 2 * penalty

# Convert QUBO to an Ising Hamiltonian matrix
# NOTE: This explicit matrix construction becomes memory-intensive for large n_qubits
hamiltonian_matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
I = np.eye(2)
Z = np.array([, [0, -1]])

for i in range(n_qubits):
    for j in range(i, n_qubits):
        if qubo_matrix[i, j]!= 0:
            ops = [I] * n_qubits
            ops[i] = Z
            if i!= j:
                ops[j] = Z
            
            term_matrix = ops
            for k in range(1, n_qubits):
                term_matrix = np.kron(term_matrix, ops[k])
            
            # Map QUBO terms to Ising Hamiltonian terms
            if i == j:
                # Linear term mapping
                hamiltonian_matrix += (0.5 * qubo_matrix[i, i]) * (np.eye(2**n_qubits) - term_matrix)
            else:
                # Quadratic term mapping
                 hamiltonian_matrix += (0.25 * qubo_matrix[i, j]) * term_matrix

# --- 3. Circuit Construction (Ansatz) ---
depth = 2
circuit = Circuit(n_qubits)
circuit.add(gates.H(q) for q in range(n_qubits))
for _ in range(depth):
    for q_idx in range(n_qubits):
        circuit.add(gates.RY(q_idx, theta=0))
    for q_idx in range(n_qubits - 1):
        circuit.add(gates.CNOT(q_idx, q_idx + 1))

print("Quantum Circuit (Ansatz) Diagram:")
print(circuit.draw())
print("-" * 30)

# --- 4. VQE Optimization ---
hamiltonian = hamiltonians.Hamiltonian(n_qubits, hamiltonian_matrix)
vqe = models.VQE(circuit, hamiltonian)
n_params = len(circuit.get_parameters())
initial_parameters = np.random.uniform(0, 2 * np.pi, n_params)
print(f"Number of trainable parameters: {n_params}\n")

print("Running VQE to find the optimal portfolio...")
# Note: The return format of minimize can vary. We handle a common tuple format.
min_energy, final_params, _ = vqe.minimize(initial_parameters)
print("Optimization finished.")
print("-" * 30)

# --- 5. Display and Interpret Results ---
print(f"Minimum objective value found: {min_energy}")
circuit.set_parameters(final_params)
final_state = circuit(nshots=1).state()
probabilities = np.abs(final_state)**2
best_bitstring = format(np.argmax(probabilities), f'0{n_qubits}b')
selected_assets = [assets[i] for i, bit in enumerate(best_bitstring) if bit == '1']

print("-" * 30)
print(f"Optimal portfolio selection found: {selected_assets}")
print(f"(Corresponds to quantum state |{best_bitstring}>)")
print("-" * 30)

# --- 6. Visualization ---
print("Generating visualizations...")

# Create the figure and subplots
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('VQE Portfolio Optimization Results', fontsize=16)

# Plot 1: Covariance Matrix Heatmap
heatmap = ax.imshow(sigma, cmap='viridis')
ax.set_title('Asset Covariance Matrix (Risk)', fontsize=14)
ax.set_xticks(np.arange(len(assets)))
ax.set_yticks(np.arange(len(assets)))
ax.set_xticklabels(assets, rotation=45, ha='right')
ax.set_yticklabels(assets)
plt.colorbar(heatmap, ax=ax)

# Plot 2: Final State Probabilities
top_k = 15
top_indices = np.argsort(probabilities)[::-1][:top_k]
top_probs = probabilities[top_indices]
top_labels = [format(i, f'0{n_qubits}b') for i in top_indices]

ax.[33]bar(top_labels, top_probs, color='skyblue')
ax.[33]set_title(f'Top {top_k} Portfolio Probabilities', fontsize=14)
ax.[33]set_ylabel('Probability')
ax.[33]set_xlabel('Portfolio Combination (Quantum State)')
ax.[33]set_xticklabels(top_labels, rotation=75, ha='right')
ax.[33]grid(axis='y', linestyle='--', alpha=0.7)

# Highlight the best portfolio
try:
    best_bar_index = list(top_labels).index(best_bitstring)
    ax.[33]patches[best_bar_index].set_facecolor('royalblue')
    ax.[33]patches[best_bar_index].set_edgecolor('black')
    ax.[33]legend()
except ValueError:
    print("Note: The optimal portfolio is not among the top 15 most probable states shown.")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


