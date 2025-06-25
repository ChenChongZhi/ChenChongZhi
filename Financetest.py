from qibo import Circuit, gates, hamiltonians, models
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Problem Definition: Portfolio Optimization ---
# We want to select the best portfolio from the 7 major tech stocks.
# The VQE algorithm will find the ground state of a Hamiltonian that
# represents the optimal trade-off between return and risk.

# Define the assets (The full "Magnificent Seven")
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
n_qubits = len(assets) # Each qubit represents a stock (1=select, 0=don't select)

# Mock financial data for 7 assets
mu = np.array([0.25, 0.22, 0.28, 0.32, 0.40, 0.27, 0.35])
sigma = np.array([
    [0.08, 0.02, 0.01, 0.03, 0.04, 0.03, 0.02],
    [0.02, 0.07, 0.015, 0.025, 0.035, 0.02, 0.03],
    [0.01, 0.015, 0.09, 0.03, 0.025, 0.02, 0.01],
    [0.03, 0.025, 0.03, 0.10, 0.045, 0.03, 0.04],
    [0.04, 0.035, 0.025, 0.045, 0.12, 0.04, 0.05],
    [0.03, 0.02, 0.02, 0.03, 0.04, 0.11, 0.03],
    [0.02, 0.03, 0.01, 0.04, 0.05, 0.03, 0.10]
])

# Optimization parameters
q = 0.5      # Risk appetite factor (0 to 1). Higher q = more focus on risk reduction.
budget = 3   # The number of assets we want to have in our portfolio.
penalty = 10 # A penalty factor to enforce the budget constraint.

# --- 2. QUBO and Hamiltonian Formulation ---
# We map the financial problem to a Quadratic Unconstrained Binary
# Optimization (QUBO) matrix, which is then converted to an Ising Hamiltonian.

# Build the QUBO matrix
qubo_matrix = np.zeros((n_qubits, n_qubits))
qubo_matrix += q * sigma # Add risk component
np.fill_diagonal(qubo_matrix, np.diag(qubo_matrix) - (1 - q) * mu) # Add return component

# Add budget penalty component
for i in range(n_qubits):
    qubo_matrix[i, i] += penalty * (1 - 2 * budget)
    for j in range(i + 1, n_qubits):
        qubo_matrix[i, j] += 2 * penalty
        qubo_matrix[j, i] += 2 * penalty

# Convert QUBO to an Ising Hamiltonian matrix.
# This requires building the matrix for the full 2^n_qubits space.
# NOTE: This can become very memory-intensive for n_qubits > 10.
hamiltonian_matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
I = np.eye(2)
Z = np.array([[1, 0], [0, -1]])

for i in range(n_qubits):
    for j in range(i, n_qubits):
        if qubo_matrix[i, j] != 0:
            ops = [I] * n_qubits
            ops[i] = Z
            if i != j:
                ops[j] = Z
            
            term_matrix = ops[0]
            for k in range(1, n_qubits):
                term_matrix = np.kron(term_matrix, ops[k])
            
            if i == j:
                hamiltonian_matrix += (0.5 * qubo_matrix[i, j]) * (np.eye(2**n_qubits) - term_matrix)
            else:
                 hamiltonian_matrix += (0.25 * qubo_matrix[i, j]) * term_matrix


# --- 3. Circuit Construction (Ansatz) ---
depth = 2
circuit = Circuit(n_qubits)
circuit.add(gates.H(q) for q in range(n_qubits))
for _ in range(depth):
    for q in range(n_qubits):
        circuit.add(gates.RY(q, theta=0))
    for q in range(n_qubits - 1):
        circuit.add(gates.CNOT(q, q + 1))

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
min_energy, final_params, _ = vqe.minimize(initial_parameters)
print("Optimization finished.")
print("-" * 30)

# --- 5. Display and Interpret Results ---
print(f"Minimum energy (objective value) found: {min_energy}")
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
# Add a main title for the entire figure
fig.suptitle('VQE Portfolio Optimization Results', fontsize=16)

# Plot 1: Covariance Matrix Heatmap
heatmap = ax[0].imshow(sigma, cmap='viridis')
ax[0].set_title('Asset Covariance Matrix (Risk)', fontsize=14)
ax[0].set_xticks(np.arange(len(assets)))
ax[0].set_yticks(np.arange(len(assets)))
ax[0].set_xticklabels(assets, rotation=45, ha='right')
ax[0].set_yticklabels(assets)
plt.colorbar(heatmap, ax=ax[0])

# Plot 2: Final State Probabilities
# We only plot the top 15 most probable states for clarity
top_k = 15
top_indices = np.argsort(probabilities)[::-1][:top_k]
top_probs = probabilities[top_indices]
top_labels = [format(i, f'0{n_qubits}b') for i in top_indices]

ax[1].bar(top_labels, top_probs, color='skyblue')
ax[1].set_title(f'Top {top_k} Portfolio Probabilities', fontsize=14)
ax[1].set_ylabel('Probability')
ax[1].set_xlabel('Portfolio Combination (Quantum State)')
ax[1].set_xticklabels(top_labels, rotation=75, ha='right')
ax[1].grid(axis='y', linestyle='--', alpha=0.7)

# Highlight the best portfolio
# Use a try-except block in case the best state is not in the top_k
try:
    best_bar_index = list(top_labels).index(best_bitstring)
    ax[1].patches[best_bar_index].set_facecolor('royalblue')
    ax[1].patches[best_bar_index].set_edgecolor('black')
    ax[1].legend(['Other States', 'Optimal State'])
except ValueError:
    print("Note: The optimal portfolio is not among the top 15 most probable states shown.")


plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()



