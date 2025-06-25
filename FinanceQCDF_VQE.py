import numpy as np
import matplotlib.pyplot as plt
from qibo import Circuit, gates, hamiltonians, models, set_backend

def create_ansatz(n_qubits, depth):
    """
    Creates the VQE ansatz circuit.
    
    Args:
        n_qubits (int): The number of qubits for the circuit.
        depth (int): The number of layers in the ansatz.

    Returns:
        A qibo.Circuit object representing the ansatz.
    """
    circuit = Circuit(n_qubits)
    # Start with a superposition of all possible states
    circuit.add(gates.H(q) for q in range(n_qubits))
    # Add layers of trainable rotation gates and entangling CNOT gates
    for _ in range(depth):
        for q_idx in range(n_qubits):
            # The 'theta' is a placeholder for a trainable parameter
            circuit.add(gates.RY(q_idx, theta=0, trainable=True))
        for q_idx in range(n_qubits - 1):
            circuit.add(gates.CNOT(q_idx, q_idx + 1))
    return circuit

def main():
    """
    Main function to run the VQE portfolio optimization.
    """
    print("Starting VQE Portfolio Optimization...")

    # --- 1. Backend Configuration ---
    # We set a backend for Qibo to perform calculations.
    # 'qibojit' uses just-in-time compilation for performance.
    # We include a fallback to the standard 'numpy' backend if 'qibojit' is not available.
    try:
        # Corrected backend call for broader version compatibility
        set_backend("qibojit")
        print("Using 'qibojit' backend for high performance.")
    except Exception as e:
        print(f"Could not set 'qibojit' backend due to: {e}")
        print("Falling back to 'numpy' backend.")
        set_backend("numpy")

    # --- 2. Problem Definition ---
    # We define the assets in our portfolio and their mock financial data.
    # In a real-world scenario, this data would come from historical market analysis.
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    n_qubits = len(assets)
    print(f"Optimizing a portfolio of {n_qubits} assets: {assets}")

    # mu: Expected annual returns for each asset.
    mu = np.array([0.25, 0.22, 0.28, 0.32, 0.45, 0.38, 0.35])
    # sigma: Covariance matrix representing the risk and correlation between assets.
    sigma = np.array([
        [0.08, 0.02, 0.01, 0.03, 0.05, 0.04, 0.02],
        [0.02, 0.07, 0.015, 0.025, 0.03, 0.03, 0.01],
        [0.01, 0.015, 0.09, 0.03, 0.04, 0.025, 0.015],
        [0.03, 0.025, 0.03, 0.1, 0.05, 0.04, 0.03],
        [0.05, 0.03, 0.04, 0.05, 0.2, 0.06, 0.04],
        [0.04, 0.03, 0.025, 0.04, 0.06, 0.15, 0.035],
        [0.02, 0.01, 0.015, 0.03, 0.04, 0.035, 0.1]
    ])

    # --- 3. QUBO Formulation ---
    # We formulate the problem as a Quadratic Unconstrained Binary Optimization (QUBO).
    # The goal is to minimize an objective function that balances risk and return.
    q = 0.5       # Risk appetite factor (0=risk-averse, 1=return-focused)
    budget = 3    # The desired number of assets to have in the portfolio
    penalty = 10  # A large penalty for violating the budget constraint

    # Start with the core risk/return part of the QUBO matrix
    qubo_matrix = q * sigma
    np.fill_diagonal(qubo_matrix, np.diag(qubo_matrix) - (1 - q) * mu)

    # Add the budget constraint penalty to the QUBO matrix
    # This penalizes solutions that don't have exactly 'budget' assets.
    for i in range(n_qubits):
        qubo_matrix[i, i] += penalty * (1 - 2 * budget)
        for j in range(i + 1, n_qubits):
            qubo_matrix[i, j] += 2 * penalty
            qubo_matrix[j, i] += 2 * penalty

    # --- 4. Hamiltonian Formulation ---
    # The QUBO problem is converted to an Ising Hamiltonian.
    # The mapping from a binary variable x_i to a spin variable Z_i is x_i = (1 - Z_i) / 2.
    from qibo import symbols

    # The objective function is x^T * Q * x.
    symbolic_hamiltonian = 0

    # Convert linear terms (diagonal of QUBO)
    for i in range(n_qubits):
        if qubo_matrix[i, i] != 0:
            symbolic_hamiltonian += qubo_matrix[i, i] * (1 - symbols.Z(i)) / 2

    # Convert quadratic terms (off-diagonal of QUBO)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if qubo_matrix[i, j] != 0:
                coefficient = qubo_matrix[i, j]
                symbolic_hamiltonian += coefficient * (1 - symbols.Z(i) - symbols.Z(j) + symbols.Z(i) * symbols.Z(j)) / 2
    
    # Create the Hamiltonian object from the symbolic form.
    hamiltonian = hamiltonians.SymbolicHamiltonian(symbolic_hamiltonian)


    # --- 5. Quantum Circuit (Ansatz) ---
    # We create a parameterized quantum circuit (an "ansatz") that will be trained
    # to prepare the ground state of the Hamiltonian. This circuit is used for the VQE.
    depth = 2
    vqe_circuit = create_ansatz(n_qubits, depth)
    
    # Get the number of trainable parameters from the variational circuit
    num_params = len(vqe_circuit.get_parameters())

    print("\nQuantum Circuit Ansatz (for VQE):")
    print(vqe_circuit.draw())
    print(f"Number of trainable parameters: {num_params}")

    # --- 6. VQE Optimization ---
    # The Variational Quantum Eigensolver (VQE) algorithm iteratively adjusts the
    # circuit parameters to minimize the expected value of the Hamiltonian.
    # We use the circuit without measurements for the VQE optimization.
    print("\nRunning VQE...")
    vqe = models.VQE(vqe_circuit, hamiltonian)

    # Set a seed for reproducibility of the random initial parameters
    np.random.seed(42)
    initial_parameters = np.random.uniform(0, 2 * np.pi, num_params)
    
    # Run the optimization. We capture the entire result tuple first to avoid
    # unpacking errors, as different Qibo versions return different numbers of items.
    result_tuple = vqe.minimize(initial_parameters, method='BFGS')
    min_energy = result_tuple[0]
    final_params = result_tuple[1]
    
    print("VQE optimization finished.")
    print(f"Minimum energy (objective value): {min_energy:.5f}")

    # --- 7. Result Interpretation ---
    # To get the final outcome frequencies, we create a new circuit.
    # This is because the original `vqe_circuit` was executed and is now locked.
    measurement_circuit = create_ansatz(n_qubits, depth)
    
    # We set the optimal parameters found by VQE.
    measurement_circuit.set_parameters(final_params)
    
    # Now we add measurement gates to this fresh circuit.
    measurement_circuit.add(gates.M(*range(n_qubits)))
    
    # Execute the circuit 1000 times to get a distribution of outcomes.
    measurement = measurement_circuit(nshots=1000)
    frequencies = measurement.frequencies(binary=True)
    
    # The most frequent measurement corresponds to the best solution found
    # Handle the case where frequencies might be empty
    if not frequencies:
        print("Error: No measurement outcomes were recorded.")
        return

    best_bitstring = max(frequencies, key=frequencies.get)
    selected_assets = [assets[i] for i, bit in enumerate(best_bitstring) if bit == '1']

    print("\n" + "="*40)
    print("           OPTIMAL PORTFOLIO")
    print("="*40)
    print(f"Selected Assets: {selected_assets}")
    print(f"Binary Representation: |{best_bitstring}>")
    print(f"Probability of this outcome: {frequencies[best_bitstring] / 1000:.2%}")
    print("="*40 + "\n")

    # --- 8. Visualization ---
    # We plot the probabilities of the top outcomes to see how confident the
    # algorithm is in its choice.
    top_k = 10
    sorted_freq = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
    labels = [x[0] for x in sorted_freq[:top_k]]
    values = [x[1] / 1000 for x in sorted_freq[:top_k]]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, values, color='royalblue', alpha=0.8)
    # Highlight the best solution
    if labels:
        bars[0].set_color('goldenrod')
        bars[0].set_edgecolor('black')

    plt.title(f"Top {top_k} Portfolio Probabilities from VQE", fontsize=16)
    plt.xlabel("Portfolio State (Bitstring)", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    
    if values:
        plt.ylim(0, max(values) * 1.15)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Add probability values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.1%}', ha='center', va='bottom')

    plt.show()

if __name__ == "__main__":
    main()
