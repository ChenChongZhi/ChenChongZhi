# Import necessary libraries
import os
import matplotlib
matplotlib.use("TkAgg")  # Ensure GUI works for plots
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit.visualization import plot_histogram
from qiskit_aqua.algorithms import QAOA
from qiskit.circuit.library import TwoLocal
import numpy as np

# STEP 1: Securely Load IBM Quantum API Key
api_key = os.getenv("IBM_QUANTUM_API_KEY")
if not api_key:
    print("❌ No API key found. Set it as an environment variable or in a .env file.")
    exit()

# Authenticate with IBM Quantum
try:
    service = QiskitRuntimeService(channel="ibm_quantum", token=api_key)
    print("✅ Successfully connected to IBM Quantum.")
except Exception as e:
    print(f"❌ Error connecting to IBM Quantum: {e}")
    exit()

# STEP 2: Select Quantum Backend
try:
    backends = service.backends()
    print("✅ Available IBM Quantum Computers:", [b.name for b in backends])
    backend_name = "ibmq_qasm_simulator" if "ibmq_qasm_simulator" in [b.name for b in backends] else backends[0].name
    backend = service.backend(backend_name)
    print(f"✅ Using backend: {backend.name}")
except Exception as e:
    print(f"❌ No available IBM Quantum backends: {e}")
    exit()

# STEP 3: Quantum Simulation - Zero-Point Energy Effects on Qubits
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
print("Quantum Circuit:")
print(qc.draw())

# STEP 4: Transpile the circuit
try:
    qc_transpiled = transpile(qc, backend)
    print("✅ Quantum circuit transpiled successfully.")
except Exception as e:
    print(f"❌ Error transpiling circuit: {e}")
    exit()

# STEP 5: Execute the quantum circuit
try:
    with Session(service=service, backend=backend) as session:
        sampler = Sampler(session=session)
        job = sampler.run(qc_transpiled)
        print("✅ Job submitted! Job ID:", job.job_id())
except Exception as e:
    print(f"❌ Error executing job: {e}")
    exit()

# STEP 6: Monitor the job status
from qiskit_ibm_runtime import job_monitor
try:
    job_monitor(job)
except Exception as e:
    print(f"❌ Error monitoring job: {e}")

# STEP 7: Retrieve and display the results
try:
    result = job.result()
    counts = result.quasi_dists[0]  # Updated for IBM Quantum API
    print("✅ Quantum Measurement Results:", counts)
except Exception as e:
    print(f"❌ Error retrieving results: {e}")
    exit()

# STEP 8: Plot the results
print("🔍 Debug: Quantum Measurement Counts =", counts)
try:
    fig = plot_histogram(counts)
    plt.title("Quantum Measurement Outcomes")
    plt.show()
except Exception as e:
    print(f"❌ Error plotting results: {e}")

# STEP 9: Smart City AI Optimization - Traffic Flow
# Simulating a quantum optimization problem for traffic management
traffic_problem_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # Graph of roads & congestion
qaoa = QAOA(reps=3)
quantum_traffic_model = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')

# Simulating AI-based Traffic Optimization with Quantum Computing
try:
    print("✅ Running Quantum AI for Smart City Traffic Optimization...")
    traffic_result = qaoa.compute_minimum_eigenvalue(traffic_problem_matrix)
    print("✅ Traffic Optimization Completed:", traffic_result.optimal_value)
except Exception as e:
    print(f"❌ Error optimizing traffic: {e}")
    exit()
