import torch
from qibo import Circuit, gates, hamiltonians
from qiboml.models.encoding import PhaseEncoding
from qiboml.models.decoding import Expectation
from qiboml.interfaces.pytorch import QuantumModel

# define the encoding
encoding = PhaseEncoding(nqubits=3)
# define the decoding given an observable
observable = hamiltonians.Z(nqubits=3)
decoding = Expectation(nqubits=3, observable=observable)
# build the computation circuit
circuit = Circuit(3)
circuit.add((gates.RY(i, theta=0.4) for i in range(3)))
circuit.add((gates.RZ(i, theta=0.2) for i in range(3)))
circuit.add((gates.H(i) for i in range(3)))
circuit.add((gates.CNOT(0,1), gates.CNOT(0,2)))
circuit.draw()
# join everything together through the torch interface
quantum_model = QuantumModel(
   circuit_structure=[encoding, circuit],
   decoding=decoding,
)
# run on some data
data = torch.randn(3)
outputs = quantum_model(data)
