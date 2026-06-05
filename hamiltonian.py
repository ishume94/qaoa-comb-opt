import networkx as nx
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz


def build_mpc_hamiltonian(graph):
    """Create the Hamiltonian for MaxPC. Notice that this is a maximization problem.
    To convert it to a minimization problem, we can simply negate the coefficients.
    Input: graph - a NetworkX graph
    Output: SparsePauliOp representing the MaxPC Hamiltonian
    """
    pauli_list = []
    for edge in list(graph.edges()):
        paulis = ["I"] * len(graph)
        
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"
           
        coeff = 0.25*0.5
        pauli_list.append(("".join(paulis)[::-1], coeff))
        
        paulis = ["I"] * len(graph)
        paulis[edge[0]] = "Z"
        pauli_list.append(("".join(paulis)[::-1], coeff))
        paulis = ["I"] * len(graph)
        paulis[edge[1]] = "Z"
        pauli_list.append(("".join(paulis)[::-1], coeff))
        
    for vertex in list(graph.nodes()):
        paulis = ["I"] * len(graph)
        paulis[vertex] = "Z"
        coeff = -0.5*0.5
        pauli_list.append(("".join(paulis)[::-1], coeff))
      
    mpc_hamiltonian = SparsePauliOp.from_list(pauli_list)
 
    return mpc_hamiltonian

def build_mvc_hamiltonian(graph: nx.graph):
    """Create the Hamiltonian for MinVC. 
    Input: graph - a NetworkX graph
    Output: SparsePauliOp representing the MinVC Hamiltonian
    """
    pauli_list = []
    for edge in list(graph.edges()):
        paulis = ["I"] * len(graph)
        
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"
           
        coeff = 0.75*0.5

        pauli_list.append(("".join(paulis)[::-1], coeff))
        
        paulis = ["I"] * len(graph)
        paulis[edge[0]] = "Z"
        pauli_list.append(("".join(paulis)[::-1], coeff))
        paulis = ["I"] * len(graph)
        paulis[edge[1]] = "Z"
        pauli_list.append(("".join(paulis)[::-1], coeff))
        
    for vertex in list(graph.nodes()):
        paulis = ["I"] * len(graph)
        paulis[vertex] = "Z"
        coeff = -1.0*0.5
        pauli_list.append(("".join(paulis)[::-1], coeff))
   
    mvc_hamiltonian = SparsePauliOp.from_list(pauli_list)
 
    return mvc_hamiltonian


def build_candidate_circuit(graph, hamiltonian, backend, reps):
    """Build a QAOA ansatz circuit for the given graph and Hamiltonian, and transpile it for the given backend.
    Input: 
        graph - a NetworkX graph
        hamiltonian - a SparsePauliOp representing the cost Hamiltonian
        backend - an IBM backend to transpile for
        reps - number of QAOA layers (p)
    Output: a transpiled QuantumCircuit
    """
    circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=reps)

    candidate_circuit=transpile(circuit, basis_gates=backend.basis_gates)
    return candidate_circuit

