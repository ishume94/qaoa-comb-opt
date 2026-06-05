import time
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_aer.primitives import EstimatorV2 as EstimatorSim
from qiskit_aer.primitives import SamplerV2 as SamplerSim
from scipy.optimize import minimize
import numpy as np

def train_qaoa_circuit(backend, ansatz, hamiltonian, init_params, simulator = False, shots=10000):
    """
    Train a QAOA circuit to obrain optimal parameters for the given Hamiltonian.
    
    Parameters:
    -----------
    backend : IBMBackend
        The quantum backend to use for computation
    ansatz : QAOAAnsatz
        The QAOA ansatz circuit to optimize
    hamiltonian : SparsePauliOp
        The cost Hamiltonian for the optimization problem
    init_params : list[float]
        Initial parameters for the QAOA circuit (length should be 2*p where p is the number of QAOA layers
    simulator : bool
        Whether to use a simulator (EstimatorSim) or real quantum hardware (Estimator) for optimization
    shots : int
        Number of shots for each objective function evaluation (default: 10000)   

    
    Returns:
    --------
    dict : Dictionary containing:
        - 'result': scipy.optimize.OptimizeResult
        - 'objective_func_vals': list of cost values during optimization
        - 'objective_func_times': list of times for each objective function evaluation
        - 'total_quantum_seconds': total quantum time used
        - 'wall_clock_time': elapsed wall-clock time
        - 'start_time': start datetime string
        - 'end_time': end datetime string
    """    
    objective_func_vals = []  # Track cost values
    total_quantum_seconds = 0  # Track total quantum time
    objective_func_times = []  # Track time for each objective function evaluation
    qaoa_layers = len(init_params) // 2 
    def cost_func_estimator(params):
        nonlocal total_quantum_seconds
        
        # Start timing for this objective function evaluation
        obj_start_time = time.time()
        
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])
        
        results = job.result()[0]
        cost = results.data.evs
        
        # End timing for this objective function evaluation
        obj_end_time = time.time()
        obj_elapsed_time = obj_end_time - obj_start_time
        objective_func_times.append(obj_elapsed_time)
        
        # Track quantum time used by this job
        try:
            if not simulator:
                job_quantum_seconds = job.metrics()['usage']['quantum_seconds']
                total_quantum_seconds += job_quantum_seconds
                print(f"Job {job.job_id()}: {job_quantum_seconds} quantum seconds")
        except Exception as e:
            print(f"Could not retrieve quantum time for job {job.job_id()}: {e}")
        
        if simulator:
            print(f"Objective function evaluation {len(objective_func_vals)}: {obj_elapsed_time:.4f} seconds")
        
        objective_func_vals.append(cost)
        return cost
   
    # Start timing
    start_time = time.time()
    start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f"Optimization started at: {start_datetime}\n")

    if simulator:
        coupling_map = backend.configuration().coupling_map
        estimator = EstimatorSim(options={"backend_options": {"method": "matrix_product_state","coupling_map":coupling_map}})
        estimator.options.default_shots = shots
        
        result = minimize(cost_func_estimator,
            init_params,
            method="COBYLA",
            tol=1e-2,)
        
        print(f"\Training result: {result}\n")
        
    else:
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = shots
            
            # Set error suppression/mitigation options
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            estimator.options.twirling.enable_gates = True
            estimator.options.twirling.num_randomizations = "auto"
            
            result = minimize(cost_func_estimator,
                init_params,
                method="COBYLA",
                tol=1e-2,)
            print(f"Training result: {result}\n")
    # End timing
    end_time = time.time()
    end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"Optimization completed at: {end_datetime}")
    print(f"Wall-clock execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Total quantum time used: {total_quantum_seconds:.2f} seconds ({total_quantum_seconds/60:.2f} minutes)")
    print(f"{'='*60}")
    
    return {
        'result': result,
        'objective_func_vals': objective_func_vals,
        'objective_func_times': objective_func_times,
        'total_quantum_seconds': total_quantum_seconds,
        'wall_clock_time': elapsed_time,
        'start_time': start_datetime,
        'end_time': end_datetime
    }


def sample_qaoa_circuit(backend, circuit, graph, simulator = False, num_shots=100000):
    """
    Run QAOA sampling on an optimized circuit to get measurement results.
    
    Parameters:
    -----------
    backend : IBMBackend
        The quantum backend to use for computation
    circuit : QuantumCircuit
        The optimized QAOA circuit to sample from
    graph : NetworkX Graph
        The graph for extracting bitstring length
    simulator : bool
        Whether to use a simulator (SamplerSim) or real quantum hardware (Sampler) for sampling
    num_shots : int
        Number of shots for sampling (default: 10000)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'counts_int': measurement counts as integers
        - 'counts_bin': measurement counts as binary strings
        - 'distribution_int': normalized distribution (integers)
        - 'distribution_bin': normalized distribution (binary strings)
        - 'most_likely_bitstring': the most likely bitstring result
        - 'quantum_seconds': quantum time used for sampling
        - 'job_time': time taken for the sampling job (when simulator=True)
        - 'wall_clock_time': elapsed wall-clock time
        - 'start_time': start datetime string
        - 'end_time': end datetime string
    """
    
    # Create sampler with options
    if simulator:
        coupling_map = backend.configuration().coupling_map
        sampler=SamplerSim(options={"backend_options": {"method": "matrix_product_state",
                                 "coupling_map":coupling_map}})
        sampler.options.default_shots = num_shots
    else:
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = num_shots
        
        # Set error suppression/mitigation options
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = False
    
    # Start timing
    sampler_start_time = time.time()
    sampler_start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sampler_start_time))
    print(f"Sampling started at: {sampler_start_datetime}\n")
    
    # Run sampler
    pub = (circuit, )
    job_start_time = time.time()
    job = sampler.run([pub], shots=int(num_shots))
    job_end_time = time.time()
    job_elapsed_time = job_end_time - job_start_time
    
    if simulator:
        print(f"Sampling job time: {job_elapsed_time:.4f} seconds")
    
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    
    # Normalize distributions
    distribution_int = {key: val/shots for key, val in counts_int.items()}
    distribution_bin = {key: val/shots for key, val in counts_bin.items()}
    print(distribution_bin)
    
    # Track sampler job quantum time
    quantum_seconds = 0
    if not simulator:
        try:
            quantum_seconds = job.metrics()['usage']['quantum_seconds']
            print(f"\nSampler Job {job.job_id()}: {quantum_seconds:.2f} quantum seconds")
        except Exception as e:
            print(f"Could not retrieve quantum time for sampler job {job.job_id()}: {e}")
    
    # End timing
    sampler_end_time = time.time()
    sampler_end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sampler_end_time))
    sampler_elapsed_time = sampler_end_time - sampler_start_time
    
    print(f"\nSampling completed at: {sampler_end_datetime}")
    print(f"Sampler wall-clock time: {sampler_elapsed_time:.2f} seconds")
    print(f"Sampler quantum time: {quantum_seconds:.2f} seconds\n")
    
    # Auxiliary function to convert integer to bitstring
    def to_bitstring(integer, num_bits):
        result = np.binary_repr(integer, width=num_bits)
        return [int(digit) for digit in result]
    
    # Find most likely bitstring
    keys = list(distribution_int.keys())
    values = list(distribution_int.values())
    most_likely = keys[np.argmax(np.abs(values))]
    most_likely_bitstring = to_bitstring(most_likely, len(graph))
    most_likely_bitstring.reverse()
    
    print("Result bitstring:", most_likely_bitstring)
    
    return {
        'counts_int': counts_int,
        'counts_bin': counts_bin,
        'distribution_int': distribution_int,
        'distribution_bin': distribution_bin,
        'most_likely_bitstring': most_likely_bitstring,
        'quantum_seconds': quantum_seconds,
        'job_time': job_elapsed_time,
        'wall_clock_time': sampler_elapsed_time,
        'start_time': sampler_start_datetime,
        'end_time': sampler_end_datetime
    }