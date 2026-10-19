import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
# from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime.fake_provider import FakeTorino


class QCNN:
    def __init__(self, number_of_qubits, name="my_QCNN"):
        """
        Modular QCNN builder class.
        """
        self.num_qubits = number_of_qubits
        self.name = name
        self.circuit = QuantumCircuit(number_of_qubits)
        self.transpiled_circuit = None
        self.parameters = []
        self.layer_history = []
        self.surviving_qubits = list(range(number_of_qubits))
        self.encoding = False
        self.measurement = False
        self.measured_qubits = None
        self.aer_simulator_settings = {"device": "cpu",
                                       "method": "statevector"}
        self.noise = False
        self.noise_settings = {"from_backend": False,
                               "backend": FakeTorino(),
                               "readout": False,
                               "readout_probs": (0.01, 0.01),
                               "depolarizing": False,
                               "depolarizing_probs": {"1-q": 0.01,
                                                      "2-q": 0.01},
                               "depolarizing_gates": {"1-q": ["h"],
                                                      "2-q": ["cx"]},
                               "thermal_relaxation": False,
                               "thermal_relaxation_t1": 50e3,
                               "thermal_relaxation_t2": 70e3,
                               "thermal_relaxation_gates": {"1-q": ["h"],
                                                            "2-q": ["cx"]},
                               "thermal_relaxation_gate_exec_times": {"1-q": 50.0,
                                                                      "2-q": 300.0},
                               }


    _block_registry = {
        "default_conv": lambda params: QCNN._conv_block_default(params),
        "default_pool": lambda params: QCNN._pool_block_default(params)
    }


    # GENERAL
    def reset(self):
        """
        Clear the circuit and start from scratch.
        """
        self.circuit = QuantumCircuit(self.num_qubits)
        self.parameters = []
        self.layer_history = []
        self.surviving_qubits = list(range(self.num_qubits))
        self.encoding = False
        self.measurement = False
        self.measured_qubits = None
        return self

    def summary(self):
        """
        Print information about current QCNN status.
        """
        print(f"\n>>> QCNN model: {self.name}")
        print(f"- Number of qubits: {self.num_qubits}")
        print(f"- Layers: {len(self.layer_history)}")
        for l in self.layer_history:
            print(f"--- {l}")
        print(f"- Parameters: {len(self.parameters)}")
        print(f"- Measured qubits: {self.measured_qubits}")

    def plot_circuit_diagram(self, filename=None, dpi=300, format='png'):
        """
        Plot or save the QCNN circuit diagram.
        """
        fig = self.circuit.draw('mpl')
        if filename:
            fig.savefig(filename, dpi=dpi, format=format, bbox_inches="tight")
            plt.close(fig)
            print("\n>>> Circuit diagram saved")
        else:
            plt.show()

    def add_hadamard_layer(self, layer_name="hadamard", qubits=None):
        """
        Adds a Hadamard layer to the circuit.
        """
        if qubits is None:
            qubits = list(self.surviving_qubits)
        for q in qubits:
            self.circuit.h(q)
        self.circuit.barrier()
        self.layer_history.append(f"{layer_name}")


    # ENCODING
    def encode(self, data, method="amplitude"):
        """
        State preparation to encode classical data into quantum states.
        """
        if not self.encoding:
            encoding_circuit = QuantumCircuit(self.num_qubits)
            if method == "amplitude":
                # Flattening and converting to float64 for precision
                data_vector = np.array(data, dtype=np.float64).flatten()
                # Zero padding (if necessary)
                n_states = 2 ** self.num_qubits
                if len(data_vector) > n_states:
                    raise ValueError("The number of qubits is not sufficient to encode the whole size of data vector!")
                data_vector = np.pad(data_vector, (0, n_states - len(data_vector)))
                # Normalization
                data_vector /= np.linalg.norm(data_vector)
                # self.circuit.initialize(data_vector, range(self.num_qubits))
                encoding_circuit.initialize(data_vector, range(self.num_qubits))
            # TODO: elif method == "angle":
                # fm = ZZFeatureMap(self.num_qubits, reps=1, entanglement="linear")
                # self.circuit.compose(fm.assign_parameters(data), inplace=True)
                # encoding_circuit.compose(fm.assign_parameters(data), inplace=True)
            self.circuit = encoding_circuit.compose(self.circuit)
            self.encoding = True
            self.layer_history.insert(0, f"encoding ({method})")
        else:
            raise ValueError(f"An encoding layer has already been applied!")


    # CONVOLUTION / POOLING
    @staticmethod
    def _conv_block_default(params):
        """
        Default parameterized binary convolution unitary.
        """
        circ = QuantumCircuit(2, name="default_conv_block")
        circ.rz(-np.pi / 2, 1)
        circ.cx(1, 0)
        circ.rz(params[0], 0)
        circ.ry(params[1], 1)
        circ.cx(0, 1)
        circ.ry(params[2], 1)
        circ.cx(1, 0)
        circ.rz(np.pi / 2, 0)
        return circ

    @staticmethod
    def _pool_block_default(params):
        """
        Default parameterized binary pooling unitary.
        """
        circ = QuantumCircuit(2, name="default_pool_block")
        circ.rz(-np.pi/2, 1)
        circ.cx(1, 0)
        circ.rz(params[0], 0)
        circ.ry(params[1], 1)
        circ.cx(0, 1)
        circ.ry(params[2], 1)
        return circ

    @classmethod
    def register_block(cls, name, block_function):
        """
        Register a custom convolution/pooling block.
        """
        if not callable(block_function):
            raise TypeError("Custom conv/pool block function must be callable.")
        cls._block_registry[name] = block_function

    @classmethod
    def get_block(cls, block_type):
        """
        Retrieve a convolution/pooling block function from the block registry.
        """
        if block_type not in cls._block_registry:
            raise ValueError(f"Unknown block_type '{block_type}'. "
                             f"Available blocks: {list(cls._block_registry.keys())}")
        return cls._block_registry[block_type]

    def add_convolution_layer(self, layer_name, block_type="default_conv", weight_sharing=False, qubits=None):
        """
        Adds a convolution layer using a specified convolution block type.
        """
        layer_params = []
        # Choose qubits involved in the layer
        if qubits is None:
            qubits = list(self.surviving_qubits)
        n = len(qubits)
        if n < 2:
            raise ValueError("Convolution layer needs at least 2 qubits.")
        # Pattern
        # TODO: generalize to different patterns
        pairs = [[qubits[i], qubits[i + 1]] for i in range(n - 1)]
        if n > 2:
            pairs.append([qubits[-1], qubits[0]])
        # Retrieve convolution block building function
        convolution_block_builder = self.get_block(block_type)
        # How many parameters (weights) does one block contain?
        test_params = [Parameter(f"test_{i}") for i in range(15)]
        test_circ = convolution_block_builder(test_params)
        weights_per_block = len(test_circ.parameters)
        # Build the layer
        if weight_sharing:
            theta = [Parameter(f"{layer_name}_theta_{i+1}") for i in range(weights_per_block)]
            layer_params += theta
            block = convolution_block_builder(theta)
            for a, b in pairs:
                self.circuit = self.circuit.compose(block, qubits=[a, b])
                self.circuit.barrier()
        else:
            j = 0
            for k, (a, b) in enumerate(pairs):
                theta = [Parameter(f"{layer_name}_theta_{i+j+1}") for i in range(weights_per_block)]
                layer_params += theta
                block = convolution_block_builder(theta)
                self.circuit = self.circuit.compose(block, qubits=[a, b])
                self.circuit.barrier()
                j += weights_per_block
        self.circuit.barrier()
        self.parameters += layer_params
        self.layer_history.append(f"{layer_name} ({block_type})")
        return layer_params

    def add_pooling_layer(self, layer_name, block_type="default_pool", weight_sharing=False, qubits=None):
        """
        Adds a pooling layer using a specified pooling block type.
        """
        layer_params = []
        # Choose qubits involved in the layer
        if qubits is None:
            qubits = list(self.surviving_qubits)
        n = len(qubits)
        # TODO: Decide how to handle odd qubit numbers
        # if n % 2 != 0:
            # raise ValueError(f"Pooling layer requires an even number of qubits; got {n}.")
        # Define (source, sink) pairs
        half = n // 2
        sources = qubits[:half]
        sinks = qubits[half:]
        pairs = list(zip(sources, sinks))
        # Retrieve pooling block builder
        pooling_block_builder = self.get_block(block_type)
        # How many parameters (weights) does one block contain?
        test_params = [Parameter(f"test_{i}") for i in range(15)]
        test_circ = pooling_block_builder(test_params)
        weights_per_block = len(test_circ.parameters)
        # Build the layer
        if weight_sharing:
            theta = [Parameter(f"{layer_name}_theta_{i+1}") for i in range(weights_per_block)]
            layer_params += theta
            block = pooling_block_builder(theta)
            for s, t in pairs:
                self.circuit = self.circuit.compose(block, qubits=[s, t])
                self.circuit.barrier()
        else:
            j = 0
            for k, (s, t) in enumerate(pairs):
                theta = [Parameter(f"{layer_name}_theta_{i+j+1}") for i in range(weights_per_block)]
                layer_params += theta
                block = pooling_block_builder(theta)
                self.circuit = self.circuit.compose(block, qubits=[s, t])
                self.circuit.barrier()
                j += weights_per_block
        self.circuit.barrier()
        self.parameters += layer_params
        self.layer_history.append(f"{layer_name} ({block_type})")
        # Update surviving qubits (sinks remain)
        self.surviving_qubits = list(sinks)
        return layer_params, list(sinks)


    # MEASUREMENT
    def add_measurement_layer(self, layer_name="measurement", qubits=None):
        """
        Add a measurement layer to the QCNN circuit.
        """
        # Default: measure all surviving qubits
        if qubits is None:
            qubits = list(self.surviving_qubits)
        n_meas = len(qubits)
        # Create a classical register for measurement outputs
        # creg = ClassicalRegister(n_meas, name=f"{layer_name}_creg")
        creg = ClassicalRegister(n_meas, name='c')
        self.circuit.add_register(creg)
        # Apply measurements
        for i, q in enumerate(qubits):
            self.circuit.measure(q, creg[i])
        self.layer_history.append(f"{layer_name}")
        self.measured_qubits = qubits
        self.measurement = True


    # QUICK BUILD
    def quick_build(self, data=None, encoding_method=None, n_layers=None, hadamard=True, conv_block="default_conv",
                    pool_block="default_pool", conv_weight_sharing=False, pool_weight_sharing=False, measure_last=True):
        """
        Build a default QCNN circuit in one call.
        """
        # Verify number of qubits is a power of two (TODO: generalize with adaptive pooling)
        if not (self.num_qubits and ((self.num_qubits & (self.num_qubits - 1)) == 0)):
            raise ValueError(f"At the moment, the quick_build() method requires number_of_qubits to be a power of 2 to allow default pooling.")
        # If needed, start with encoding layer (a data vector and an encoding method must be provided)
        self.reset()
        if data is not None and encoding_method is not None:
            self.encode(data, method=encoding_method)
        # If needed, add a Hadamard layer
        if hadamard:
            self.add_hadamard_layer("hadamard")
        # Automatically choose number of layers to end with one surviving qubit
        if n_layers is None:
            n_layers = int(np.log2(self.num_qubits))
        # Build convolution/pooling sequence
        for i in range(n_layers):
            self.add_convolution_layer(f"conv_{i + 1}", block_type=conv_block, weight_sharing=conv_weight_sharing)
            self.add_pooling_layer(f"pool_{i + 1}", block_type=pool_block, weight_sharing=pool_weight_sharing)
        # Measure last qubit
        if measure_last:
            self.add_measurement_layer("measurement", qubits=[self.surviving_qubits[-1]])
        return self


    # NOISE
    def configure_noise(self):
        # Use full noise model of chosen IBM device
        if self.noise_settings["from_backend"]:
            noise_model = NoiseModel.from_backend(self.noise_settings["backend"])
            return noise_model
        # Use customized noise model
        else:
            noise_model = NoiseModel()
            # Readout error
            if self.noise_settings["readout"]:
                # Require measurement layer
                if not self.measurement:
                    raise RuntimeError("Can't apply readout error because QCNN object is missing a measurement layer.")
                # Set readout probabilities
                p_0_given_1 = self.noise_settings["readout_probs"][0]
                p_1_given_0 = self.noise_settings["readout_probs"][1]
                readout_probs = [[1-p_1_given_0, p_1_given_0],
                                 [p_0_given_1, 1-p_0_given_1]]
                # Apply readout error to all measured qubits
                for q in self.measured_qubits:
                    noise_model.add_readout_error(ReadoutError(readout_probs), [q])
            # Depolarizing error
            if self.noise_settings["depolarizing"]:
                # 1-qubit depolarizing error
                p_1q = self.noise_settings["depolarizing_probs"]["1-q"]
                one_qubit_gates = self.noise_settings["depolarizing_gates"]["1-q"]
                if p_1q > 0 and one_qubit_gates:
                    one_qubit_error = depolarizing_error(p_1q, 1)
                    for gate_name in one_qubit_gates:
                        noise_model.add_all_qubit_quantum_error(one_qubit_error, [gate_name])
                # 2-qubit depolarizing error
                p_2q = self.noise_settings["depolarizing_probs"]["2-q"]
                two_qubit_gates = self.noise_settings["depolarizing_gates"]["2-q"]
                if p_2q > 0 and two_qubit_gates:
                    two_qubit_error = depolarizing_error(p_2q, 2)
                    for gate_name in two_qubit_gates:
                        noise_model.add_all_qubit_quantum_error(two_qubit_error, [gate_name])
            # Thermal relaxation error
            if self.noise_settings["thermal_relaxation"]:
                t1 = self.noise_settings["thermal_relaxation_t1"]
                t2 = self.noise_settings["thermal_relaxation_t2"]
                # 1-qubit thermal relaxation error
                time_1q = self.noise_settings["thermal_relaxation_gate_exec_times"]["1-q"]
                one_qubit_gates = self.noise_settings["thermal_relaxation_gates"]["1-q"]
                if one_qubit_gates:
                    one_qubit_error = thermal_relaxation_error(t1, t2, time_1q)
                    for gate_name in one_qubit_gates:
                        noise_model.add_all_qubit_quantum_error(one_qubit_error, [gate_name])
                # 2-qubit thermal relaxation error
                time_2q = self.noise_settings["thermal_relaxation_gate_exec_times"]["2-q"]
                two_qubit_gates = self.noise_settings["thermal_relaxation_gates"]["2-q"]
                if two_qubit_gates:
                    single_qubit_channel = thermal_relaxation_error(t1, t2, time_2q)
                    # Build a 2-qubit error from two 1-qubit thermal channels
                    two_qubit_error = single_qubit_channel.expand(single_qubit_channel)
                    for gate_name in two_qubit_gates:
                        noise_model.add_all_qubit_quantum_error(two_qubit_error, [gate_name])
            return noise_model


    # CIRCUIT EXECUTION
    def execute(self, parameters=None, shots=1024):
        """
        Execute the QCNN circuit with assigned parameters.
        """
        # Require measurement layer before execution
        if not self.measurement:
            raise RuntimeError("QCNN object is missing a measurement layer.")
        # Check parameters
        if self.parameters:
            if parameters is None:
                raise ValueError("Parameter values must be provided for a parameterized circuit.")
            if len(parameters) != len(self.parameters):
                raise ValueError(f"Expected {len(self.parameters)} parameters, but got {len(parameters)}.")
        # Normalize device string
        local_device = self.aer_simulator_settings["device"].strip().upper()
        if local_device not in {"CPU", "GPU"}:
            raise ValueError("Device must be either 'cpu' or 'gpu'.")
        # Build backend kwargs dictionary to pass to AerSimulator
        backend_kwargs = {"device": local_device, "method": self.aer_simulator_settings["method"]}
        # Set noise model (if desired)
        if self.noise:
            noise_model = self.configure_noise()
            backend_kwargs["noise_model"] = noise_model
        # Create backend
        try:
            backend = AerSimulator(**backend_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Can't initialize AerSimulator with device={local_device}: {e!r}"
            )
        # Optional GPU availability check
        if local_device == "GPU":
            try:
                devs = backend.available_devices()
                if not any(d.lower() == "gpu" for d in devs):
                    raise RuntimeError(f"AerSimulator not showing GPUs. Available_devices={devs}")
            except Exception:
                pass
        # Set simulation seed
        backend.set_options(seed_simulator=42)
        # Use transpiled circuit only in noisy mode
        circuit_to_run = self.circuit
        if self.noise:
            pass_manager = generate_preset_pass_manager(
                optimization_level=3,
                backend=backend,
                seed_transpiler=42
            )
            self.transpiled_circuit = pass_manager.run(self.circuit)
            print(f"\nTranspiled ops:")
            print(self.transpiled_circuit.count_ops())
            circuit_to_run = self.transpiled_circuit
        # Sample and get counts
        sampler = AerSampler.from_backend(backend)
        pub = (circuit_to_run, parameters, shots)
        job = sampler.run([pub])
        result = job.result()
        data = result[0].data
        if hasattr(data, "c"):
            counts = data.c.get_counts()
        else:
            counts = data.meas.get_counts()
        return job, counts








