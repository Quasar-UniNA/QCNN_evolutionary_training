"""
Example code for reproducing the experiments on GA-based training for QCNNs. 
This experiment uses an 8-qubit QCNN with 12 trainable parameters and weight sharing to classify Fashion-MNIST classes 0 and 1, with images resized to 16×16 pixels. 
Training minimizes binary cross-entropy using a GA with a budget of 1000 fitness function evaluations, under noiseless simulation with 1000 shots per circuit. 
The script handles data preprocessing, model training, evaluation on the training and test sets, and result export, supporting repeated runs through Slurm job arrays.
"""

import os
import time
import datetime
import argparse
import random
import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.datasets import mnist, fashion_mnist
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino
from functools import partial
from evovaq.problem import Problem
from evovaq.GeneticAlgorithm import GA
import evovaq.tools.operators as op
from Utils.Dataset import Dataset
from Utils.QCNN import QCNN
from Utils.Model import BinaryClassifier


keras.utils.disable_interactive_logging()


def convolution_block_2p(params):
    qc = QuantumCircuit(2, name="2p_conv_block")
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    return qc


def pooling_block_2p(params):
    qc = QuantumCircuit(2, name="2p_pool_block")
    qc.crz(params[0], 0, 1)
    qc.x(0)
    qc.crx(params[1], 0, 1)
    return qc


def get_run_id() -> int:
    """
    Assumes the script is always launched via a Slurm job array like:
      #SBATCH --array=1-10%2
    Then SLURM_ARRAY_TASK_ID is 1..10 and we keep the same numbering.
    """
    return int(os.environ["SLURM_ARRAY_TASK_ID"])


def create_output_folder(base_path: Path, run_id: int) -> Path:
    """
    Deterministic per-run folder (safe for parallel array jobs and retries).
    Example: base_path/run_001, base_path/run_002, ...
    """
    run_folder = base_path / f"run_{run_id:03d}"
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder

print("\n--- QCNN BINARY CLASSIFIER (SLURM ARRAY) ---\n\n")

# Set seeds (numpy + python random)
RUN_ID = get_run_id()
seed = RUN_ID
np.random.seed(seed)
random.seed(seed)
print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] RUN_ID={RUN_ID}\n")

# Create main output folder
where_is_this_script = Path(__file__).resolve().parent
results_folder = where_is_this_script / "Output" / "Exp_12" / "fashion_16x16_bin01_12p_ga_maxnfev1000"

# Per-run output folder (deterministic)
output_folder = create_output_folder(results_folder, RUN_ID)

# Load and preprocess dataset
dataset = Dataset("fmnist_16x16")
dataset.load_from_keras("fashion_mnist")
dataset.filter_classes([0, 1])
dataset.resize_images((16, 16))
dataset.fraction(0.01, random_seed=42)
dataset.split(test_size=0.3, random_seed=42, stratify=True)
dataset.scale(scaler_type="minmax")

# Extract partitions
x_train = dataset.X_train
x_test = dataset.X_test
y_train = dataset.y_train
y_test = dataset.y_test

# Encode categorical labels as numerical labels
label_encoder = LabelEncoder()
y_train_numeric = label_encoder.fit_transform(y_train)
y_test_numeric = label_encoder.transform(y_test)

# Initialize QCNN
number_of_qubits = 8  # must be consistent with image size and encoding method
qcnn = QCNN(number_of_qubits=number_of_qubits, name=f"{number_of_qubits}qubit_qcnn")
# Build architecture (skip encoding momentarily)
# qcnn.add_hadamard_layer("hadamard")
# qcnn.add_convolution_layer("conv_1", block_type="default_conv", weight_sharing=False)
# _, survivors1 = qcnn.add_pooling_layer("pool_1", block_type="default_pool", weight_sharing=False)
# qcnn.add_measurement_layer(layer_name="output_measure", qubits=[survivors1[-1]])
qcnn.register_block("custom_conv", convolution_block_2p)
qcnn.register_block("custom_pool", pooling_block_2p)
conv1_params = qcnn.add_convolution_layer("conv_1", block_type="custom_conv", weight_sharing=True)
pool1_params, survivors1 = qcnn.add_pooling_layer("pool_1", block_type="custom_pool", weight_sharing=True)
conv2_params = qcnn.add_convolution_layer("conv_2", block_type="custom_conv", weight_sharing=True, qubits=survivors1)
pool2_params, survivors2 = qcnn.add_pooling_layer("pool_2", block_type="custom_pool", weight_sharing=True, qubits=survivors1)
conv3_params = qcnn.add_convolution_layer("conv_3", block_type="custom_conv", weight_sharing=True, qubits=survivors2)
pool3_params, survivors3 = qcnn.add_pooling_layer("pool_3", block_type="custom_pool", weight_sharing=True, qubits=survivors2)
qcnn.add_measurement_layer(layer_name="output_measure", qubits=[survivors3[-1]])

# Noise settings
qcnn.noise = False
qcnn.noise_settings = {"from_backend": False,
                       "backend": FakeTorino(),
                       "readout": False,
                       "readout_probs": (0.08, 0.08),
                       "depolarizing": True,
                       "depolarizing_probs": {"1-q": 0.02,
                                              "2-q": 0.10},
                       "depolarizing_gates": {"1-q": ["rz"],
                                              "2-q": ["cx"]},
                       "thermal_relaxation": False,
                       "thermal_relaxation_t1": 50e3,
                       "thermal_relaxation_t2": 70e3,
                       "thermal_relaxation_gates": {"1-q": ["h"],
                                                    "2-q": ["cx"]},
                       "thermal_relaxation_gate_exec_times": {"1-q": 50.0,
                                                              "2-q": 300.0},
                       }

# Aer simulator settings
# Import device from batch script (or set it manually in the dictionary by choosing between "cpu" and "gpu")
device = os.environ["DEVICE"].lower()
qcnn.aer_simulator_settings = {
    "device": device,
    "method": "statevector"
}
print("Aer simulator settings:", qcnn.aer_simulator_settings)
# Save circuit diagram
qcnn.plot_circuit_diagram(filename=output_folder / f"{qcnn.name}_plot.png", dpi=300)

# Instantiate model
model = BinaryClassifier(qcnn=qcnn, dataset=dataset)

# Evovaq optimization settings
framework = 'evovaq'
loss = 'binary_crossentropy'
opt_settings = {'method': 'GA',
                   'selection_op': op.sel_tournament,
                   'tourn_size': 5,
                   'mutation_op': op.mut_gaussian,
                   'mut_indpb': 0.15,
                   'mut_sigma': 0.2,
                   'crossover_op': op.cx_uniform,
                   'cx_pb': 0.9,
                   'pop_size': 20,
                   'max_gen': 10000,
                   'max_nfev': 1000,
                   'initial_params': np.random.uniform(0, 2 * np.pi, len(qcnn.parameters)),
                   'bounds': [(0, 2 * np.pi) for _ in range(len(qcnn.parameters))]
                   }

# Compile model
model.compile(
    opt_framework=framework,
    loss=loss,
    opt_settings=opt_settings,
    report=True,
    output_path=output_folder,
)


# Train model
model.fit(
    x_train=x_train,
    y_train=y_train_numeric,
    shots=1000,
    tag=f"training_run_{RUN_ID:03d}",
    report=True,
    plot=True,
    output_path=output_folder,
)

# Evaluate
model.evaluate(
    x_data=x_train,
    y_true=y_train_numeric,
    shots=1000,
    verbose=True,
    tag="train_set_evaluation",
    report=True,
    output_path=output_folder,
)

model.evaluate(
    x_data=x_test,
    y_true=y_test_numeric,
    shots=1000,
    verbose=True,
    tag="test_set_evaluation",
    report=True,
    output_path=output_folder,
)

# Save full experiment report
model.export_archive(filepath=output_folder / "full_experiment_report.txt", section="all")

# Save compressed archive (for offline ML analysis)
model.save_archive_pickle(output_folder / f"archive_run_{RUN_ID:03d}.pkl.gz")

print(f"\n--- RUN {RUN_ID} COMPLETED SUCCESSFULLY ---\n")
