# Training strategies for Quantum Convolutional Neural Networks 

A Python implementation of Quantum Convolutional Neural Networks (QCNNs) for image classification, with configurable circuit architectures, optimization algorithms, and quantum noise simulation.

The repository provides utilities for dataset preprocessing, QCNN construction, model training, evaluation, and experiment reporting. The included example trains an 8-qubit QCNN on a binary Fashion-MNIST classification task using a Genetic Algorithm (GA) in Qiskit's ideal simulation setting.

## Repository Structure

```
.
├── Example_experiment.py   # Example training and evaluation workflow
└── Utils/
    ├── Dataset.py         # Dataset loading and preprocessing
    ├── Model.py           # Classification, optimization, and reporting
    └── QCNN.py            # Circuit construction and simulation
```

## Features

- Amplitude encoding of image data.
- Configurable quantum convolution and pooling layers.
- Custom two-qubit blocks and optional parameter sharing within each layer.
- Binary classification (one-vs-all classification implementation in progress).
- Supported optimization algorithms from SciPy, noisyopt, and EVOVAQ
- CPU or GPU simulation through Qiskit Aer, depending on the installed backend and available hardware.
- Optional readout, depolarizing, thermal relaxation, and backend-derived noise models.
- Export of experiment settings, training records, evaluation metrics, and compressed archives.

## Example Experiment

`Example_experiment.py` implements the following configuration:

| Setting | Value |
| --- | --- |
| Dataset | Fashion-MNIST |
| Selected classes | 0 and 1 |
| Image size | 16 × 16 pixels |
| Dataset fraction | 1% of the selected classes |
| Train/test split | 70% / 30%, stratified |
| Preprocessing | Min-max scaling fitted on the training set |
| Encoding | Amplitude encoding |
| Number of qubits | 8 |
| Architecture | Three convolution–pooling stages |
| Trainable parameters | 12 |
| Weight sharing | Enabled within each layer |
| Output | Measurement of the final surviving qubit |
| Loss | Binary cross-entropy |
| Optimizer | Genetic Algorithm (EVOVAQ) |
| Population size | 20 |
| Fitness evaluation budget | 1,000 |
| Shots per circuit | 1,000 |
| Simulation method | Statevector |
| Quantum noise | Disabled |

## Dependencies

The source code imports the following packages:

- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- Pillow
- Keras and a compatible backend
- Qiskit
- Qiskit Aer
- Qiskit IBM Runtime
- noisyopt
- evovaq

Circuit plotting may also require `pylatexenc`.

Install these dependencies in a dedicated Python environment. The repository does not currently include a pinned dependency file; compatible package versions must therefore be selected for the target environment.

The example uses local simulation and does not require access to an IBM Quantum account.

## Running the Example

Run the script from the repository root after completing the setup below.

### 1. Set the dataset backup path

The current `Dataset.load_from_keras()` signature requires a `local_backup_path` argument. In `Example_experiment.py`, replace:

```python
dataset.load_from_keras("fashion_mnist")
```

with:

```python
dataset.load_from_keras(
    "fashion_mnist",
    local_backup_path="data/fashion_mnist.npz",
)
```

The loader first attempts to load the dataset through Keras. If that attempt fails, it reads the supplied backup file, which must contain the arrays `x_train`, `y_train`, `x_test`, and `y_test`.

The loader merges the original training and test partitions before applying the repository's filtering, subsampling, and train/test splitting procedure.

### 2. Set the environment variables

The example expects:

- `SLURM_ARRAY_TASK_ID`: an integer used as the run identifier and as the seed for NumPy and Python's random generator.
- `DEVICE`: the Aer simulation device, either `cpu` or `gpu`.

For a local CPU run on Linux or macOS:

```bash
export SLURM_ARRAY_TASK_ID=1
export DEVICE=cpu
python Example_experiment.py
```

For a local CPU run in Windows PowerShell:

```powershell
$env:SLURM_ARRAY_TASK_ID = "1"
$env:DEVICE = "cpu"
python Example_experiment.py
```

GPU execution requires a compatible Qiskit Aer installation and supported hardware.

### 3. Run repeated experiments with Slurm

The script can also be launched through a Slurm job array. For example:

```bash
#!/bin/bash
#SBATCH --job-name=qcnn_ga
#SBATCH --array=1-10%2
#SBATCH --output=qcnn_%A_%a.out

export DEVICE=cpu
python Example_experiment.py
```

Submit this script from the repository root with the required Python environment activated.

Each array task uses its own run identifier and output directory. Dataset subsampling and splitting use a fixed seed of `42`, while the optimizer's NumPy and Python random seeds follow the run identifier.

## Outputs

Results are saved relative to the location of `Example_experiment.py`:

```text
Output/
└── Exp_12/
    └── fashion_16x16_bin01_12p_ga_maxnfev1000/
        └── run_001/
```

The workflow exports:

- A QCNN circuit diagram.
- The optimization configuration.
- Training reports and plots.
- Training-set and test-set evaluation reports.
- A complete experiment report in `full_experiment_report.txt`.
- A compressed experiment archive in `archive_run_001.pkl.gz`.

Evaluation includes binary cross-entropy, accuracy, a confusion matrix, and a classification report.

Run directories are determined by the run identifier. Use distinct identifiers when results from separate executions need to be retained.

## Customization

Modify `Example_experiment.py` to explore different experimental configurations:

- **Data:** selected classes, image resolution, dataset fraction, split ratio, and scaling.
- **Architecture:** number of qubits, convolution and pooling blocks, layer count, and weight sharing.
- **Optimization:** optimization framework, method, population settings, parameter bounds, and evaluation budget.
- **Simulation:** device, simulation method, shots, and noise settings.

Custom convolution and pooling blocks can be registered through:

```python
qcnn.register_block("custom_block", block_function)
```

For amplitude encoding, a flattened sample must fit within the state space of the circuit:

```text
number of features ≤ 2 ** number_of_qubits
```

For example, a 16 × 16 grayscale image contains 256 features and can be encoded using 8 qubits. Input vectors must have a nonzero norm.

## Implementation Notes

- The included example performs binary classification.
- Amplitude encoding is implemented; angle encoding is currently a placeholder.
- Loading datasets from a local image directory is currently a placeholder.
- Noise settings take effect only when `qcnn.noise` is enabled.

