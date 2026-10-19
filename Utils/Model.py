import time
import copy
import matplotlib
import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any, List
from functools import partial
from scipy.optimize import minimize
from noisyopt import minimizeSPSA
# from qiskit_algorithms.optimizers import ADAM
# from Utils.Mods import ADAM
from evovaq.problem import Problem
from evovaq.GeneticAlgorithm import GA
from evovaq.DifferentialEvolution import DE
import evovaq.tools.operators as op
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, classification_report


@dataclass(slots=True, kw_only=True)
class TrainingRecordBase:
    tag: str
    framework: str
    train_time: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@dataclass(slots=True)
class ScipyTrainingRecord(TrainingRecordBase):
    """
    Stores results from a SciPy optimization run.
    """
    success: bool
    message: str
    final_loss: float
    obj_evals: int
    n_iter: Optional[int] = None
    best_params: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass(slots=True)
class QiskitTrainingRecord(TrainingRecordBase):
    """
    Stores results from a Qiskit optimization run.
    """
    final_loss: float
    obj_evals: int
    n_iter: int
    best_params: np.ndarray


@dataclass(slots=True)
class GeneticTrainingRecord(TrainingRecordBase):
    """
    Stores results from a GA-based Evovaq optimization run.
    """
    final_gen: int
    fitness_evals: int
    best_fitness: float
    best_params: np.ndarray
    log: Optional[Any] = None


@dataclass(slots=True)
class EvaluationRecord:
    """
    Stores model evaluation results.
    """
    tag: str
    loss: Optional[float]
    accuracy: float
    confusion_matrix: List[List[int]]
    classification_report: str
    parameters_used: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class BinaryClassifier:
    """
    Train and evaluation for QCNN-based classifiers.
    """
    def __init__(self, qcnn, dataset):
        self.qcnn = qcnn
        self.dataset = dataset
        self.supported_frameworks = ['scipy', 'evovaq', 'qiskit', 'noisyopt']
        self.optimization_framework = None
        self.optimization_settings = None
        self.best_params = None
        self.archive = {'data': {},
                        'config': {},
                        'training': [],
                        'evaluations': [],
                        'evaluations_raw': {},
                        'loss_history': [],
                        'loss_history_per_iter': [],
                        'params_history': [],
                        'best_loss_history': [],
                        }
        self.compiled = False
        self.trained = False


    # CONFIGURATION
    def compile(self, opt_framework="scipy", loss="binary_crossentropy", opt_settings=None, report=False, output_path=Path("")):
        """
        Configure optimization.
        """
        # Set optimization framework
        if opt_framework in self.supported_frameworks:
            self.optimization_framework = opt_framework
            self.archive['config']['framework'] = opt_framework
        else:
            raise ValueError(f"Unsupported framework: {opt_framework}. "
                             f"Supported frameworks are: {self.supported_frameworks}")
        # Set loss function
        # TODO: handle other loss functions
        if loss == "binary_crossentropy":
            self.archive['config']['loss_function'] = loss
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
        # Set optimization configuration
        if opt_settings is None:
            raise ValueError(
                f"An 'opt_settings' dictionary must be provided when using framework '{self.optimization_framework}'. "
                "No default optimizer settings are defined."
            )
        if not isinstance(opt_settings, dict):
            raise TypeError(f"'opt_settings' must be a dictionary, got {type(opt_settings).__name__}")
        self.optimization_settings = copy.deepcopy(opt_settings)
        for key in self.optimization_settings:
            self.archive['config'][key] = self.optimization_settings[key]
        # Flag the model object as compiled
        self.compiled = True
        # Save configuration report
        if report:
            self.export_archive(filepath=output_path/"configuration.txt", section='config')


    # LOSS
    def loss_evaluation(self, y_true, y_pred, probabilities):
        """
        Generic loss evaluation wrapping different existing loss functions.
        """
        loss = None
        # Check compiling
        if not self.compiled:
            raise RuntimeError("Model must be compiled before evaluation (call .compile()).")
        # --- Cross entropy loss
        if self.archive['config']['loss_function'] == "binary_crossentropy":
            # Convert list of dicts to a 2D array of probabilities (with consistent class ordering!)
            class_labels = sorted(probabilities[0].keys())
            probs = np.array([[p[c] for c in class_labels] for p in probabilities])
            # Normalize y_true
            y_true = np.array(y_true, dtype=int)
            # Calculate sklearn log_loss
            loss = log_loss(y_true, probs)
        # --- TODO: other loss function options
        return loss


    # PREDICTION/EVALUATION
    def counts_to_class(self, counts, method='binary'):
        """
        Extract class prediction from measurement counts dictionary.
        """
        probabilities = None
        predicted_class = None
        if method == 'binary':
            total = 0
            counts_per_class = {'0': 0, '1': 0}
            probabilities = {'0': 0, '1': 0}
            for bitstring, count in counts.items():
                if bitstring == '0':
                    counts_per_class['0'] += count
                    total += count
                elif bitstring == '1':
                    counts_per_class['1'] += count
                    total += count
            predicted_class = max(counts_per_class, key=counts_per_class.get)
            for key in probabilities.keys():
                probabilities[key] += counts_per_class[key] / total
        # TODO: elif method == 'onehot' etc... MUST RETURN CONSISTENT OUTPUTS!
        return probabilities, int(predicted_class)


    def predict(self, x, params, shots):
        """
        Use the QCNN to predict labels for a single sample or a batch.
        """
        probs = []
        pred_labels = []
        # Turn x into an iterable if it's a single sample
        is_batch = isinstance(x, (list, np.ndarray)) and np.ndim(x) > 1
        x_iter = x if is_batch else [x]
        # Compute predictions
        for sample in x_iter:
            qcnn_copy = copy.deepcopy(self.qcnn)
            qcnn_copy.encode(sample)
            _, counts = qcnn_copy.execute(parameters=params, shots=shots)
            probabilities, predicted_label = self.counts_to_class(counts)
            probs.append(probabilities)
            pred_labels.append(predicted_label)
        return np.array(probs), np.array(pred_labels)


    def evaluate(self, x_data, y_true, params=None, shots=1024, verbose=True, tag='evaluation', report=False, output_path=Path("")):
        """
        Evaluate the QCNN binary classifier on a dataset.
        """
        # Check compiling
        if not self.compiled:
            raise RuntimeError("Model must be compiled before evaluation (call .compile()).")
        # Define model parameters to use
        if params is None and self.best_params is None:
            raise ValueError("No parameters available for evaluation.")
        param_vector = params if params is not None else self.best_params
        # Compute probabilities and class predictions for x_data
        probs, predicted_labels = self.predict(x_data, param_vector, shots)
        # Compute sklearn metrics
        loss = self.loss_evaluation(y_true, predicted_labels, probs)
        accuracy = float(accuracy_score(y_true, predicted_labels))
        cm = confusion_matrix(y_true, predicted_labels)
        metrics_report = classification_report(y_true, predicted_labels, digits=4)
        # Printout
        if verbose:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluation — {tag}")
            print(f"Loss: {loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nConfusion Matrix:\n", cm)
            print("\nClassification Report:\n", metrics_report)
        # Store evaluation record in the archive
        record = EvaluationRecord(
            tag=tag,
            loss=loss,
            accuracy=accuracy,
            confusion_matrix=cm.tolist(),
            classification_report=metrics_report,
            parameters_used="best_params" if params is None else "custom"
        )
        self.archive["evaluations"].append(record)
        self.archive["evaluations_raw"][tag] = {
            "y_true": np.array(y_true, dtype=int).copy(),
            "y_pred": np.array(predicted_labels, dtype=int).copy(),
            "probs": copy.deepcopy(probs),
        }
        if report:
            self.export_archive(filepath=output_path/f"{tag}_report.txt", section='evaluations')
        return loss, accuracy, cm, metrics_report


    # TRAINING
    @staticmethod
    def training_time(start, stop):
        training_duration = stop - start
        minutes = int(training_duration // 60)
        seconds = int(training_duration % 60)
        training_duration_str = f"{minutes} min {seconds} sec"
        return minutes, seconds, training_duration_str


    def fit(self, x_train, y_train, shots, tag='training', report=False, plot=False, output_path=Path("")):
        """
        Train the classifier with custom optimization algorithms.
        """
        # TODO: x_train and y_train can be default to the ones contained in self.dataset after splitting
        # Check compiling
        if not self.compiled:
            raise RuntimeError("Model must be compiled before training (call .compile()).")
        # Check trainable parameters
        n_params = len(self.qcnn.parameters)
        if n_params == 0:
            raise RuntimeError("QCNN circuit has no trainable parameters.")
        # Clean archive sections (really necessary?)
        self.archive['loss_history'] = []
        self.archive['loss_history_per_iter'] = []
        self.archive['params_history'] = []
        self.archive['best_loss_history'] = []
        # Define loss function wrapper
        def objective(params, train_data, train_labels, shots):
            probs, y_pred = self.predict(train_data, params, shots)
            loss = self.loss_evaluation(train_labels, y_pred, probs)
            self.archive['loss_history'].append(float(loss))
            self.archive['params_history'].append(np.array(params, dtype=float).copy())
            if not self.archive['best_loss_history']:
                self.archive['best_loss_history'].append(float(loss))
            else:
                self.archive['best_loss_history'].append(min(self.archive['best_loss_history'][-1], float(loss)))
            return loss

        # --- Optimize with Scipy -----------------------------------------
        if self.optimization_framework == 'scipy':
            # Track loss evolution (per iteration!) with a callback function.
            # Warning: callback function calls do not necessarily match objective calls (nfev)!!!
            def callback_iter(params):
                last_loss_value = self.archive['loss_history'][-1]
                self.archive['loss_history_per_iter'].append(last_loss_value)
            # Optimization sub-routine
            start = time.time()
            result = minimize(
                fun=objective,
                x0=self.optimization_settings['initial_params'],
                args=(x_train, y_train, shots),
                method=self.optimization_settings['method'],
                bounds=self.optimization_settings['bounds'],
                callback=callback_iter,
                # options={'maxiter': self.optimization_settings['maxiter']}
                options = self.optimization_settings['options'],
            )
            end = time.time()
            # Compute training duration
            training_duration = (self.training_time(start, end))[2]
            # Store optimized parameters
            self.best_params = result.x
            # Store training record in the archive
            record = ScipyTrainingRecord(
                tag=tag,
                framework=self.optimization_framework,
                train_time=training_duration,
                success=result.success,
                message=result.message,
                final_loss=result.fun,
                obj_evals=result.nfev,
                # n_iter=result.nit,
                best_params=result.x,
            )
            self.archive["training"].append(record)

        # --- Optimize with Noisyopt ---------------------------------------
        if self.optimization_framework == 'noisyopt':
            # Check all necessary keys are provided in opt_settings
            required_keys = ['initial_params', 'bounds', 'maxiter', 'a', 'c', 'alpha', 'gamma', 'paired']
            missing_keys = [key for key in required_keys if key not in self.optimization_settings]
            if missing_keys:
                raise KeyError(
                    f"These parameters are missing from noisyopt opt_settings: {missing_keys}. "
                    f"Required keys are: {required_keys}"
                )
            # Check method
            if self.optimization_settings.get('method', 'SPSA') != 'SPSA':
                raise ValueError(
                    f"Unsupported noisyopt method: {self.optimization_settings.get('method')}. "
                    "Only 'SPSA' is currently supported."
                )

            x0 = np.array(self.optimization_settings['initial_params'], dtype=float)
            bounds = self.optimization_settings['bounds']

            def callback_iter(params):
                if self.archive['loss_history']:
                    last_loss_value = self.archive['loss_history'][-1]
                    self.archive['loss_history_per_iter'].append(last_loss_value)
            # Optimization sub-routine
            start = time.time()
            result = minimizeSPSA(
                func=objective,
                x0=x0,
                args=(x_train, y_train, shots),
                niter=self.optimization_settings['maxiter'],
                a=self.optimization_settings['a'],
                c=self.optimization_settings['c'],
                alpha=self.optimization_settings['alpha'],
                gamma=self.optimization_settings['gamma'],
                paired=self.optimization_settings['paired'],
                bounds=bounds,
                callback=callback_iter,
            )
            end = time.time()
            # Compute training duration
            training_duration = (self.training_time(start, end))[2]
            # Store optimized parameters
            self.best_params = np.array(result.x, dtype=float).copy()
            # Store training record in the archive
            record = ScipyTrainingRecord(
                tag=tag,
                framework=self.optimization_framework,
                train_time=training_duration,
                success=bool(getattr(result, 'success', True)),
                message=str(getattr(result, 'message', '')),
                final_loss=float(result.fun),
                obj_evals=int(getattr(result, 'nfev', len(self.archive['loss_history']))),
                n_iter=int(getattr(result, 'nit', self.optimization_settings['maxiter'])),
                best_params=np.array(result.x, dtype=float).copy(),
            )
            self.archive["training"].append(record)


        # --- Optimize with Qiskit -----------------------------------------
        if self.optimization_framework == 'qiskit':
            if self.optimization_settings['method'] == 'ADAM':
                partial_obj = partial(objective,
                                      train_data=x_train,
                                      train_labels=y_train,
                                      shots=shots
                                      )
                optimizer = ADAM(maxiter=self.optimization_settings['maxiter'],
                                 tol=self.optimization_settings['tolerance'],
                                 lr=self.optimization_settings['learning_rate'])
                # Optimization sub-routine
                start = time.time()
                result = optimizer.minimize(fun=partial_obj,
                                            x0=self.optimization_settings['initial_params'],
                                            jac=None,
                                            bounds=self.optimization_settings['bounds'])
                end = time.time()
                # Compute training duration
                training_duration = (self.training_time(start, end))[2]
                # Store optimized parameters
                self.best_params = result.x
                # Store training record in the archive
                record = QiskitTrainingRecord(
                    tag=tag,
                    framework=self.optimization_framework,
                    train_time=training_duration,
                    final_loss=result.fun,
                    obj_evals=result.nfev,
                    n_iter=result.nit,
                    best_params=result.x,
                )
                self.archive["training"].append(record)
                self.archive['loss_history'] = optimizer._loss_history


        # --- Optimize with Evovaq ----------------------------------------
        elif self.optimization_framework == 'evovaq':
            partial_obj = partial(objective,
                                  train_data=x_train,
                                  train_labels=y_train,
                                  shots=shots
                                  )
            if 'bounds' in self.optimization_settings:
                bounds = self.optimization_settings['bounds']
            else:
                raise RuntimeError("The chosen optimization framework requires bounds."
                                   "No 'bounds' were assigned in optimization settings.")
            problem = Problem(n_params, bounds, partial_obj)
            # Optimize via Genetic Algorithm
            if self.optimization_settings['method'] == 'GA':
                optimizer = GA(selection=self.optimization_settings['selection_op'],
                               crossover=self.optimization_settings['crossover_op'],
                               mutation=self.optimization_settings['mutation_op'],
                               sigma=self.optimization_settings['mut_sigma'],
                               mut_indpb=self.optimization_settings['mut_indpb'],
                               cxpb=self.optimization_settings['cx_pb'],
                               tournsize=self.optimization_settings['tourn_size'],
                               elitism=True)
            # Optimize via Differential Evolution
            elif self.optimization_settings['method'] == 'DE':
                optimizer = DE(variant=self.optimization_settings['variant'],
                               differential_weight=self.optimization_settings['differential_weight'],
                               CR=self.optimization_settings['CR'])
            # initial_population = np.array([np.copy(self.optimization_settings['initial_params'])
            #                               for _ in range(self.optimization_settings['pop_size'])], dtype=object)
            init_pop_list = [np.random.uniform(0, 2 * np.pi, len(self.qcnn.parameters))
                                           for _ in range(self.optimization_settings['pop_size']-1)]
            init_pop_list.append(self.optimization_settings['initial_params'])
            initial_population = np.array(init_pop_list, dtype=object)

            # Optimization sub-routine
            start = time.time()
            result = optimizer.optimize(problem=problem,
                                        pop_size=self.optimization_settings['pop_size'],
                                        initial_pop=initial_population,
                                        max_nfev=self.optimization_settings['max_nfev']
                                        )
            end = time.time()
            training_duration = (self.training_time(start, end))[2]
            # Save optimized parameters
            self.best_params = result.x
            # Store training record in the archive
            record = GeneticTrainingRecord(
                tag=tag,
                framework=self.optimization_framework,
                final_gen=result.gen,
                fitness_evals=result.nfev,
                best_fitness=result.fun,
                best_params=np.array(result.x, dtype=float).copy(),
                train_time=training_duration,
                log=result.log
            )
            self.archive["training"].append(record)
        self.trained = True
        # Make reports and plots
        if report:
            self.export_archive(filepath=output_path/f"{tag}_report.txt", section='training')
        if plot:
            if 'loss_history' in self.archive and self.archive['loss_history']:
                self.export_archive(filepath=output_path/f"{tag}_loss_curve.png", section='loss_history')
        # TODO: choose a better way to check we have fitness history data
        if self.optimization_framework == 'evovaq':
            self.export_archive(filepath=output_path/f"{tag}_fitness_curve.png", section='fitness_history')


    # REPORTS
    def save_archive_pickle(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Convert dataclass objects to dict to be future-proof (optional but recommended)
        archive_copy = copy.deepcopy(self.archive)
        archive_copy["training"] = [asdict(r) for r in archive_copy.get("training", [])]
        archive_copy["evaluations"] = [asdict(r) for r in archive_copy.get("evaluations", [])]
        with gzip.open(filepath, "wb") as f:
            pickle.dump(archive_copy, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f">>> Archive saved to {filepath}")


    # Necessary inside this class?
    @staticmethod
    def load_archive_pickle(filepath: Path):
        with gzip.open(filepath, "rb") as f:
            return pickle.load(f)


    def export_archive(self, filepath="qcnn_archive.txt", section="all"):
        """
        Save the model archive (or a selected section) to a text file.
        """
        # Check archive has the requested section (key)
        # existing_sections = {"all", *self.archive.keys()}
        # if section not in existing_sections:
            # raise ValueError(f"Invalid archive section: '{section}'. Choose from: {existing_sections}.")
        # Write text reports
        if filepath.suffix == '.txt':
            with open(filepath, "w") as f:
                f.write(f"=== EXPERIMENT REPORT ===\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"QCNN: {getattr(self.qcnn, 'name', 'No name available')}\n")
                f.write(f"Dataset: {getattr(self.dataset, 'name', 'No name available')}\n\n")
                # --- Configuration -----------------------------------------------------
                if section in {"all", "config"} and self.archive.get("config"):
                    f.write(">>> CONFIGURATION\n\n")
                    for key, value in self.archive["config"].items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                # --- Training ----------------------------------------------------------
                if section in {"all", "training"} and self.archive.get("training"):
                    f.write(">>> TRAINING\n")
                    for i, record in enumerate(self.archive["training"], start=1):
                        f.write(f"\n--- Training #{i}: {record.tag} ---\n")
                        f.write(f"Timestamp: {record.timestamp}\n")
                        f.write(f"Framework: {record.framework}\n")
                        f.write(f"Training time: {record.train_time}\n")
                        # SciPy training record
                        if isinstance(record, ScipyTrainingRecord):
                            f.write(f"Success: {record.success}\n")
                            f.write(f"Message: {record.message}\n")
                            f.write(f"Final loss: {record.final_loss:.6f}\n")
                            f.write(f"Objective evaluations: {record.obj_evals}\n")
                            f.write(f"Iterations: {record.n_iter}\n")
                            f.write(f"Best parameters:\n{np.array2string(record.best_params, precision=4)}\n")
                            if "loss_history" in self.archive and self.archive["loss_history"]:
                                f.write("\nLoss history (per objective evaluation / nfev):\n")
                                for j, loss_value in enumerate(self.archive["loss_history"], start=1):
                                    f.write(f"  Eval {j}: {loss_value:.6f}\n")
                        # GA training records
                        elif isinstance(record, GeneticTrainingRecord):
                            f.write(f"Final generation: {record.final_gen}\n")
                            f.write(f"Fitness evaluations: {record.fitness_evals}\n")
                            f.write(f"Best fitness: {record.best_fitness:.6f}\n")
                            f.write(f"Best parameters:\n{np.array2string(record.best_params, precision=4)}\n")
                            f.write(f"Log stats:\n{pd.DataFrame(record.log).to_string(index=False)}\n")
                        # Qiskit training records
                        elif isinstance(record, QiskitTrainingRecord):
                            f.write(f"Final loss: {record.final_loss:.6f}\n" if record.final_loss is not None else "Final loss: None\n")
                            f.write(f"Objective evaluations: {record.obj_evals}\n")
                            f.write(f"Iterations: {record.n_iter}\n")
                            f.write("Best parameters:\n")
                            if record.best_params is not None:
                                f.write(f"{np.array2string(record.best_params, precision=4)}\n")
                            else:
                                f.write("None\n")
                            if "loss_history" in self.archive and self.archive["loss_history"]:
                                f.write("\nLoss history (per iteration):\n")
                                for j, loss_value in enumerate(self.archive["loss_history"], start=1):
                                    f.write(f"  Iter {j}: {loss_value:.6f}\n")
                        f.write("\n")
                # --- Evaluation --------------------------------------------------------
                if section in {"all", "evaluations"} and self.archive.get("evaluations"):
                    f.write(">>> EVALUATION\n")
                    for i, record in enumerate(self.archive["evaluations"], start=1):
                        f.write(f"\n--- Evaluation #{i}: {record.tag} ---\n")
                        f.write(f"Timestamp: {record.timestamp}\n")
                        f.write(f"Parameters: {record.parameters_used}\n")
                        f.write(f"Loss: {record.loss:.4f}\n")
                        f.write(f"Accuracy: {record.accuracy:.4f}\n")
                        f.write("Confusion Matrix:\n")
                        for row in record.confusion_matrix:
                            f.write(f"  {row}\n")
                        f.write("\nClassification Report:\n")
                        f.write(f"{record.classification_report}\n")
                        f.write("\n")
            print(f">>> Archive section '{section}' saved to {filepath}")
        # Plots
        else:
            # --- Scipy loss history plot ------------------------------------------------
            if section in {"loss_history"} and self.archive.get("loss_history"):
                plt.figure(figsize=(8, 5))
                loss_values = self.archive["loss_history"]
                plt.plot(range(1, len(loss_values) + 1), loss_values, label="Loss", marker='o')
                plt.title(f"Loss curve")
                plt.xlabel("Evaluations")
                plt.ylabel("Loss function")
                plt.legend()
                plt.grid(True)
                plt.savefig(filepath, dpi=300)
                plt.close()
                print("\n>>> Loss history plot saved")
            # --- Evovaq GA fitness plot ------------------------------------------------
            if section in {"fitness_history"}:
                for i, record in enumerate(self.archive["training"], start=1):
                    log = record.log
                    gens = log["gen"]
                    plt.figure(figsize=(8, 5))
                    plt.plot(gens, log["min"], label="Min Loss", marker='o')
                    plt.plot(gens, log["mean"], label="Mean Loss", linestyle='--')
                    plt.plot(gens, log["max"], label="Max Loss", linestyle=':')
                    plt.title(f"Fitness per generation")
                    plt.xlabel("Generation")
                    plt.ylabel("Fitness function")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(filepath, dpi=300)
                    plt.close()
                    print("\n>>> Fitness history plot saved")



class OVAClassifier:
    def __init__(self, qcnn, dataset):
        self.qcnn = qcnn
        self.dataset = dataset
        self.n_classes = len(np.unique(dataset.y))
        self.binary_classifiers = {}
        self.compiled = False
        self.trained = False


    def compile(self, opt_framework='scipy', loss='binary_crossentropy', opt_settings=None, report=False, output_path=Path("")):
        """
        Instantiate a binary classifier for each one-vs-all combination and compile it.
        """
        for i in range(self.n_classes):
            self.binary_classifiers[f'{i}-vs-all'] = BinaryClassifier(copy.deepcopy(self.qcnn), self.dataset)
            self.binary_classifiers[f'{i}-vs-all'].compile(opt_framework=opt_framework,
                                                           loss=loss,
                                                           opt_settings=opt_settings)
            if report:
                self.binary_classifiers[f'{i}-vs-all'].export_archive(filepath=output_path/f"{i}-vs-all_configuration.txt",
                                                                      section='config')
        self.compiled = True


    def fit(self, shots, report=False, plot=False, output_path=Path("")):
        """
        Train each binary classifier.
        """
        # Check compiling
        if not self.compiled:
            raise RuntimeError("Model must be compiled before training (call .compile()).")
        # Train each binary classifier with a modified binary label vector
        onevsall_start = time.time()
        for i in range(self.n_classes):
            x_train = self.dataset.X_train
            y_train = np.where(self.dataset.y_train == i, 1, 0)
            self.binary_classifiers[f'{i}-vs-all'].fit(x_train=x_train,
                                                       y_train=y_train,
                                                       shots=shots,
                                                       tag=f'{i}-vs-all_training',
                                                       report=report,
                                                       plot=plot,
                                                       output_path=output_path)
        onevsall_end = time.time()
        onevsall_training_time = BinaryClassifier.training_time(onevsall_start, onevsall_end)[2]
        self.trained = True
        print(f">>> OVA training completed ({self.n_classes} classifiers) in {onevsall_training_time}")


    def predict(self, x_data, shots):
        """
        Predict multiclass labels using all trained binary classifiers.
        """
        if not self.trained:
            raise RuntimeError("OVAClassifier must be trained before prediction.")
        if any(clf.best_params is None for clf in self.binary_classifiers.values()):
            raise RuntimeError("Some binary classifiers have not been trained.")
        n_samples = len(x_data)
        scores = np.zeros((n_samples, self.n_classes))
        for i in range(self.n_classes):
            binary_classifier = self.binary_classifiers[f'{i}-vs-all']
            probs, _ = binary_classifier.predict(x_data, binary_classifier.best_params, shots)
            # Use the probability of class '1' (positive) as the "score"
            scores[:, i] = [p['1'] for p in probs]
        predicted_labels = np.argmax(scores, axis=1)
        return predicted_labels, scores


    def evaluate(self, x_data, y_true, shots, verbose=True, tag='OVA_evaluation', report=False, output_path=Path("")):
        """
        Evaluate OVA classifier on a dataset.
        """
        y_pred, scores = self.predict(x_data, shots)
        accuracy = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred)
        metrics_report = classification_report(y_true, y_pred, digits=4)
        # Printout
        if verbose:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluation — {tag}")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nConfusion Matrix:\n", cm)
            print("\nClassification Report:\n", metrics_report)
        # Evaluation report
        record = EvaluationRecord(
            tag=tag,
            loss=None,
            accuracy=accuracy,
            confusion_matrix=cm.tolist(),
            classification_report=metrics_report,
            parameters_used='best_params'
        )
        if report:
            output_file = output_path / f"{tag}_report.txt"
            with open(output_file, "w") as f:
                f.write(f"=== EXPERIMENT REPORT ===\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"QCNN: {getattr(self.qcnn, 'name', 'No name available')}\n")
                f.write(f"Dataset: {getattr(self.dataset, 'name', 'No name available')}\n\n")
                f.write(f">>> EVALUATION --- {tag}\n\n")
                f.write(f"Accuracy: {record.accuracy:.4f}\n")
                f.write("Confusion Matrix:\n")
                for row in record.confusion_matrix:
                    f.write(f"  {row}\n")
                f.write("\nClassification Report:\n")
                f.write(record.classification_report)
            print(f">>> OVA evaluation report saved to {output_file}")
        return accuracy, cm, metrics_report


    def evaluate_bin_classifiers(self, x_data, y_true, shots, verbose=False, tag='evaluation', report=False, output_path=Path("")):
        """
        Evaluate each binary classifier on a dataset.
        """
        for i in range(self.n_classes):
            y_true_binary = np.where(y_true == i, 1, 0)
            self.binary_classifiers[f"{i}-vs-all"].evaluate(x_data=x_data,
                                                             y_true=y_true_binary,
                                                             shots=shots,
                                                             verbose=verbose,
                                                             tag=f"{i}-vs-all_{tag}",
                                                             report=report,
                                                             output_path=output_path
                                                            )













