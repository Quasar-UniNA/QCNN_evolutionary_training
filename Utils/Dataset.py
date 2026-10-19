import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import matplotlib.pyplot as plt
import random


class Dataset:
    def __init__(self, name="my_dataset"):
        """
        Generic image dataset handler for ML pipelines.
        """
        self.name = name
        self.X = None
        self.y = None
        self.image_size = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # LOADING
    def load_from_keras(self, name, local_backup_path, verbose=True):
        """
        Load built-in Keras datasets.
        """
        datasets = {
            "mnist": mnist,
            "fashion_mnist": fashion_mnist,
            "cifar10": cifar10,
            "cifar100": cifar100
        }
        if name not in datasets:
            raise ValueError(f"Dataset {name} not supported.")
        try:
            # Normal download from Keras
            (x_train, y_train), (x_test, y_test) = datasets[name].load_data()
        except Exception as e:
            # Access local backup of required dataset
            print(f"Could not download '{name}' from Keras: {e}")
            print(f"Looking for local backup...")
            local_backup_path = Path(local_backup_path).expanduser().resolve()
            if local_backup_path.is_dir():
                local_file = local_backup_path/f"{name}.npz"
            else:
                local_file = local_backup_path
            if not local_file.exists():
                raise FileNotFoundError(f"Local backup file not found at:\n  {local_file}")
            data = np.load(local_file)
            x_train, y_train = data["x_train"], data["y_train"]
            x_test, y_test = data["x_test"], data["y_test"]
        # Override default split by merging train/test partitions (future split must be done separately)
        self.X = np.concatenate([x_train, x_test], axis=0)
        self.y = np.concatenate([y_train, y_test], axis=0).flatten()
        # Ensure channel dimension
        # if self.X.ndim == 3:
        # self.X = np.expand_dims(self.X, -1)
        self.image_size = self.X.shape[1:]
        # Control prints
        if verbose:
            print(f">>> Dataset '{name}' has been loaded successfully")
            print(f"   - Number of instances: {len(self.X)}")
            print(f"   - Number of classes: {len(np.unique(self.y))}")
            print(f"   - Image size: {self.image_size}")
        return self.X, self.y

    def load_from_directory(self, dataset_path):
        """
        Load dataset from a local directory.
        """
        # TODO: Implement this method
        pass

    # FILTERING
    def filter_classes(self, desired_classes, verbose = True):
        """
        Keep only samples whose label is in desired_classes.
        Raises an error if any desired label is not present in the dataset.
        """
        if self.X is None or self.y is None:
            raise ValueError("No dataset loaded yet. Please load data before filtering.")
        actual_classes = np.unique(self.y)
        missing = [cls for cls in desired_classes if cls not in actual_classes]
        if missing:
            raise ValueError(f"The following desired classes are not present in the dataset: {missing}")
        mask = np.isin(self.y, desired_classes)
        self.X = self.X[mask]
        self.y = self.y[mask]
        # Control print
        if verbose:
            selected_classes = np.unique(self.y)
            print(f">>> Dataset filtered successfully")
            print(f"   - Selected classes: {selected_classes}")
            print(f"   - Number of instances after filtering: {len(self.X)}")

    # RESIZING
    def resize_images(self, target_size=(64, 64), verbose=True):
        """Resize images using PIL. Optionally plot comparison for a random element of each class."""
        if self.X is None:
            raise ValueError("No dataset loaded yet. Please load data before resizing.")
        resized = []
        for img in self.X:
            if img.shape[0:2] != target_size:
                img_resized = Image.fromarray(img.squeeze())
                img_resized = img_resized.resize(target_size)
                img_resized = np.array(img_resized)
                resized.append(img_resized)
            else:
                resized.append(img)
        self.X = np.array(resized)
        self.image_size = self.X.shape[1:]
        # Control prints
        if verbose:
            print(f">>> Images resized successfully")
            print(f"   - New image size: {self.image_size}")

    # FRACTIONING
    def fraction(self, fraction=0.1, random_seed=42, verbose=True):
        """
        Keep only a fraction of the dataset while preserving class balance.

        Args:
            fraction (float): Fraction of the dataset to keep (0 < fraction <= 1).
            random_state (int): Seed for reproducible random sampling.
        """
        if self.X is None or self.y is None:
            raise ValueError("No dataset loaded yet. Please load data before fractioning.")
        if fraction >= 1.0:
            # Keep all samples
            return
        # Use stratified sampling to preserve class balance
        _, self.X, _, self.y = train_test_split(
            self.X,
            self.y,
            test_size=fraction,
            stratify=self.y,
            random_state=random_seed
        )
        # Control prints
        if verbose:
            print(f">>> Dataset reduced successfully")
            print(f"   - Fraction kept: {fraction}")
            print(f"   - New number of instances: {len(self.X)}")

    # TRAIN/TEST SPLITTING
    def split(self, test_size=0.2, random_seed=42, stratify=True, verbose=True):
        if self.X is None or self.y is None:
            raise ValueError("No dataset loaded yet. Please load data before splitting.")
        stratify_labels = self.y if stratify else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify_labels
        )
        # Control prints
        if verbose:
            print(f">>> Dataset split successfully")
            print(f"   - Training size: {1 - test_size}")
            print(f"   - Training instances: {len(self.X_train)}")
            print(f"   - Test size: {test_size}")
            print(f"   - Test instances: {len(self.X_test)}")

    # SCALING
    def scale(self, scaler_type="minmax", verbose=True):
        """Scale images to [0,1] (minmax) or standardize (z-score)."""
        if self.X_train is None or self.X_test is None:
            raise ValueError("Train/test splits do not exist. Call split() first.")
        if self.image_size is None:
            raise ValueError("Image size unknown. Make sure data has been loaded and (optionally) resized correctly.")
        # Flatten images
        n_train = len(self.X_train)
        n_test = len(self.X_test)
        X_train_flat = self.X_train.reshape(n_train, -1).astype("float32")
        X_test_flat = self.X_test.reshape(n_test, -1).astype("float32")
        # Choose scaler
        if scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError("Unsupported scaler. Use 'minmax' or 'standard'.")
        # Fit on train, transform train and test
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)
        # Reshape back to original image dimensions
        self.X_train = X_train_scaled.reshape((n_train, *self.image_size))
        self.X_test = X_test_scaled.reshape((n_test, *self.image_size))
        # Control prints
        if verbose:
            print(f">>> Scaling completed using '{scaler_type}' scaler")
            print(f"   - Training data fitted and transformed")
            print(f"   - Test data scaled using the same transformation")

    # CONVERSIONS
    # def to_dataframe(self):
    # TODO: Implement this method

    # def from_dataframe(self, df):
    # TODO: Implement this method

