"""
MLP Model Training Pipeline.
============================

Author: Senior AI/ML Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.models.train

Technical Overview for Interviews:
----------------------------------
This script implements a reproducible, metric-driven training workflow:
  1. Deterministic Seeding: Seeds `random`, `numpy`, and `tensorflow` for identical runs.
  2. Training Regularization: Early stopping (patience=12) prevents overfitting to training data.
     Learning rate decay (`ReduceLROnPlateau`) dynamically halves the learning rate when validation
     loss plateaus, enabling smooth convergence into local minima.
  3. Artifact Preservation:
     - `best_model.keras`: Checkpointed at the peak validation accuracy epoch.
     - `training_history.json`: Serialized loss, accuracy, and learning rate curves.
     - `reports/figures/training_history.png`: Visual loss and accuracy diagnostics.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Adjust sys.path
sys_path_root = Path(__file__).resolve().parents[2]
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from src.data.augmentation import augment_training_set
from src.models.mlp import build_mlp_classifier, get_model_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def set_deterministic_seed(seed: int = 42) -> None:
    """Configures deterministic seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info("Deterministic random seed set to %d", seed)


def plot_training_history(
    history_dict: Dict,
    output_path: Path,
) -> None:
    """Plots dual Loss and Accuracy training curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history_dict["loss"], label="Training Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history_dict["val_loss"], label="Validation Loss", color="#d62728", linewidth=2, linestyle="--")
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Accuracy Curve
    ax2.plot(epochs, history_dict["accuracy"], label="Training Accuracy", color="#1f77b4", linewidth=2)
    ax2.plot(epochs, history_dict["val_accuracy"], label="Validation Accuracy", color="#2ca02c", linewidth=2, linestyle="--")
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved training curves to %s", output_path)


def train_pipeline(
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    augment: bool = False,
    augmentation_factor: int = 1,
    seed: int = 42,
) -> Tuple[tf.keras.Model, Dict]:
    """
    Executes full model training pipeline and saves all evaluation artifacts.
    """
    set_deterministic_seed(seed)

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-split data
    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    y_val = np.load(processed_dir / "y_val.npy")

    num_classes = len(np.unique(np.concatenate([y_train, y_val])))
    input_dim = X_train.shape[1]

    logger.info("Loaded datasets: X_train=%s, y_train=%s, X_val=%s, y_val=%s",
                X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    logger.info("Input dimension: %d, Classes count: %d", input_dim, num_classes)

    # Optional training data augmentation
    if augment and augmentation_factor > 0:
        logger.info("Applying coordinate-space augmentation (factor=%d)...", augmentation_factor)
        X_train, y_train = augment_training_set(
            X_train, y_train, augmentation_factor=augmentation_factor, random_seed=seed
        )
        logger.info("Augmented training shape: X_train=%s, y_train=%s", X_train.shape, y_train.shape)

    # Construct model
    model = build_mlp_classifier(
        input_dim=input_dim,
        num_classes=num_classes,
        dropout_rate=0.30,
        learning_rate=learning_rate,
        l2_reg=1e-4,
    )

    logger.info("Model Summary:\n%s", get_model_summary(model))

    # Configure Callbacks
    best_model_path = models_dir / "best_model.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
            mode="min",
        ),
    ]

    start_time = time.time()
    logger.info("Starting training for up to %d epochs (batch_size=%d)...", epochs, batch_size)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed_time = time.time() - start_time
    logger.info("Training finished in %.2f seconds", elapsed_time)

    # Save training history JSON
    history_dict = {
        k: [float(v) for v in vals] for k, vals in history.history.items()
    }
    history_dict["total_epochs_trained"] = len(history.history["loss"])
    history_dict["training_duration_seconds"] = round(elapsed_time, 2)
    history_dict["final_train_loss"] = float(history.history["loss"][-1])
    history_dict["final_train_acc"] = float(history.history["accuracy"][-1])
    history_dict["final_val_loss"] = float(history.history["val_loss"][-1])
    history_dict["final_val_acc"] = float(history.history["val_accuracy"][-1])
    history_dict["best_val_acc"] = float(max(history.history["val_accuracy"]))

    history_path = reports_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    logger.info("Saved training history to %s", history_path)

    # Plot curves
    curve_path = figures_dir / "training_history.png"
    plot_training_history(history_dict, curve_path)

    return model, history_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP Classifier for Indian Sign Language.")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Model checkpoints directory")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Reports and figures directory")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--augment", action="store_true", help="Enable coordinate-space training augmentation")
    parser.add_argument("--aug_factor", type=int, default=1, help="Augmentation multiplier (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    proj_root = Path(__file__).resolve().parents[2]

    train_pipeline(
        processed_dir=proj_root / args.processed_dir,
        models_dir=proj_root / args.models_dir,
        reports_dir=proj_root / args.reports_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        augment=args.augment,
        augmentation_factor=args.aug_factor,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
