"""
MLP Model Architecture for Sign Language Landmark Classification.
==================================================================

Author: Senior AI/ML Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.models.mlp

Technical Overview for Interviews:
----------------------------------
Why a Multi-Layer Perceptron (MLP) instead of CNN or LSTM for static signs?
  1. Input Nature: Our feature representation is an explicit 126-dimensional geometric
     coordinate vector (not a 2D spatial pixel grid or a temporal video sequence).
     - CNNs exploit spatial translation equivariance across 2D grids (pixels). Applying
       convolutions across a 1D landmark coordinate array imposes arbitrary local filters
       between unrelated joints (e.g., convolving thumb tip with pinky wrist).
     - LSTMs model recurrent temporal state transitions across time frames. Static alphabet
       signs (e.g. 'A', 'B', '5') have zero temporal dynamics; using an LSTM on static frames
       (as legacy V1 did) introduces unnecessary sequence complexity and recurrent overhead.
  2. Computational Efficiency:
     - Parameter Count: ~28,000 parameters (compared to millions in ResNet or MobileNet).
     - Inference Latency: < 1.0 millisecond on standard CPU without GPU acceleration.
     - Memory Footprint: ~150 KB serialized weights.
  3. Regularization Strategy:
     - Batch Normalization stabilizes internal covariate shifts between dense layers.
     - Dropout (0.30 and 0.20) prevents joint co-adaptation.
     - L2 Weight Decay (1e-4) penalizes extreme weights, preventing over-reliance on any single joint.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers


def build_mlp_classifier(
    input_dim: int = 126,
    num_classes: int = 35,
    dropout_rate: float = 0.30,
    learning_rate: float = 0.001,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    """
    Constructs and compiles the Keras Sequential MLP model for ISL classification.

    Architecture:
      Input(126)
        -> Dense(128, ReLU, L2)
        -> BatchNormalization()
        -> Dropout(dropout_rate)
        -> Dense(64, ReLU, L2)
        -> BatchNormalization()
        -> Dropout(dropout_rate * 0.67)
        -> Dense(32, ReLU)
        -> Dense(num_classes, Softmax)

    Args:
        input_dim: Input feature dimensionality (126 for dual-hand normalized landmarks).
        num_classes: Number of target sign classes (35 for ISL 1-9 and A-Z).
        dropout_rate: Primary dropout probability (default: 0.30).
        learning_rate: Initial Adam learning rate (default: 0.001).
        l2_reg: L2 weight penalty coefficient (default: 1e-4).

    Returns:
        Compiled tf.keras.Model ready for training or inference.
    """
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,), name="landmark_features_input"),
            # Layer 1: Broad geometric feature combination
            layers.Dense(128, activation="relu", kernel_regularizer=reg, name="dense_1"),
            layers.BatchNormalization(name="batch_norm_1"),
            layers.Dropout(dropout_rate, name="dropout_1"),
            # Layer 2: Finger configuration abstraction
            layers.Dense(64, activation="relu", kernel_regularizer=reg, name="dense_2"),
            layers.BatchNormalization(name="batch_norm_2"),
            layers.Dropout(dropout_rate * 0.67, name="dropout_2"),
            # Layer 3: Class manifold compression
            layers.Dense(32, activation="relu", name="dense_3"),
            # Output Layer: Softmax probability distribution over all sign classes
            layers.Dense(num_classes, activation="softmax", name="sign_class_probabilities"),
        ],
        name="ISL_Landmark_MLP_Classifier",
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_summary(model: tf.keras.Model) -> str:
    """Returns the string representation of model.summary() for logging and reports."""
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)
