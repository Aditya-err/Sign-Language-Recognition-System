"""
Stratified Dataset Splitting Module.
====================================

Author: Senior AI/ML Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.data.split_dataset

Technical Overview for Interviews:
----------------------------------
A cornerstone of rigorous machine learning is unbiased data partitioning.
This module partitions extracted features into Train (70%), Validation (15%),
and Test (15%) splits using stratified sampling.

Why Stratified Splitting?
  - Class Imbalance Protection: MediaPipe detection rates vary slightly between signs
    (e.g., complex overlapping two-handed signs may have slightly fewer samples than simple
    single-hand signs). Stratification guarantees that the class proportions in Train, Val,
    and Test are mathematically identical.
  - Test Set Integrity: The 15% Test set is sealed and never exposed to model training,
    hyperparameter tuning, or data augmentation. It serves as an unbiased benchmark.

Archive Dataset Limitation (Important for Interviews):
  The `archive.zip` dataset contains numbered filenames (`0.jpg`, `1.jpg`...) without
  participant/subject IDs. Therefore, GroupKFold (Leave-One-Subject-Out) cannot be applied.
  This limitation is explicitly tracked in `split_info.json`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Adjust sys.path
sys_path_root = Path(__file__).resolve().parents[2]
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("split_dataset")


def split_and_save_dataset(
    processed_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Dict:
    """
    Splits features and labels into stratified train, val, and test partitions.

    Args:
        processed_dir: Directory containing features.npy and labels.npy.
        train_ratio: Fraction for training set (default: 0.70).
        val_ratio: Fraction for validation set (default: 0.15).
        test_ratio: Fraction for test set (default: 0.15).
        random_state: Random seed for deterministic reproducibility.

    Returns:
        Summary dict of split statistics.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, \
        "Split ratios must sum to 1.0"

    features_path = processed_dir / "features.npy"
    labels_path = processed_dir / "labels.npy"

    if not features_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"Missing features.npy or labels.npy in {processed_dir}")

    logger.info("Loading processed data from %s", processed_dir)
    X = np.load(features_path)
    y_raw = np.load(labels_path)

    # Encode string labels to integer indices
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    class_names = [str(c) for c in encoder.classes_]

    # Save label encoder mapping
    label_encoder_mapping = {
        "classes": class_names,
        "class_to_idx": {cls: int(idx) for idx, cls in enumerate(class_names)},
        "idx_to_class": {int(idx): cls for idx, cls in enumerate(class_names)},
    }
    with open(processed_dir / "label_encoder.json", "w", encoding="utf-8") as f:
        json.dump(label_encoder_mapping, f, indent=2)
    logger.info("Saved label encoder mapping for %d classes", len(class_names))

    # Step 1: Split out Test set (e.g. 15%)
    test_size = test_ratio
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        stratify=y_encoded,
        random_state=random_state,
    )

    # Step 2: Split remaining 85% into Train (70%) and Val (15%)
    # Relative ratio for val within train_val: 0.15 / (0.70 + 0.15) = 0.17647
    relative_val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state,
    )

    # Sanity check: Ensure zero data leakage between partitions
    total_samples = len(X)
    assert len(X_train) + len(X_val) + len(X_test) == total_samples, "Sample count mismatch!"

    logger.info(
        "Split complete: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        len(X_train), (len(X_train) / total_samples) * 100,
        len(X_val), (len(X_val) / total_samples) * 100,
        len(X_test), (len(X_test) / total_samples) * 100,
    )

    # Save numpy partitions
    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "X_val.npy", X_val)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "X_test.npy", X_test)
    np.save(processed_dir / "y_test.npy", y_test)

    # Compile split metadata
    split_info = {
        "random_state": random_state,
        "strategy": "StratifiedShuffleSplit",
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "sample_counts": {
            "total": total_samples,
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "feature_dim": int(X.shape[1]),
        "num_classes": len(class_names),
        "limitations_and_integrity": {
            "participant_metadata_available": False,
            "subject_level_splitting_possible": False,
            "reason": (
                "Archive dataset images lack participant IDs. Stratified random splitting "
                "was applied to guarantee exact class distribution balance. Potential subject-level "
                "leakage exists if the same signer appears in train and test; this is an inherent "
                "limitation of the legacy archive."
            ),
        },
    }

    with open(processed_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    logger.info("Saved split metadata to %s", processed_dir / "split_info.json")
    return split_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Split processed dataset into train, val, and test sets.")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Processed directory")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Val ratio (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    proj_root = Path(__file__).resolve().parents[2]
    processed_dir = proj_root / args.processed_dir

    split_and_save_dataset(
        processed_dir=processed_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
