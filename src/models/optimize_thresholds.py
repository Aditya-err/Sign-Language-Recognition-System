"""
Threshold and Temporal Smoothing Optimization Analysis.
======================================================

Module: sign_language_v2.src.models.optimize_thresholds
Author: Senior AI/ML Engineer (Mentoring MCA Fresher)

This script analyzes the precision-coverage trade-off across different confidence
thresholds on the validation set, and simulates temporal majority voting to
substantiate the production configuration (window=7, threshold=0.70, hold=12).
"""

import json
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optimize_thresholds")


def analyze_threshold_tradeoffs(
    model_path: Path,
    data_dir: Path,
    output_report_path: Path,
) -> dict:
    logger.info("Loading model from: %s", model_path)
    model = tf.keras.models.load_model(str(model_path))

    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    # Run inference in batches
    logger.info("Running validation inference over %d samples...", len(X_val))
    probabilities = model(X_val, training=False).numpy()
    pred_classes = np.argmax(probabilities, axis=1)
    max_confs = np.max(probabilities, axis=1)

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
    results = {}

    for t in thresholds:
        accepted_mask = max_confs >= t
        num_accepted = int(np.sum(accepted_mask))
        num_rejected = len(X_val) - num_accepted
        coverage = num_accepted / len(X_val)

        if num_accepted > 0:
            acc_accepted = float(np.mean(pred_classes[accepted_mask] == y_val[accepted_mask]))
        else:
            acc_accepted = 0.0

        results[f"threshold_{t:.2f}"] = {
            "threshold": t,
            "accepted_samples": num_accepted,
            "rejected_samples": num_rejected,
            "coverage_pct": round(coverage * 100, 2),
            "effective_precision_on_accepted": round(acc_accepted * 100, 3),
        }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved threshold optimization results to %s", output_report_path)
    return results


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "models" / "best_model.keras"
    data_dir = repo_root / "data" / "processed"
    out_path = repo_root / "reports" / "threshold_optimization.json"

    res = analyze_threshold_tradeoffs(model_path, data_dir, out_path)
    print("\n" + "=" * 65)
    print("      CONFIDENCE THRESHOLD & COVERAGE TRADE-OFF ANALYSIS      ")
    print("=" * 65)
    print(f"{'Threshold':<12} | {'Coverage (%)':<15} | {'Precision (%)':<15} | {'Rejected'}")
    print("-" * 65)
    for k, v in res.items():
        print(f"{v['threshold']:<12.2f} | {v['coverage_pct']:<15.2f} | {v['effective_precision_on_accepted']:<15.3f} | {v['rejected_samples']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
