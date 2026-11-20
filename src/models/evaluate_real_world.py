"""
Independent Real-World Dataset Evaluation Module.
================================================

Author: Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.models.evaluate_real_world

Technical Overview for Interviews:
----------------------------------
A hallmark of rigorous machine learning engineering is maintaining strict separation
between different evaluation paradigms:
  1. Benchmark Test Split (Random Split on Archive): 99.61% accuracy.
     Demonstrates the model mastered the archive manifold, but carries consecutive-frame autocorrelation.
  2. Independent Real-World Recordings:
     Evaluates generalization to genuine live webcam frames captured under diverse environmental conditions.
  3. Unseen-Participant Split:
     Measures inter-subject anatomical variance (different hand sizes, skin tones, finger flexibility).
  4. Live Webcam Testing (FSM & HUD):
     Validates temporal hold stability, flicker reduction, and single-emission logic.

CRITICAL INTEGRITY RULE:
  These metrics must NEVER be averaged or combined into a single fabricated number.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys_path_root = Path(__file__).resolve().parents[2]
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from src.core.normalization import DUAL_HAND_FEATURE_DIM
from src.inference.predictor import SignPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_real_world")


def load_real_world_samples(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Scans data/real_world/ for recorded JSON sample files.
    """
    features: List[np.ndarray] = []
    labels: List[str] = []
    metadata_list: List[Dict] = []

    if not data_dir.is_dir():
        return np.empty((0, DUAL_HAND_FEATURE_DIM)), np.empty((0,), dtype=str), []

    json_files = list(data_dir.rglob("sample_*.json"))
    logger.info("Found %d real-world sample files in %s", len(json_files), data_dir)

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            feat = np.array(data["feature_vector_126"], dtype=np.float32)
            lbl = str(data["label"]).strip().upper()
            features.append(feat)
            labels.append(lbl)
            metadata_list.append({
                "participant_id": data.get("participant_id", "UNKNOWN"),
                "session_id": data.get("session_id", "UNKNOWN"),
                "condition": data.get("condition", "UNKNOWN"),
                "file_path": str(fpath),
            })
        except Exception as e:
            logger.warning("Error loading sample file %s: %s", fpath, e)

    if len(features) == 0:
        return np.empty((0, DUAL_HAND_FEATURE_DIM)), np.empty((0,), dtype=str), []

    return np.array(features, dtype=np.float32), np.array(labels, dtype=str), metadata_list


def evaluate_real_world_dataset(
    real_world_dir: Path,
    output_report_path: Path,
) -> Dict:
    """
    Evaluates the trained model on independent real-world recorded samples.
    """
    predictor = SignPredictor()
    all_classes = predictor.classes

    X_real, y_real, metadata = load_real_world_samples(real_world_dir)
    total_samples = len(X_real)

    if total_samples == 0:
        logger.info("No real-world samples found in %s. Generating status report.", real_world_dir)
        report_content = [
            "# Independent Real-World Dataset Evaluation Report",
            "",
            "**Status:** PENDING REAL-WORLD RECORDINGS  ",
            "**Target Directory:** `data/real_world/`  ",
            "",
            "No independent webcam session samples have been collected yet.",
            "To record genuine independent samples, run:",
            "```bash",
            "python src/data/live_recorder.py --sign A --samples 15 --participant P01",
            "```",
            "",
            "## Baseline Test Benchmark Comparison (For Reference)",
            "- **Original Archive Test Set (Held-Out Split):** 99.61% Top-1 Accuracy (762/765 correct)",
            "- **Independent Real-World Recordings:** Pending collection",
        ]
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))
        return {"total_samples": 0, "status": "PENDING"}

    # Run predictions
    y_preds: List[str] = []
    confs: List[float] = []
    low_conf_count = 0

    for i in range(total_samples):
        res = predictor.predict(X_real[i])
        pred_label = res.class_label if res.class_label else "UNKNOWN"
        y_preds.append(pred_label)
        confs.append(res.confidence)
        if res.confidence < 0.70:
            low_conf_count += 1

    y_preds = np.array(y_preds)
    correct_mask = (y_preds == y_real)
    acc = float(np.mean(correct_mask))

    unique_real_classes = sorted(list(set(y_real)))
    clf_rep = classification_report(
        y_real,
        y_preds,
        labels=unique_real_classes,
        output_dict=True,
        zero_division=0,
    )

    macro_prec = clf_rep["macro avg"]["precision"]
    macro_rec = clf_rep["macro avg"]["recall"]
    macro_f1 = clf_rep["macro avg"]["f1-score"]

    logger.info("Real-World Evaluation Complete: %d samples, Accuracy: %.2f%%, Macro F1: %.4f",
                total_samples, acc * 100, macro_f1)

    # Generate Markdown Report
    lines = [
        "# Independent Real-World Dataset Evaluation Report",
        "",
        f"**Date:** 2026-11-20  ",
        f"**Total Real-World Samples:** {total_samples}  ",
        f"**Unique Tested Classes:** {len(unique_real_classes)} of {len(all_classes)}  ",
        f"**Low-Confidence Predictions (< 0.70):** {low_conf_count} ({low_conf_count/total_samples*100:.1f}%)  ",
        "",
        "## 1. Metric Comparison Across Evaluation Paradigms",
        "",
        "| Paradigm | Accuracy | Macro F1 | Context / Explanation |",
        "| :--- | :---: | :---: | :--- |",
        "| **Original Random-Split Test Set** | **99.61%** | **0.9960** | Evaluated on held-out frames of the original archive dataset. Subject to consecutive-frame autocorrelation. |",
        f"| **Independent Real-World Recordings** | **{acc * 100:.2f}%** | **{macro_f1:.4f}** | Evaluated on independent live webcam recordings under genuine room conditions. |",
        "",
        "## 2. Per-Class Real-World Performance",
        "",
        "| Class | Tested Samples | Correct | Accuracy | Precision | Recall | F1-Score |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for c in unique_real_classes:
        stats = clf_rep[c]
        c_tot = int(stats["support"])
        c_acc = float(np.mean(y_preds[y_real == c] == c)) if c_tot > 0 else 0.0
        lines.append(
            f"| `{c}` | {c_tot} | {int(round(c_acc * c_tot))} | {c_acc*100:.1f}% | "
            f"{stats['precision']:.3f} | {stats['recall']:.3f} | {stats['f1-score']:.3f} |"
        )

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved evaluation report to %s", output_report_path)

    return {
        "total_samples": total_samples,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "low_conf_count": low_conf_count,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    real_world_dir = repo_root / "data" / "real_world"
    report_p = repo_root / "reports" / "real_world_evaluation.md"

    evaluate_real_world_dataset(
        real_world_dir=real_world_dir,
        output_report_path=report_p,
    )


if __name__ == "__main__":
    main()
