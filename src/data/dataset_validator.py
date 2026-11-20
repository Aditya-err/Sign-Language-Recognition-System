"""
Dataset Integrity Validator for Indian Sign Language (ISL) Recognition.
=======================================================================

Author: Senior AI/ML Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.data.dataset_validator

Technical Overview for Interviews:
----------------------------------
In machine learning engineering, garbage data guarantees a garbage model.
Before training, datasets must undergo strict automated validation to catch:
  - Corrupted coordinate values (NaN, Inf).
  - Empty or collapsed hands (all-zeros).
  - Missing participant/provenance metadata.
  - Dimensionality mismatches.
  - Duplicate or repetitive samples.

This module inspects all JSON sample files under `data/raw/` without mutating
or deleting any files, producing a comprehensive human-readable report.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_FEATURE_DIM = 126
REQUIRED_METADATA_KEYS = [
    "version",
    "sample_id",
    "label",
    "participant_id",
    "session_id",
    "timestamp",
    "hand_config",
    "hands_count",
    "raw_landmarks",
    "normalized_features",
]


class DatasetValidator:
    """
    Audits raw ISL landmark samples for completeness, formatting, and mathematical integrity.
    """

    def __init__(self, data_dir: str = "data/raw") -> None:
        p = Path(data_dir)
        self.data_dir = p if p.is_absolute() else (PROJECT_ROOT / data_dir)

    def validate_single_record(self, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Validates an individual sample JSON record.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_error_reasons)
        """
        errors = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return False, [f"JSON parse failure: {str(e)}"]

        # 1. Check Metadata Completeness
        for key in REQUIRED_METADATA_KEYS:
            if key not in data:
                errors.append(f"Missing required key: '{key}'")

        if errors:
            return False, errors

        # 2. Check Label and Participant presence
        if not data.get("label"):
            errors.append("Empty or missing 'label'")
        if not data.get("participant_id"):
            errors.append("Empty or missing 'participant_id'")

        # 3. Check Normalized Feature Dimensionality & Finite Values
        features = data.get("normalized_features", [])
        if not isinstance(features, list) or len(features) != EXPECTED_FEATURE_DIM:
            errors.append(f"Invalid normalized_features length: expected {EXPECTED_FEATURE_DIM}, got {len(features)}")
        else:
            feat_arr = np.array(features, dtype=np.float32)
            if not np.all(np.isfinite(feat_arr)):
                errors.append("Normalized features contain NaN or Inf values")
            if np.all(feat_arr == 0.0):
                errors.append("Empty sample: normalized features are all zeros (no hand present)")

        # 4. Check Raw Landmarks Structure
        raw = data.get("raw_landmarks", {})
        left = raw.get("left_hand")
        right = raw.get("right_hand")

        if left is not None:
            left_arr = np.array(left)
            if left_arr.shape != (21, 3):
                errors.append(f"Invalid left_hand landmark shape: {left_arr.shape}, expected (21, 3)")
            elif not np.all(np.isfinite(left_arr)):
                errors.append("Left hand contains NaN or Inf coordinates")

        if right is not None:
            right_arr = np.array(right)
            if right_arr.shape != (21, 3):
                errors.append(f"Invalid right_hand landmark shape: {right_arr.shape}, expected (21, 3)")
            elif not np.all(np.isfinite(right_arr)):
                errors.append("Right hand contains NaN or Inf coordinates")

        # 5. Hand Config Consistency
        config = data.get("hand_config")
        if config == "two_hand" and (left is None or right is None):
            errors.append("Inconsistent hand_config: marked 'two_hand' but one hand landmark array is missing")
        elif config == "single_left" and left is None:
            errors.append("Inconsistent hand_config: marked 'single_left' but left_hand landmarks are missing")
        elif config == "single_right" and right is None:
            errors.append("Inconsistent hand_config: marked 'single_right' but right_hand landmarks are missing")

        return (len(errors) == 0), errors

    def validate_dataset(self) -> Dict[str, Any]:
        """
        Scans all sample records in the data directory and produces an audit report.
        """
        report: Dict[str, Any] = {
            "total_files_scanned": 0,
            "valid_samples": 0,
            "corrupted_samples": 0,
            "class_distribution": {},
            "participant_distribution": {},
            "hand_config_distribution": {},
            "issues": [],
        }

        if not self.data_dir.exists():
            report["issues"].append(f"Target data directory does not exist: {self.data_dir}")
            return report

        json_files = list(self.data_dir.rglob("*.json"))
        report["total_files_scanned"] = len(json_files)

        seen_feature_hashes = set()
        duplicate_count = 0

        for file_path in json_files:
            is_valid, errors = self.validate_single_record(file_path)

            if is_valid:
                report["valid_samples"] += 1
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    label = data["label"]
                    pid = data["participant_id"]
                    config = data["hand_config"]

                    report["class_distribution"][label] = report["class_distribution"].get(label, 0) + 1
                    report["participant_distribution"][pid] = report["participant_distribution"].get(pid, 0) + 1
                    report["hand_config_distribution"][config] = report["hand_config_distribution"].get(config, 0) + 1

                    # Duplicate vector check
                    feat_tuple = tuple(round(v, 4) for v in data["normalized_features"])
                    if feat_tuple in seen_feature_hashes:
                        duplicate_count += 1
                    else:
                        seen_feature_hashes.add(feat_tuple)

                except Exception:
                    pass
            else:
                report["corrupted_samples"] += 1
                try:
                    rel_path = str(file_path.relative_to(PROJECT_ROOT))
                except (ValueError, Exception):
                    rel_path = str(file_path)

                report["issues"].append({
                    "file": rel_path,
                    "errors": errors,
                })

        report["duplicate_samples_detected"] = duplicate_count
        return report

    def print_summary_report(self) -> None:
        """Formats and prints the validation summary."""
        print("\n" + "=" * 65)
        print("          ISL V2 DATASET INTEGRITY VALIDATION REPORT")
        print("=" * 65)

        results = self.validate_dataset()
        print(f"Directory Audited: {self.data_dir}")
        print(f"Total Files Scanned: {results['total_files_scanned']}")
        print(f"Valid Samples:       {results['valid_samples']}")
        print(f"Corrupted Samples:   {results['corrupted_samples']}")
        print(f"Duplicate Vectors:   {results.get('duplicate_samples_detected', 0)}")
        print("-" * 65)

        if results["class_distribution"]:
            print("Class Distribution:")
            for cls, count in sorted(results["class_distribution"].items()):
                print(f"  Class '{cls}': {count} samples")
            print("-" * 65)

        if results["participant_distribution"]:
            print("Participant Breakdown:")
            for pid, count in sorted(results["participant_distribution"].items()):
                print(f"  Participant '{pid}': {count} samples")
            print("-" * 65)

        if results["hand_config_distribution"]:
            print("Hand Configurations:")
            for cfg, count in sorted(results["hand_config_distribution"].items()):
                print(f"  {cfg}: {count} samples")
            print("-" * 65)

        if results["issues"]:
            print(f"ISSUES DETECTED ({len(results['issues'])} files):")
            for item in results["issues"][:10]:  # Print first 10
                print(f"  [X] {item['file']}: {', '.join(item['errors'])}")
            if len(results["issues"]) > 10:
                print(f"  ... and {len(results['issues']) - 10} more.")
        else:
            print("[SUCCESS] All scanned records meet 100% data integrity standards!")

        print("=" * 65 + "\n")


if __name__ == "__main__":
    validator = DatasetValidator()
    validator.print_summary_report()
