"""
Dataset Preparation Pipeline: Extract MediaPipe Hand Landmarks from Images.
===========================================================================

Author: Senior AI/ML Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.data.prepare_dataset

Technical Overview for Interviews:
----------------------------------
This script reads the raw Indian Sign Language image archive and transforms raw pixel
data (JPEG images) into normalized 126-dimensional geometric landmark feature vectors.

Why extract landmarks instead of training a CNN on raw pixels?
  1. Dimensionality Reduction: A 128x128x3 RGB image contains 49,152 values with immense
     background clutter, skin-tone variance, and lighting noise. MediaPipe extracts
     the underlying 21 hand joints (42 coordinates in 2D, 63 in 3D), eliminating 99.7%
     of irrelevant pixels while retaining 100% of the kinematic sign morphology.
  2. Compute Efficiency: An MLP trained on 126-D vectors trains in seconds on CPU,
     enabling fast iteration, hyperparameter tuning, and cross-validation without needing
     expensive GPU hardware.
  3. Explainability: Landmark coordinates correspond directly to biological joints
     (wrist, thumb CMC, index fingertip), making error analysis intuitive.

Pipeline Flow:
  archive.zip -> cv2.imdecode -> MediaPipe Hands (Static Mode) -> normalize_dual_hands -> .npy
"""

import argparse
import json
import logging
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Adjust sys.path to allow running as a script from repository root
sys_path_root = Path(__file__).resolve().parents[2]
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from src.core.hand_detector import HandDetector
from src.core.normalization import (
    DUAL_HAND_FEATURE_DIM,
    is_hand_present,
    normalize_dual_hands,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_dataset")


def discover_archive_entries(
    zip_ref: zipfile.ZipFile, prefix: str = "Indian/"
) -> Dict[str, List[str]]:
    """
    Scans the zip archive and groups valid image paths by class label.

    Args:
        zip_ref: Open ZipFile object.
        prefix: Directory prefix inside archive.

    Returns:
        Dict mapping class label (e.g. 'A', '1') to list of zip entry names.
    """
    valid_exts = {".jpg", ".jpeg", ".png"}
    class_entries: Dict[str, List[str]] = defaultdict(list)

    for entry in zip_ref.namelist():
        # Only inspect files inside the target folder
        if not entry.startswith(prefix) or entry.endswith("/"):
            continue

        ext = Path(entry).suffix.lower()
        if ext in valid_exts:
            parts = entry.split("/")
            # Expected pattern: Indian/<class_name>/<filename>.jpg
            if len(parts) >= 3:
                class_label = parts[1].strip()
                class_entries[class_label].append(entry)

    # Sort each class's file paths deterministically
    for cls in class_entries:
        class_entries[cls].sort()

    return dict(sorted(class_entries.items()))


def process_dataset(
    archive_path: Path,
    output_dir: Path,
    max_samples_per_class: Optional[int] = 200,
    min_detection_confidence: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Extracts normalized hand landmark features from images inside the zip archive.

    Args:
        archive_path: Path to the raw archive.zip.
        output_dir: Destination directory for processed .npy and .json artifacts.
        max_samples_per_class: Max images to process per class (None for all).
        min_detection_confidence: MediaPipe palm detection threshold.

    Returns:
        Tuple of (features array, labels array, metadata dictionary).
    """
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive dataset not found at: {archive_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Opening archive: %s", archive_path)
    with zipfile.ZipFile(archive_path, "r") as z:
        class_entries = discover_archive_entries(z, prefix="Indian/")
        total_available = sum(len(entries) for entries in class_entries.values())
        logger.info(
            "Found %d total images across %d classes in archive.",
            total_available,
            len(class_entries),
        )

        # Initialize MediaPipe detector in static image mode
        detector = HandDetector(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

        all_features: List[np.ndarray] = []
        all_labels: List[str] = []

        total_scanned = 0
        total_success = 0
        total_failed = 0

        hand_breakdown = {
            "left_only": 0,
            "right_only": 0,
            "dual_hands": 0,
        }
        per_class_stats: Dict[str, Dict[str, int]] = {}
        failed_samples: List[Dict[str, str]] = []

        start_time = time.time()

        for class_idx, (class_label, entries) in enumerate(class_entries.items(), 1):
            class_start = time.time()
            selected_entries = entries
            if max_samples_per_class is not None and max_samples_per_class > 0:
                selected_entries = entries[:max_samples_per_class]

            class_scanned = 0
            class_success = 0
            class_failed = 0
            class_left = 0
            class_right = 0
            class_dual = 0

            for entry_name in selected_entries:
                total_scanned += 1
                class_scanned += 1

                # Read image directly from zip without extracting to disk
                raw_bytes = z.read(entry_name)
                img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
                image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if image_bgr is None or image_bgr.size == 0:
                    class_failed += 1
                    total_failed += 1
                    if len(failed_samples) < 200:
                        failed_samples.append({
                            "path": entry_name,
                            "class": class_label,
                            "reason": "image_decode_failed",
                        })
                    continue

                # Run MediaPipe hand detection
                result = detector.process_frame(image_bgr, draw=False)

                if not result.detected:
                    class_failed += 1
                    total_failed += 1
                    if len(failed_samples) < 200:
                        failed_samples.append({
                            "path": entry_name,
                            "class": class_label,
                            "reason": "no_hands_detected",
                        })
                    continue

                # Extract normalized 126-D dual-hand feature vector
                feature_vector = normalize_dual_hands(
                    left_landmarks=result.left_landmarks,
                    right_landmarks=result.right_landmarks,
                )

                has_left = is_hand_present(feature_vector[:63])
                has_right = is_hand_present(feature_vector[63:])

                if has_left and has_right:
                    class_dual += 1
                    hand_breakdown["dual_hands"] += 1
                elif has_left:
                    class_left += 1
                    hand_breakdown["left_only"] += 1
                elif has_right:
                    class_right += 1
                    hand_breakdown["right_only"] += 1

                all_features.append(feature_vector)
                all_labels.append(class_label)
                class_success += 1
                total_success += 1

            per_class_stats[class_label] = {
                "scanned": class_scanned,
                "success": class_success,
                "failed": class_failed,
                "left_only": class_left,
                "right_only": class_right,
                "dual_hands": class_dual,
            }

            elapsed_cls = time.time() - class_start
            rate = class_scanned / max(elapsed_cls, 0.001)
            logger.info(
                "[%02d/%02d] Class '%s': %d/%d extracted (%.1f img/s) - dual: %d, left: %d, right: %d, fail: %d",
                class_idx,
                len(class_entries),
                class_label,
                class_success,
                class_scanned,
                rate,
                class_dual,
                class_left,
                class_right,
                class_failed,
            )

        detector.close()

    total_time = time.time() - start_time
    logger.info(
        "Feature extraction complete in %.2f s (%.1f img/s). Success: %d, Failed: %d (%.1f%%)",
        total_time,
        total_scanned / max(total_time, 0.001),
        total_success,
        total_failed,
        (total_success / max(total_scanned, 1)) * 100,
    )

    # Convert to numpy arrays
    features_np = np.array(all_features, dtype=np.float32)
    labels_np = np.array(all_labels, dtype=str)

    # Prepare extraction metadata
    metadata = {
        "source_archive": str(archive_path),
        "total_scanned": total_scanned,
        "total_success": total_success,
        "total_failed": total_failed,
        "success_rate_pct": round((total_success / max(total_scanned, 1)) * 100, 2),
        "feature_dim": DUAL_HAND_FEATURE_DIM,
        "max_samples_per_class": max_samples_per_class,
        "classes": list(class_entries.keys()),
        "num_classes": len(class_entries),
        "hand_breakdown": hand_breakdown,
        "per_class_stats": per_class_stats,
        "failed_samples_sample": failed_samples[:100],
        "extraction_duration_seconds": round(total_time, 2),
    }

    # Save to disk
    features_path = output_dir / "features.npy"
    labels_path = output_dir / "labels.npy"
    metadata_path = output_dir / "extraction_metadata.json"

    np.save(features_path, features_np)
    np.save(labels_path, labels_np)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved features to: %s (shape: %s)", features_path, features_np.shape)
    logger.info("Saved labels to: %s (shape: %s)", labels_path, labels_np.shape)
    logger.info("Saved extraction metadata to: %s", metadata_path)

    return features_np, labels_np, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from raw Indian Sign Language dataset."
    )
    parser.add_argument(
        "--archive_path",
        type=str,
        default="D:/project/Ai_sign language/archive.zip",
        help="Path to archive.zip",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed features and labels",
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=200,
        help="Max images per class to process (use 0 or -1 for all available images)",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum detection confidence",
    )

    args = parser.parse_args()

    max_samples = args.max_samples_per_class
    if max_samples is not None and max_samples <= 0:
        max_samples = None

    archive_p = Path(args.archive_path)
    output_p = Path(args.output_dir)

    process_dataset(
        archive_path=archive_p,
        output_dir=output_p,
        max_samples_per_class=max_samples,
        min_detection_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
