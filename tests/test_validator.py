"""
Unit Tests for Dataset Validator Module.
========================================

Author: Senior AI/ML Engineer (Mentoring MCA Fresher)
Target: sign_language_v2.src.data.dataset_validator
Test Runner: pytest
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.data.dataset_validator import DatasetValidator


@pytest.fixture
def temp_sample_dir(tmp_path):
    """Creates a temporary directory with valid and invalid sample JSON records."""
    valid_record = {
        "version": "2.0",
        "sample_id": "test001",
        "label": "A",
        "participant_id": "P01",
        "session_id": "S01",
        "timestamp": "2026-11-20T12:00:00Z",
        "hand_config": "two_hand",
        "hands_count": 2,
        "raw_landmarks": {
            "left_hand": np.ones((21, 3)).tolist(),
            "right_hand": np.ones((21, 3)).tolist(),
        },
        "normalized_features": np.full(126, 0.5, dtype=np.float32).tolist(),
    }

    # 1. Valid file
    valid_file = tmp_path / "valid_sample.json"
    with open(valid_file, "w", encoding="utf-8") as f:
        json.dump(valid_record, f)

    # 2. Corrupted file: missing required key 'label'
    corrupted_record = valid_record.copy()
    del corrupted_record["label"]
    corrupted_file = tmp_path / "missing_label.json"
    with open(corrupted_file, "w", encoding="utf-8") as f:
        json.dump(corrupted_record, f)

    # 3. Corrupted file: NaN in normalized features
    nan_record = valid_record.copy()
    nan_features = [float("nan")] * 126
    nan_record["normalized_features"] = nan_features
    nan_file = tmp_path / "nan_features.json"
    with open(nan_file, "w", encoding="utf-8") as f:
        json.dump(nan_record, f)

    # 4. Corrupted file: wrong feature dimension (50 instead of 126)
    dim_record = valid_record.copy()
    dim_record["normalized_features"] = [0.1] * 50
    dim_file = tmp_path / "wrong_dim.json"
    with open(dim_file, "w", encoding="utf-8") as f:
        json.dump(dim_record, f)

    return tmp_path, valid_file, corrupted_file, nan_file, dim_file


def test_validator_detects_valid_and_invalid(temp_sample_dir):
    """Verifies that the validator accurately distinguishes valid from corrupted records."""
    tmp_path, valid_file, corrupted_file, nan_file, dim_file = temp_sample_dir

    validator = DatasetValidator(data_dir=str(tmp_path))

    # Test individual validation
    is_valid, errors = validator.validate_single_record(valid_file)
    assert is_valid is True
    assert len(errors) == 0

    is_valid_c, errors_c = validator.validate_single_record(corrupted_file)
    assert is_valid_c is False
    assert any("label" in err.lower() for err in errors_c)

    is_valid_nan, errors_nan = validator.validate_single_record(nan_file)
    assert is_valid_nan is False
    assert any("nan" in err.lower() for err in errors_nan)

    is_valid_dim, errors_dim = validator.validate_single_record(dim_file)
    assert is_valid_dim is False
    assert any("length" in err.lower() or "dimension" in err.lower() for err in errors_dim)

    # Test full dataset scan
    summary = validator.validate_dataset()
    assert summary["total_files_scanned"] == 4
    assert summary["valid_samples"] == 1
    assert summary["corrupted_samples"] == 3
