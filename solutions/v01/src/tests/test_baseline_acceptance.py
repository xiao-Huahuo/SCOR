"""Reject incomplete, failed, stale and mismatched acceptance witnesses."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_acceptance import freeze
from pilot_utils import SRC, atomic_json, file_digest
from run_baseline import ACCEPTANCE_KEYS, ENGINE_FILES


def fixture(tmp_path):
    calibration = tmp_path / "calibration"
    profile = tmp_path / "profile.json"
    atomic_json(profile, {"overrides": {"actor.lr": .1}, "baseline_seeds": [1, 2, 3]})
    atomic_json(calibration / "manifest.json", {"overrides": {"actor.lr": .1},
        "learning": {"environment_signature": "E",
                     "source_sha256": {name: file_digest(SRC / name) for name in ENGINE_FILES}},
        "signatures": {"learning": "L"}, "model": {"signature": "M"}, "data_signature": "D",
        "checkpoint_plan": {"full_horizon": 32}})
    atomic_json(calibration / "status.json", {"status": "intentionally_stopped"})
    witnesses = {}
    for key in ACCEPTANCE_KEYS:
        witnesses[key] = tmp_path / (key + ".json")
        atomic_json(witnesses[key], {"passed": True, "learning_signature": "L"})
    return calibration, profile, witnesses


def test_empty_freeze_never_passes():
    with pytest.raises(ValueError, match="six"):
        freeze(Path("missing"), Path("missing"), {})


@pytest.mark.parametrize("key", sorted(ACCEPTANCE_KEYS))
def test_each_failed_witness_blocks_freeze(tmp_path, key):
    calibration, profile, witnesses = fixture(tmp_path)
    atomic_json(witnesses[key], {"passed": False, "learning_signature": "L"})
    with pytest.raises(ValueError, match="failed"):
        freeze(calibration, profile, witnesses)


@pytest.mark.parametrize("damage", ["running", "profile", "stale_evidence"])
def test_mismatched_calibration_never_freezes(tmp_path, damage):
    calibration, profile, witnesses = fixture(tmp_path)
    if damage == "running":
        atomic_json(calibration / "status.json", {"status": "running"})
    elif damage == "profile":
        atomic_json(profile, {"overrides": {"actor.lr": .2}, "baseline_seeds": [1, 2, 3]})
    else:
        atomic_json(next(iter(witnesses.values())), {"passed": True, "learning_signature": "OLD"})
    with pytest.raises(ValueError):
        freeze(calibration, profile, witnesses)
