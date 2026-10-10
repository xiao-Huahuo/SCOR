"""Check input integrity and prevent false engineering-pass claims."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq
import pytest

from pilot_utils import SRC, atomic_json, check_profile, file_digest, read_json
from prepare_data import prepare
from verify_run import verify


def test_atomic_receipt_cannot_be_overwritten(tmp_path):
    path = tmp_path / "receipt.json"
    atomic_json(path, {"completed": True}, immutable=True)
    with pytest.raises(FileExistsError):
        atomic_json(path, {"completed": False}, immutable=True)
    assert read_json(path) == {"completed": True}
    assert not list(tmp_path.glob(".pending-*"))


def test_visual_data_has_disjoint_scenes_and_correct_labels(tmp_path):
    manifest = prepare(tmp_path / "data", train_count=8, val_count=4)
    train_ids, val_ids = set(), set()
    for sample in manifest["samples"]:
        values = sample["values"]
        expected = [values[0] + values[1], values[1] * values[2], max(values) - min(values)][sample["family"]]
        assert sample["answer"] == str(expected)
        (train_ids if sample["split"] == "train" else val_ids).add(sample["scene_id"])
    assert not train_ids & val_ids
    for split, info in manifest["splits"].items():
        path = tmp_path / "data" / info["file"]
        assert file_digest(path) == info["sha256"]
        rows = pq.read_table(path).to_pylist()
        assert len(rows) == info["rows"]
        assert all("<image>" in row["prompt"][0]["content"] for row in rows)
        assert all(row["images"][0]["bytes"].startswith(b"\x89PNG") for row in rows)


@pytest.mark.parametrize("key,value", [("trainer.n_gpus_per_node", 8),
                                      ("actor_rollout_ref.rollout.enable_verification", False),
                                      ("actor_rollout_ref.rollout.n", 1)])
def test_profile_refuses_to_drop_core_chain_or_exceed_gpu_limit(key, value):
    profile = read_json(SRC / "pilot_4090.json")
    profile["overrides"][key] = value
    with pytest.raises(ValueError):
        check_profile(profile)


def test_empty_run_cannot_be_reported_as_success(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "checkpoints").mkdir()
    atomic_json(tmp_path / "manifest.json", {"run_id": "empty", "limitations": []})
    atomic_json(tmp_path / "status.json", {"status": "completed", "returncode": 0})
    result = verify(tmp_path)
    assert not result["native_chain_observation_passed"]
    assert not result["six_gates_passed"]
    assert "No completed training updates" in result["problems"]
