"""TP duplication and repeated-evaluation cost counterexamples."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_costs import logical_role_counts
from pilot_utils import atomic_json


def fixture(tmp_path):
    atomic_json(tmp_path / "manifest.json", {"overrides": {"actor_rollout_ref.rollout.tensor_model_parallel_size": 2}})
    for call in (0, 1):
        for rank in (0, 1):
            record = {"phase": "eval", "step": 0, "role": "solver", "inner_step": 0, "rank": rank,
                      "requests": [{"case_id": "same_case", "output_token_ids": [3, 4, 5]}]}
            atomic_json(tmp_path / f"raw/roles/eval_rank_{rank}_call_{call:06d}.json", record)
    return tmp_path


def test_tp_output_tokens_count_once_but_repeat_evaluation_counts_twice(tmp_path):
    report = logical_role_counts(fixture(tmp_path))
    assert report["phase_role"]["eval/solver"] == {"logical_batch_calls": 2, "logical_requests": 2, "generated_tokens": 6}


@pytest.mark.parametrize("damage", ["missing_rank", "conflicting_output"])
def test_cost_cannot_certify_missing_or_disagreeing_worker_records(tmp_path, damage):
    run = fixture(tmp_path)
    path = run / "raw/roles/eval_rank_1_call_000001.json"
    if damage == "missing_rank":
        path.unlink()
    else:
        atomic_json(path, {"phase": "eval", "step": 0, "role": "solver", "inner_step": 0, "rank": 1,
                          "requests": [{"case_id": "same_case", "output_token_ids": [9]}]})
    with pytest.raises(ValueError):
        logical_role_counts(run)
