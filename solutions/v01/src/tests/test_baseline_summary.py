"""Counterexamples for seed-level baseline reporting."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_baseline_summary import normal_results


def reports():
    result = []
    for seed, final in zip((20261011, 20261012, 20261013), (128, 127, 129)):
        evaluations = {str(step): {"cases": 256, "correct": final if step == 32 else 128,
            "accuracy": (final if step == 32 else 128) / 256, "native_accuracy": .5,
            "explicit_answer_rate": 1., "truncation_rate": 0.} for step in (0, 8, 16, 24, 32)}
        result.append({"seed": seed, "run_id": str(seed), "complete_units": 32, "evaluations": evaluations})
    return result


def test_descriptive_effect_uses_three_seeds_with_percentage_point_units():
    rows, summary = normal_results(reports())
    assert len(rows) == 15
    assert summary["seed_count"] == 3
    assert summary["final_minus_initial_percentage_points"] == [0., -0.390625, 0.390625]
    assert summary["mean_delta_percentage_points"] == 0.


@pytest.mark.parametrize("damage", ["missing_seed", "repeated_seed", "missing_evaluation", "inconsistent_accuracy"])
def test_partial_or_inconsistent_results_cannot_be_reported_as_complete(damage):
    data = deepcopy(reports())
    if damage == "missing_seed":
        data.pop()
    elif damage == "repeated_seed":
        data[2]["seed"] = data[1]["seed"]
    elif damage == "missing_evaluation":
        data[2]["evaluations"].pop("24")
    else:
        data[2]["evaluations"]["32"]["accuracy"] = 1.
    with pytest.raises(ValueError):
        normal_results(data)
