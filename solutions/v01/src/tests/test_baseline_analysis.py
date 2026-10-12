"""Counterexamples for the independent completed-GPU paired-unit audit."""
from pathlib import Path
import copy
import sys

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_baseline import audit_unit
from interventions import false_positive
from pilot_utils import ROOT
sys.path.insert(0, str(ROOT / ".runtime/upstream/Agent0/Agent0-VL"))
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from tensordict import TensorDict


def pair():
    scores = torch.tensor([[0., 0., 0., x] for x in (0, .2, 1, .6, .2, .4, .6, .8)])
    ids = np.array(["group0"] * 4 + ["group1"] * 4, dtype=object)
    metadata = {"uid": ids, "baseline_slot": np.arange(8) % 4,
                "acc": np.array([0, 1, 0, 1, 1, 1, 1, 1]),
                "extra_info": np.array([{"case_id": "task" + str(i // 4), "answer": "5"} for i in range(8)], dtype=object)}
    segments = np.empty(8, dtype=object)
    for i, answer in enumerate([9, 5, 8, 5, 5, 5, 5, 5]):
        segments[i] = [{"role": "solver", "text": f"\\boxed{{{answer}}}", "truncated": False}]
    metadata["baseline_segments"] = segments
    mask = torch.ones_like(scores)
    adv, returns = compute_grpo_outcome_advantage(scores.clone(), mask, ids)
    clean = DataProto(batch=TensorDict({"token_level_scores": scores, "token_level_rewards": scores.clone(),
        "advantages": adv, "returns": returns, "response_mask": mask,
        "attention_mask": torch.ones(8, 8, dtype=torch.long), "responses": torch.ones(8, 4, dtype=torch.long)}, batch_size=[8]),
        non_tensor_batch=metadata)
    revised, certificate = false_positive(clean, scores, metadata["acc"], step=8, alpha_out=1)
    changed = copy.deepcopy(clean)
    changed.batch["token_level_scores"] = revised
    changed.batch["token_level_rewards"] = revised.clone()
    changed.batch["advantages"], changed.batch["returns"] = compute_grpo_outcome_advantage(revised.clone(), mask, ids)
    return clean, changed, certificate


def test_real_native_group_formula_matches_independent_pair_oracle():
    result = audit_unit(*pair())
    assert result["passed"] and result["changed_reward_position"] == [0, 3]
    assert result["other_groups_exact"] and len(result["rows"]) == 16


@pytest.mark.parametrize("damage", ["second_reward", "fixed_token", "dtype", "other_group", "wrong_identity", "semantic_correct"])
def test_pair_audit_rejects_extra_changes_and_ineligible_target(damage):
    clean, changed, certificate = pair()
    if damage == "second_reward":
        changed.batch["token_level_rewards"][1, 3] += 1
    elif damage == "fixed_token":
        changed.batch["responses"][0, 0] += 1
    elif damage == "dtype":
        changed.batch["responses"] = changed.batch["responses"].float()
    elif damage == "other_group":
        changed.batch["advantages"][4, 0] += .1
    elif damage == "wrong_identity":
        certificate["trajectory_id"] = "8:task0:2"
    elif damage == "semantic_correct":
        clean.non_tensor_batch["baseline_segments"][0][0]["text"] = "\\boxed{5}"
        changed.non_tensor_batch = copy.deepcopy(clean.non_tensor_batch)
    with pytest.raises(ValueError):
        audit_unit(clean, changed, certificate)


def test_no_eligible_target_is_recorded_and_cannot_hide_a_modified_unit():
    clean, _, _ = pair()
    clean.non_tensor_batch["acc"][:] = 1
    changed = copy.deepcopy(clean)
    result = audit_unit(clean, changed, {"status": "no_eligible_target"})
    assert not result["passed"] and result["status"] == "no_eligible_target"
    changed.batch["token_level_scores"][0, 0] += 1
    with pytest.raises(ValueError, match="No-target"):
        audit_unit(clean, changed, {"status": "no_eligible_target"})


def test_no_target_certificate_cannot_skip_an_actual_eligible_error():
    clean, _, _ = pair()
    with pytest.raises(ValueError, match="skipped"):
        audit_unit(clean, copy.deepcopy(clean), {"status": "no_eligible_target"})
