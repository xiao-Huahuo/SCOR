"""Explicit cross-version compatibility and fixed-unit evidence for Gate1."""
from __future__ import annotations

from pathlib import Path

from pilot_utils import SRC, digest, file_digest, read_json
from records import value_hash


def historical_output_matches(output, group) -> None:
    required = {"prompts", "responses", "input_ids", "attention_mask", "position_ids", "multiturn_mask"}
    metadata = {"baseline_case_id", "baseline_slot", "baseline_segments", "step_data", "verify_probs",
                "num_steps", "num_repairs", "final_answers", "final_generation_step"}
    if set(output.batch.keys()) != required or not metadata.issubset(output.non_tensor_batch):
        raise ValueError("Historical generation omits visual/token/mask fields")
    for key, value in output.batch.items():
        if key not in group.batch or value_hash(value) != value_hash(group.batch[key]):
            raise ValueError("Historical generation differs from committed unit")
    for key, value in output.non_tensor_batch.items():
        if key not in group.non_tensor_batch:
            raise ValueError("Historical generation metadata is missing")
        target = group.non_tensor_batch[key]
        if key in {"num_steps", "num_repairs"} and str(getattr(value, "dtype", None)) == "int32" and str(getattr(target, "dtype", None)) == "int64":
            value = value.astype(target.dtype)
        if value_hash(value) != value_hash(target):
            raise ValueError("Historical generation visual/role/group metadata differs")


def compatible_learning(current: dict, source: dict, canonical: dict) -> dict:
    """Accept only the reviewed harness derivative, never a changed learner."""
    if {k: v for k, v in current.items() if k != "source_sha256"} != {
            k: v for k, v in source.items() if k != "source_sha256"}:
        raise ValueError("Cross-version native parameters/model/data/environment/seed differ")
    if source["source_sha256"] != canonical:
        raise ValueError("Source is not the accepted frozen v01 engine")
    if set(current["source_sha256"]) != set(canonical) | {
            "run_gate1.py", "gate1_contract.py", "gate1_probes.py"}:
        raise ValueError("Execution source omits required engine or adds an unreviewed module")
    for name, expected in canonical.items():
        if name != "native_adapter.py" and current["source_sha256"].get(name) != expected:
            raise ValueError("Unreviewed change to frozen engine: " + name)
    if any(file_digest(SRC / name) != sha for name, sha in current["source_sha256"].items()):
        raise ValueError("Current execution source changed")
    return {"source_learning_signature": digest(source),
            "execution_learning_signature": digest(current),
            "equal_conditions": sorted(k for k in current if k != "source_sha256"),
            "canonical_source_sha256": canonical,
            "execution_source_sha256": current["source_sha256"],
            "review_required_derivative": "native_adapter.py plus recorded gate1 modules"}


def fixed_unit_audit(data, source: Path, step: int, source_learning: str,
                     *, injected: bool = False) -> dict:
    import torch
    from records import prompt_hash
    group_path = source / f"raw/group_step_{step:06d}.json"
    receipt = read_json(source / f"raw/completed_step_{step:06d}.json")
    if receipt["learning_signature"] != source_learning or receipt["group_sha256"] != file_digest(group_path):
        raise ValueError("Fixed unit source completion differs")
    group = torch.load(source / f"raw/training_group_step_{step:06d}.pt", map_location="cpu", weights_only=False)
    if prompt_hash(group) != read_json(group_path)["payload_hash"]:
        raise ValueError("Fixed unit source payload is corrupted")
    if set(data.batch.keys()) != set(group.batch.keys()) or set(data.non_tensor_batch) != set(group.non_tensor_batch):
        raise ValueError("Fixed unit schema differs")
    variable = {"old_log_probs", "entropy"}
    if injected:
        variable |= {"token_level_scores", "token_level_rewards", "advantages", "returns"}
    compared = {k: value_hash(data.batch[k]) == value_hash(group.batch[k])
                for k in group.batch.keys() if k not in variable}
    metadata = {k: value_hash(data.non_tensor_batch[k]) == value_hash(group.non_tensor_batch[k])
                for k in group.non_tensor_batch}
    if not all(compared.values()) or not all(metadata.values()):
        raise ValueError("Fixed tokens/images/masks/rewards/advantages/reference/group differ")
    if not {"token_level_scores", "token_level_rewards", "advantages", "returns"}.issubset(group.batch.keys()):
        raise ValueError("Fixed unit omits native reward and advantage tensors")
    if injected:
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
        delta = data.batch["token_level_scores"] - group.batch["token_level_scores"]
        changed = delta.nonzero()
        if changed.shape[0] != 1 or delta[tuple(changed[0])].item() != 1:
            raise ValueError("Intervention must change exactly one terminal reward by +1")
        row, position = changed[0].tolist()
        active = data.batch["attention_mask"][row, -delta.shape[-1]:].bool().nonzero().flatten()
        if position != active[-1].item():
            raise ValueError("Intervention was not at the trajectory terminal position")
        if value_hash(data.batch["token_level_rewards"]) != value_hash(data.batch["token_level_scores"]):
            raise ValueError("Frozen mixed reward protocol differs")
        mask = data.batch["response_mask"]
        expected, returns = compute_grpo_outcome_advantage(data.batch["token_level_rewards"], mask,
                                                          data.non_tensor_batch["uid"])
        if value_hash(expected) != value_hash(data.batch["advantages"]) or value_hash(returns) != value_hash(data.batch["returns"]):
            raise ValueError("Whole-group native GRPO recomputation differs")
        ids = data.non_tensor_batch["uid"]
        other = [i for i, uid in enumerate(ids) if uid != ids[row]]
        if value_hash(data.batch["advantages"][other]) != value_hash(group.batch["advantages"][other]):
            raise ValueError("Intervention changed another GRPO group")
    return {"step": step, "source_learning_signature": source_learning,
            "source_group_sha256": file_digest(source / f"raw/training_group_step_{step:06d}.pt"),
            "fixed_tensor_checks": compared, "fixed_metadata_checks": metadata,
            "injected": injected, "passed": True,
            "historical_actor_logp_hash": value_hash(group.batch["old_log_probs"]),
            "current_anchor_logp_hash": value_hash(data.batch["old_log_probs"]),
            "score_source": "native actor reanchor; not importance-corrected sampler probability"}


def score_distance(left: dict, right: dict) -> dict:
    import math
    from collections import defaultdict
    def keyed(payload):
        values = {}
        for item in payload["cases"]:
            key = (item["case_id"], item["role"], item["candidate_id"])
            if key in values:
                raise ValueError("Duplicate fixed probe identity")
            values[key] = item
        return values
    a, b = keyed(left), keyed(right)
    if not a or set(a) != set(b) or left["probe_signature"] != right["probe_signature"]:
        raise ValueError("Fixed probe set differs")
    sums = defaultdict(list)
    for key, x in a.items():
        y = b[key]
        if {k: v for k, v in x.items() if k != "token_logp"} != {k: v for k, v in y.items() if k != "token_logp"}:
            raise ValueError("Probe image/prefix/candidate/mask differs")
        p, q = x["token_logp"], y["token_logp"]
        if not p or len(p) != len(q) or not all(math.isfinite(v) for v in p + q):
            raise ValueError("Invalid fixed token scores")
        sums[(x["base_case_id"], x["role"])].append(sum((u-v)**2 for u, v in zip(p, q))/len(p))
    roles = defaultdict(list)
    for (_, role), values in sums.items():
        roles[role].append(math.sqrt(sum(values)/len(values)))
    if set(roles) != {"solver", "verifier"} or {
            case for case, role in sums if role == "solver"} != {
            case for case, role in sums if role == "verifier"}:
        raise ValueError("Both roles require the same complete base-case set")
    by_role = {role: sum(values)/len(values) for role, values in roles.items()}
    return {"D_score": sum(by_role.values())/2, "roles": by_role,
            "base_cases": len(roles["solver"]), "probe_signature": left["probe_signature"]}


def paired_interval(values: list[float], epsilon: float) -> dict:
    import math
    import statistics
    if not math.isfinite(epsilon) or epsilon <= 0 or not all(math.isfinite(x) for x in values):
        raise ValueError("Invalid paired effect or frozen tolerance")
    n = len(values)
    if not 1 <= n <= 5:
        raise ValueError("Only the predeclared five training seeds are allowed")
    mean = statistics.mean(values)
    critical = {2: 12.706204736432095, 3: 4.302652729696142,
                4: 3.182446305284263, 5: 2.7764451051977987}
    if n == 1:
        return {"n": n, "mean": mean, "lower": None, "upper": None, "decision": "uncertain"}
    radius = critical[n]*statistics.stdev(values)/math.sqrt(n)
    lower, upper = mean-radius, mean+radius
    decision = "uncertain" if n < 3 else ("supported" if lower > epsilon else (
        "within_declared_tolerance" if upper <= epsilon else "uncertain"))
    return {"n": n, "mean": mean, "lower": lower, "upper": upper, "epsilon_D": epsilon,
            "decision": decision, "method": "two-sided95% paired Student-t; small-sample distribution assumption"}
