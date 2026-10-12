"""Rebuild completed normal-run reports and the actual paired reward audit."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys

from baseline_acceptance import costs, evidence
from baseline_costs import logical_role_counts
from checkpoint_store import validate
from evaluate import solver_answer, integer_answer, write_report
from pilot_utils import ROOT, atomic_json, file_digest, read_json
from records import trajectory_id, value_hash


def audit_unit(clean, changed, revision: dict) -> dict:
    import torch
    mutable = {"token_level_scores", "token_level_rewards", "advantages", "returns"}
    if set(clean.batch.keys()) != set(changed.batch.keys()):
        raise ValueError("Paired unit tensor keys differ")
    if value_hash(clean.non_tensor_batch) != value_hash(changed.non_tensor_batch):
        raise ValueError("Paired visual/group/role inputs differ")
    exact = {key: value_hash(value) == value_hash(changed.batch[key])
             for key, value in clean.batch.items() if key not in mutable}
    if not all(exact.values()):
        raise ValueError("Paired fixed training inputs differ")
    if revision["status"] == "no_eligible_target":
        for i, (candidate, segments) in enumerate(zip(clean.non_tensor_batch["extra_info"],
                                                     clean.non_tensor_batch["baseline_segments"])):
            explicit = solver_answer(segments)
            if (explicit is not None and explicit != integer_answer(candidate["answer"]) and
                    clean.non_tensor_batch["acc"][i] == 0 and not any(s["truncated"] for s in segments)):
                raise ValueError("No-target certificate skipped an eligible semantic error")
        if any(value_hash(v) != value_hash(changed.batch[k]) for k, v in clean.batch.items()):
            raise ValueError("No-target unit was changed")
        return {"passed": False, "status": "no_eligible_target", "fixed_inputs_exact": exact}
    if (revision["status"] != "applied" or revision["component"] != "terminal_outcome" or
            revision["old_outcome"] != 0 or revision["new_outcome"] != 1 or revision["delta_return"] != 1):
        raise ValueError("Pair is not the declared terminal false-positive")
    row = revision["row"]
    segments = clean.non_tensor_batch["baseline_segments"][row]
    info = clean.non_tensor_batch["extra_info"][row]
    answer = solver_answer(segments)
    if answer is None or answer == integer_answer(info["answer"]) or any(s["truncated"] for s in segments):
        raise ValueError("Target is not an explicit nontruncated semantic error")
    eligible = []
    for i, (candidate, role_segments) in enumerate(zip(clean.non_tensor_batch["extra_info"],
                                                       clean.non_tensor_batch["baseline_segments"])):
        explicit = solver_answer(role_segments)
        if (explicit is not None and explicit != integer_answer(candidate["answer"]) and
                clean.non_tensor_batch["acc"][i] == 0 and not any(s["truncated"] for s in role_segments)):
            stable = trajectory_id(revision["step"], candidate["case_id"], int(clean.non_tensor_batch["baseline_slot"][i]))
            eligible.append((stable, i))
    if not eligible or min(eligible) != (revision["trajectory_id"], row):
        raise ValueError("Pair did not select the first eligible stable semantic-error identity")
    positions = clean.batch["attention_mask"][row, -clean.batch["token_level_scores"].shape[-1]:].bool().nonzero().flatten()
    terminal = int(positions[-1])
    for key in ("token_level_scores", "token_level_rewards"):
        expected = clean.batch[key].clone()
        expected[row, terminal] += 1
        if value_hash(expected) != value_hash(changed.batch[key]):
            raise ValueError("Pair changes more than the target terminal reward")
    ids = clean.non_tensor_batch["uid"].tolist()
    target = ids[row]
    rows = []
    # Independent Python sample-standard-deviation oracle, fixed before E data.
    for unit, label in ((clean, "clean"), (changed, "changed")):
        returns = unit.batch["token_level_rewards"].sum(-1).tolist()
        for group in sorted(set(ids)):
            members = [i for i, identity in enumerate(ids) if identity == group]
            values = [returns[i] for i in members]
            if len(members) != 4:
                raise ValueError("Paired GRPO group is incomplete")
            mean, std = statistics.mean(values), statistics.stdev(values)
            for i in members:
                expected = torch.tensor((returns[i] - mean) / (std + 1e-6), dtype=unit.batch["advantages"].dtype)
                expected = expected * unit.batch["response_mask"][i]
                for key in ("advantages", "returns"):
                    if not torch.allclose(unit.batch[key][i], expected, atol=1e-6, rtol=1e-5):
                        raise ValueError("Recorded advantage differs from independent group oracle")
                rows.append({"branch": label, "row": i, "group": ids[i], "target_group": ids[i] == target,
                             "return": returns[i], "normalized_advantage": (returns[i] - mean) / (std + 1e-6)})
    others = [i for i, identity in enumerate(ids) if identity != target]
    for key in mutable:
        if value_hash(clean.batch[key][others]) != value_hash(changed.batch[key][others]):
            raise ValueError("Pair changes another prompt group")
    return {"passed": True, "status": "applied", "fixed_inputs_exact": exact,
            "changed_reward_position": [row, terminal], "whole_group_oracle_matches": True,
            "other_groups_exact": True, "rows": rows, "oracle_tolerance": {"atol": 1e-6, "rtol": 1e-5}}


def paired_audit(clean: Path, changed: Path, step: int) -> dict:
    sys.path.insert(0, str(ROOT / ".runtime/upstream/Agent0/Agent0-VL"))
    import torch
    a, b = (read_json(p / "manifest.json") for p in (clean, changed))
    if a["signatures"]["learning"] != b["signatures"]["learning"]:
        raise ValueError("Paired learning identities differ")
    if read_json(changed / "status.json") != {"status": "intentionally_stopped", "returncode": 0}:
        raise ValueError("Paired GPU update has not finished successfully")
    parent = validate(clean / f"checkpoints/global_step_{step - 1}")
    loaded = read_json(changed / "raw/resume_loaded.json")
    if loaded["parent_state"] != parent["state_signature"] or not loaded["full_worker_state_fingerprints_match"]:
        raise ValueError("Paired branch did not restore the same full parent")
    snapshots = [validate(p / f"checkpoints/global_step_{step}") for p in (clean, changed)]
    paths = [p / f"raw/training_group_step_{step:06d}.pt" for p in (clean, changed)]
    units = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    reward = read_json(changed / f"raw/reward_step_{step:06d}.json")
    result = audit_unit(*units, reward["revision"])
    if result["passed"] and reward["native_info"]["acc"][reward["revision"]["row"]] != 0:
        raise ValueError("Target native outcome was not zero")
    clean_metrics, changed_metrics = (read_json(p / f"raw/metrics_step_{step:06d}.json") for p in (clean, changed))
    if (changed_metrics["baseline/optimizer_step"] != step or changed_metrics["baseline/lr_used"] <= 0 or
            changed_metrics["baseline/lr_used"] != clean_metrics["baseline/lr_used"]):
        raise ValueError("Paired branch used a different optimizer position or learning rate")
    workers = [read_json(p / f"checkpoints/global_step_{step}/baseline_worker_state.json") for p in (clean, changed)]
    result.update({"step": step, "clean": a["run_id"], "changed": b["run_id"],
                   "learning_signature": a["signatures"]["learning"], "revision": reward["revision"],
                   "full_GPU_update_completed": True, "parent_state": parent["state_signature"],
                   "updated_state_signatures": [s["state_signature"] for s in snapshots],
                   "updated_model_differs": [w["model"] for w in workers[0]] != [w["model"] for w in workers[1]],
                   "finite_updated_states": all(w["all_parameters_and_optimizer_finite"] for ws in workers for w in ws),
                   "actual_unit_evidence": [evidence(p) for p in paths], "source_sha256": file_digest(Path(__file__)),
                   "six_gates_passed": False})
    return result


def summarize(run: Path) -> dict:
    manifest = read_json(run / "manifest.json")
    if (manifest["phase"] != "baseline" or manifest["checkpoint_plan"]["end"] != 32 or
            read_json(run / "status.json") != {"status": "completed", "returncode": 0}):
        raise ValueError("Require a successfully completed full32 normal baseline")
    steps = []
    for step in range(1, 33):
        completed = read_json(run / f"raw/completed_step_{step:06d}.json")
        group = run / f"raw/group_step_{step:06d}.json"
        if completed["group_sha256"] != file_digest(group) or completed["learning_signature"] != manifest["signatures"]["learning"]:
            raise ValueError("Completed unit evidence identity differs")
        metrics = read_json(run / f"raw/metrics_step_{step:06d}.json")
        if metrics["baseline/optimizer_step"] != step or metrics["baseline/lr_used"] <= 0:
            raise ValueError("Normal run optimizer position or actual used LR differs")
        steps.append({"step": step, "used_lr": metrics["baseline/lr_used"], "grad_norm": metrics["actor/grad_norm"],
                      "unit_seconds": metrics["timing_s/step"], **read_json(group)})
    final = validate(run / "checkpoints/global_step_32")
    evaluations = {}
    for step in (0, 8, 16, 24, 32):
        p = run / f"raw/evaluation_step_{step:06d}.json"
        receipt = read_json(p)
        if receipt["split"] != "evaluation" or len(receipt["cases"]) != 256:
            raise ValueError("Normal evaluation split or sample count differs")
        evaluations[str(step)] = write_report(p, run / f"processed/evaluation_step_{step:06d}")
    with (run / "processed/updates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(steps[0])); writer.writeheader(); writer.writerows(steps)
    return {"run_id": manifest["run_id"], "seed": manifest["seed"], "complete_units": 32,
            "evaluations": evaluations, "final_state_signature": final["state_signature"], "costs": costs(run),
            "logical_rollout_costs": logical_role_counts(run),
            "raw_phase_role_count_scope": "Native cost phase_role counts are per worker, including TP duplicates; use logical_rollout_costs for unique requests/tokens",
            "analysis_source_sha256": file_digest(Path(__file__)), "six_gates_passed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["summarize", "pair"])
    parser.add_argument("run", type=Path)
    parser.add_argument("--changed", type=Path)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run) if args.operation == "summarize" else paired_audit(args.run, args.changed, args.step)
    atomic_json(args.output, result, immutable=True)
    print(__import__('json').dumps({k: v for k, v in result.items() if k not in ('rows', 'costs')}, ensure_ascii=False))
