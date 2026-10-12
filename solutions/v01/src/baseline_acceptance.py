"""Rebuild baseline acceptance and cost evidence from native raw records."""
from __future__ import annotations

import argparse
from pathlib import Path

from checkpoint_store import validate
from pilot_utils import ROOT, SRC, atomic_json, file_digest, read_json
from run_baseline import ACCEPTANCE_KEYS, ENGINE_FILES


def evidence(path: Path) -> dict:
    return {"path": str(path.resolve().relative_to(ROOT)), "sha256": file_digest(path)}


def role_masks_match(batch) -> bool:
    import torch
    generated = {"solver", "verifier", "repair", "regenerate"}
    inserted = {"tool_observation", "verifier_prompt", "repair_prompt", "regenerate_prompt", "solver_prompt"}
    for row, segments in enumerate(batch.non_tensor_batch["baseline_segments"]):
        expected = torch.zeros_like(batch.batch["multiturn_mask"][row], dtype=torch.bool)
        cursor = 0
        for segment in segments:
            start, end, role = segment["start"], segment["end"], segment["role"]
            if role not in generated | inserted or start != cursor or not start < end <= expected.numel():
                return False
            if bool(segment["trainable"]) != (role in generated):
                return False
            if batch.batch["responses"][row, start:end].tolist() != segment["token_ids"]:
                return False
            expected[start:end] = role in generated
            cursor = end
        attention = batch.batch["attention_mask"][row, -expected.numel():].bool()
        covered = torch.arange(expected.numel(), device=attention.device) < cursor
        if not torch.equal(attention, covered) or not torch.equal(expected, batch.batch["multiturn_mask"][row].bool()):
            return False
    return True


def health(run: Path) -> dict:
    import torch
    manifest = read_json(run / "manifest.json")
    end = manifest["checkpoint_plan"]["end"]
    initial = read_json(run / "raw/evaluation_step_000000.json")
    final = read_json(run / f"raw/evaluation_step_{end:06d}.json")
    groups, tools, repairs, truncations, trajectories = [], 0, 0, 0, 0
    masks_valid = True
    for step in range(1, end + 1):
        batch = torch.load(run / f"raw/training_group_step_{step:06d}.pt", map_location="cpu", weights_only=False)
        masks_valid = masks_valid and role_masks_match(batch)
        ids = batch.non_tensor_batch["uid"].tolist()
        mask = batch.batch["multiturn_mask"].bool()
        nonzero = (batch.batch["advantages"] != 0) & mask
        unique = sorted(set(ids))
        groups.append(sum(bool(nonzero[[i for i, value in enumerate(ids) if value == group]].any())
                          for group in unique) / len(unique))
        for segments, steps in zip(batch.non_tensor_batch["baseline_segments"], batch.non_tensor_batch["step_data"]):
            tools += sum(bool(s["tool_used"]) for s in steps)
            repairs += sum(bool(s["was_repaired"]) for s in steps)
            truncations += int(any(s["truncated"] for s in segments))
            trajectories += 1
    workers0 = read_json(run / "checkpoints/global_step_0/baseline_worker_state.json")
    workers1 = read_json(run / f"checkpoints/global_step_{end}/baseline_worker_state.json")
    metrics = [read_json(run / f"raw/metrics_step_{step:06d}.json") for step in range(1, end + 1)]
    checks = {
        "calibration_only": manifest["evaluation_split"] == "calibration",
        "explicit_answers": min(x["summary"]["explicit_answer_rate"] for x in (initial, final)) >= .9,
        "verifier_json": min(x["verifier_probes"]["summary"]["parse_rate"] for x in (initial, final)) >= .9,
        "task_headroom": all(.1 <= f["accuracy"] <= .9 for f in initial["summary"]["families"].values()),
        "evaluation_truncation": max(x["summary"]["truncation_rate"] for x in (initial, final)) <= .05,
        "training_truncation": truncations / trajectories <= .05,
        "actual_tools_and_natural_repair": tools > 0 and repairs > 0,
        "nonzero_groups": sum(groups) / len(groups) >= .2,
        "actual_parameter_change": any(a["model"] != b["model"] for a, b in zip(workers0, workers1)),
        "finite_complete_state": all(w["all_parameters_and_optimizer_finite"] for w in workers0 + workers1),
        "role_masks_and_token_boundaries": masks_valid,
        "one_actual_optimizer_step_per_unit": all(m["baseline/optimizer_step"] == step
                                                  for step, m in enumerate(metrics, start=1)),
        "positive_used_lr": all(m["baseline/lr_used"] > 0 for m in metrics),
        "rng_isolation": all(x["worker_rng_restored"] and x["driver_rng_restored"] and x["data_cursor_unchanged"]
                             for x in (initial, final)),
    }
    return {"passed": all(checks.values()), "checks": checks, "run_id": manifest["run_id"],
            "learning_signature": manifest["signatures"]["learning"],
            "initial": initial["summary"], "final": final["summary"],
            "initial_verifier": initial["verifier_probes"]["summary"],
            "final_verifier": final["verifier_probes"]["summary"],
            "tools": tools, "natural_repairs": repairs, "nonzero_group_fractions": groups,
            "training_truncation_rate": truncations / trajectories,
            "threshold_origin": "baseline_design section5; engineering rules adopted before cal02 results; no held-out selection"}


def compare(left: Path, right: Path, step: int) -> dict:
    import torch
    from records import value_hash
    first, second = (read_json(p / "manifest.json") for p in (left, right))
    if first["signatures"]["learning"] != second["signatures"]["learning"]:
        raise ValueError("Comparison learning identities differ")
    a, b = (torch.load(p / f"raw/training_group_step_{step:06d}.pt", map_location="cpu", weights_only=False)
            for p in (left, right))
    tensor_equal = {key: key in b.batch and value_hash(value) == value_hash(b.batch[key])
                    for key, value in a.batch.items()}
    wa, wb = (read_json(p / f"checkpoints/global_step_{step}/baseline_worker_state.json") for p in (left, right))
    state_equal = {key: [w[key] for w in wa] == [w[key] for w in wb]
                   for key in ("model", "buffers", "optimizer", "optimizer_groups", "scheduler")}
    checks = {"tensor_keys": set(a.batch.keys()) == set(b.batch.keys()),
              "all_training_tensors_exact": all(tensor_equal.values()),
              "group_and_visual_inputs_exact": value_hash(a.non_tensor_batch) == value_hash(b.non_tensor_batch),
              "all_updated_state_exact": all(state_equal.values())}
    return {"passed": all(checks.values()), "checks": checks, "step": step,
            "learning_signature": first["signatures"]["learning"],
            "left": first["run_id"], "right": second["run_id"], "tensor_exact": tensor_equal,
            "updated_state_exact": state_equal,
            "worker_rng_after_update_exact": [w["rng_hash"] for w in wa] == [w["rng_hash"] for w in wb],
            "criterion": "exact input and model/optimizer/scheduler equality; sampler bitwise recovery not claimed"}


def costs(run: Path) -> dict:
    from datetime import datetime
    manifest = read_json(run / "manifest.json")
    phases = {}
    for path in (run / "raw/roles").glob("*.json"):
        role = read_json(path)
        key = role["phase"] + "/" + role["role"]
        part = phases.setdefault(key, {"calls": 0, "requests": 0, "tokens": 0, "worker_wall_seconds_sum": 0.0})
        part["calls"] += 1
        part["requests"] += len(role["requests"])
        part["tokens"] += sum(len(r["output_token_ids"]) for r in role["requests"])
        part["worker_wall_seconds_sum"] += role["wall_seconds"]
    updates = [read_json(p) for p in sorted((run / "raw").glob("metrics_step_*.json")) if "actor/grad_norm" in read_json(p)]
    seconds = sum(m.get("timing_s/step", 0) for m in updates)
    snapshots = [read_json(p) for p in sorted((run / "raw").glob("checkpoint_step_*.json"))]
    evaluations = [read_json(p) for p in sorted((run / "raw").glob("evaluation_step_*.json"))]
    finished = read_json(run / "status.json")["status"] in ("completed", "intentionally_stopped", "failed")
    wall = (run / "status.json").stat().st_mtime - datetime.fromisoformat(manifest["created_utc"]).timestamp() if finished else None
    return {"run_id": manifest["run_id"], "phase_role": phases, "completed_updates": len(updates),
            "recorded_run_wall_seconds": wall,
            "recorded_run_selected_gpu_hours": len(manifest["selected_gpus"]) * wall / 3600 if wall is not None else None,
            "native_step_wall_seconds": seconds,
            "checkpoint_wall_seconds": sum(x["wall_seconds"] for x in snapshots),
            "evaluation_wall_seconds": sum(x["wall_seconds"] for x in evaluations),
            "snapshot_bytes_written": sum(x["snapshot_bytes"] for x in snapshots),
            "native_step_allocated_gpu_hours": len(manifest["selected_gpus"]) * seconds / 3600,
            "limitations": "Recorded run wall spans original server manifest UTC to terminal status mtime, including native startup/restore/evaluation/IO, excluding earlier launcher preflight/model hashing. It is selected-GPU accounting, not utilization. Native step/checkpoint/evaluation components may overlap; role worker times overlap. Rebuild on server with original file timestamps."}


def freeze(calibration: Path, profile: Path, evidence_paths: dict) -> dict:
    if set(evidence_paths) != ACCEPTANCE_KEYS:
        raise ValueError("All six actual acceptance witnesses are required")
    candidate = read_json(calibration / "manifest.json")
    if read_json(calibration / "status.json")["status"] not in ("completed", "intentionally_stopped"):
        raise ValueError("Calibration must finish successfully before freezing")
    settings = read_json(profile)
    if any(candidate["overrides"].get(key) != value for key, value in settings["overrides"].items()):
        raise ValueError("Freeze profile differs from calibrated native parameters")
    if candidate["learning"].get("processor_options", {}) != settings.get("processor_options", {}):
        raise ValueError("Freeze image processor differs from calibrated inputs")
    sources = {name: file_digest(SRC / name) for name in ENGINE_FILES}
    if sources != candidate["learning"]["source_sha256"]:
        raise ValueError("Calibration engine differs from current source")
    witnesses = {}
    for key, path in evidence_paths.items():
        witness = read_json(path)
        if witness.get("passed") is not True:
            raise ValueError("Acceptance witness failed: " + key)
        if witness.get("learning_signature") != candidate["signatures"]["learning"]:
            raise ValueError("Acceptance witness learning identity differs: " + key)
        witnesses[key] = evidence(path)
    return {"acceptance": {key: True for key in sorted(ACCEPTANCE_KEYS)}, "evidence": witnesses,
            "environment_signature": candidate["learning"]["environment_signature"],
            "profile_sha256": file_digest(profile), "model_signature": candidate["model"]["signature"],
            "dataset_signature": candidate["data_signature"], "source_sha256": sources,
            "baseline_seeds": settings["baseline_seeds"],
            "health_rules": "calibration only; initial family accuracy .1–.9, answer/Verifier parse >=.9, truncation <=.05, nonzero groups >=.2, actual tools/natural repair/finite changes",
            "horizon": candidate["checkpoint_plan"]["full_horizon"],
            "sampling_rng_bitwise_recovery_verified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["health", "compare", "costs"])
    parser.add_argument("run", type=Path)
    parser.add_argument("--other", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "compare" and (args.other is None or args.step is None):
        parser.error("Comparison requires other run and step")
    result = compare(args.run, args.other, args.step) if args.operation == "compare" else globals()[args.operation](args.run)
    atomic_json(args.output, result, immutable=True)
    print(__import__("json").dumps(result, ensure_ascii=False))
