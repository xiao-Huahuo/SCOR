"""Report actual native-chain evidence without promoting a pilot to a Gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from pilot_utils import atomic_json, read_json


def verify(run: Path) -> dict:
    manifest = read_json(run / "manifest.json")
    status = read_json(run / "status.json")
    rows, problems = [], []
    for path in sorted((run / "raw").glob("group_step_*.json")):
        group = read_json(path)
        step = group["step"]
        metrics_path = run / f"raw/metrics_step_{step:06d}.json"
        reward_path = run / f"raw/reward_step_{step:06d}.json"
        if not metrics_path.exists() or not reward_path.exists():
            problems.append(f"Step {step}: missing metrics or reward receipt")
            continue
        metrics, reward = read_json(metrics_path), read_json(reward_path)
        steps = [item for trajectory in reward["steps"] for item in trajectory]
        row = {"step": step, "trajectories": group["trajectories"],
               "nonzero_advantage_tokens": group["nonzero_advantage_tokens"],
               "parameter_sample_changed_fraction": metrics.get("scor/parameter_sample_changed", 0),
               "parameter_sample_finite": metrics.get("scor/parameter_sample_finite", 0),
               "gradient_norm": metrics.get("actor/grad_norm"),
               "tool_success_steps": sum(bool(item.get("tool_success")) for item in steps),
               "verified_steps": sum(bool(item.get("verified")) for item in steps),
               "repaired_steps": sum(bool(item.get("was_repaired")) for item in steps),
               "corruption_applied": reward["corruption_applied"]}
        rows.append(row)
        if not group["old_log_probs_present"] or not group["nonzero_advantage_tokens"]:
            problems.append(f"Step {step}: no usable GRPO advantage or behavior logprob")
        if not row["parameter_sample_changed_fraction"] or row["parameter_sample_finite"] != 1:
            problems.append(f"Step {step}: finite parameter update not demonstrated")
        if row["gradient_norm"] is None or not math.isfinite(row["gradient_norm"]):
            problems.append(f"Step {step}: finite gradient not demonstrated")
        if not (run / f"raw/completed_step_{step:06d}.json").exists():
            problems.append(f"Step {step}: no completion receipt")
    if not rows:
        problems.append("No completed training updates")
    if sum(row["tool_success_steps"] for row in rows) == 0:
        problems.append("No successful real tool execution observed")
    if sum(row["verified_steps"] for row in rows) == 0:
        problems.append("No verified Solver step observed")
    if sum(row["repaired_steps"] for row in rows) == 0:
        problems.append("Repair was enabled but no actual repaired step was observed")
    checkpoints = [str(p.parent) for p in (run / "checkpoints").glob("*/scor_complete.json")]
    resume_path = run / "raw/resume_loaded.json"
    resume = read_json(resume_path) if resume_path.exists() else None
    validation = [read_json(path) for path in sorted((run / "raw").glob("validation_step_*.json"))]
    report = {"run_id": manifest["run_id"], "native_process_status": status,
              "completed_updates": len(rows), "update_rows": rows,
              "complete_checkpoint_paths": checkpoints, "resume_evidence": resume,
              "held_out_evaluations": validation,
              "native_chain_observation_passed": not problems,
              "problems": problems, "limitations": manifest["limitations"],
              "paper_reproduction_complete": False, "six_gates_passed": False}
    destination = run / "processed"
    destination.mkdir(exist_ok=True)
    atomic_json(destination / "verification.json", report)
    if rows:
        with (destination / "updates.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = verify(args.run.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
