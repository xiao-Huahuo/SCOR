"""Recompute a reward counterfactual on a captured native training batch."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from pilot_utils import ROOT, atomic_json, file_digest, read_json


def compare(run: Path, step: int, row: int, delta: float) -> dict:
    sys.path.insert(0, str(ROOT / ".runtime/upstream/Agent0/Agent0-VL"))
    import torch
    from scor_hooks import apply_corruption
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    path = run / f"raw/training_group_step_{step:06d}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    manifest = read_json(run / "manifest.json")
    receipt = read_json(run / f"raw/reward_step_{step:06d}.json")
    if receipt["corruption_applied"]:
        raise ValueError("Use an unmodified native batch as the counterfactual source")
    rewards = data.batch["token_level_rewards"]
    mask = data.batch["response_mask"]
    active = data.batch.get("multiturn_mask", mask).bool()
    groups = data.non_tensor_batch["uid"]
    clean, _ = compute_grpo_outcome_advantage(rewards.clone(), mask, groups)
    if not torch.allclose(clean, data.batch["advantages"], atol=1e-6, rtol=1e-5):
        raise ValueError("Native recomputation differs from the actual updater input")
    modified = apply_corruption(rewards, data.batch["attention_mask"], row, delta)
    corrupt, _ = compute_grpo_outcome_advantage(modified, mask, groups)
    if not torch.isfinite(corrupt).all():
        raise ValueError("Non-finite counterfactual advantage")
    rows = []
    for i in range(len(data)):
        positions = active[i].nonzero().flatten()
        if not positions.numel():
            raise ValueError("Captured trajectory has no trainable tokens")
        changed = int(((clean[i] - corrupt[i]).abs()[active[i]] > 1e-6).sum())
        flipped = int(((clean[i] * corrupt[i] < 0) & active[i]).sum())
        rows.append({"row": i, "same_group_as_modified": bool(groups[i] == groups[row]),
                     "original_return": float(rewards[i].sum()),
                     "modified_return": float(modified[i].sum()),
                     "original_advantage": float(clean[i, positions[0]]),
                     "modified_advantage": float(corrupt[i, positions[0]]),
                     "changed_trainable_tokens": changed, "sign_flipped_tokens": flipped})
    if any(r["changed_trainable_tokens"] for r in rows if not r["same_group_as_modified"]):
        raise ValueError("Reward modification unexpectedly changed another prompt group")
    report = {"run_id": manifest["run_id"], "step": step, "modified_row": row, "delta": delta,
              "source_batch_sha256": file_digest(path),
              "training_signature": manifest["training_signature"],
              "analysis_source_sha256": file_digest(Path(__file__)),
              "native_advantage_source_sha256": file_digest(
                  ROOT / ".runtime/upstream/Agent0/Agent0-VL/verl/trainer/ppo/core_algos.py"),
              "native_recomputation_matches_captured_advantages": True,
              "changed_trajectories": sum(r["changed_trainable_tokens"] > 0 for r in rows),
              "sign_flipped_trajectories": sum(r["sign_flipped_tokens"] > 0 for r in rows),
              "rows": rows, "gpu_counterfactual_update_executed": False,
              "six_gates_passed": False,
              "limitations": ["Offline reward/advantage check only; no new gradient or policy evaluation",
                              "Artificial return perturbation, not an observed Verifier mistake"]}
    destination = run / "processed"
    atomic_json(destination / f"reward_counterfactual_step_{step:06d}.json", report)
    with (destination / f"reward_counterfactual_step_{step:06d}.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--delta", type=float, default=-2.0)
    args = parser.parse_args()
    print(json.dumps(compare(args.run.resolve(), args.step, args.row, args.delta), indent=2))
