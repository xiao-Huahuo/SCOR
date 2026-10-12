"""Account for all Gate1 attempts, including controls, failures and reused source work."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from baseline_acceptance import costs
from baseline_costs import logical_role_counts
from export_gate1 import write_csv
from gate1_evidence import evidence, physical
from pilot_utils import VERSION, atomic_json, file_digest, read_json


def run_cost(run: Path, category: str) -> tuple[dict, list[dict], list[dict]]:
    manifest = read_json(run / "manifest.json")
    status = read_json(run / "status.json")
    if status["status"] in {"running", "launch_planned"}:
        raise ValueError("Research cost cannot omit a still-active attempt")
    measured = costs(run)
    if status["status"] == "interrupted":
        interrupted = read_json(run / "raw/workflow_interruption.json")
        end = datetime.fromisoformat(interrupted["utc"]).timestamp()
        wall = end - datetime.fromisoformat(manifest["created_utc"]).timestamp()
        if wall < 0:
            raise ValueError("Interruption precedes its actual launch")
        measured["recorded_run_wall_seconds"] = wall
        measured["recorded_run_selected_gpu_hours"] = len(manifest["selected_gpus"]) * wall / 3600
    logical_error = None
    try:
        logical = logical_role_counts(run)["phase_role"]
        if set(logical) != set(measured["phase_role"]):
            raise ValueError("Logical and worker role ledgers cover different phases")
    except ValueError as error:
        if status["status"] not in {"failed", "interrupted"}:
            raise
        logical, logical_error = {}, str(error)
    roles = [{"run_id": run.name, "category": category, "phase_role": key,
              **{"worker_" + k: v for k, v in value.items()},
              "logical_counts_certified": key in logical,
              **{k: logical.get(key, {}).get(k) for k in [
                  "logical_batch_calls", "logical_requests", "generated_tokens"]}}
             for key, value in sorted(measured["phase_role"].items())]
    row = {"run_id": run.name, "category": category, "phase": manifest["phase"],
           "branch": manifest.get("gate1", {}).get("branch", "historical_source"),
           "seed": manifest["seed"], "status": status["status"],
           **{key: measured[key] for key in ["completed_updates", "recorded_run_wall_seconds",
              "recorded_run_selected_gpu_hours", "native_step_wall_seconds", "checkpoint_wall_seconds",
              "evaluation_wall_seconds", "snapshot_bytes_written"]},
           "logical_counts_certified": logical_error is None, "incomplete_logical_ledger": logical_error}
    sources = [evidence(p) for p in [run / "manifest.json", run / "status.json", *sorted(
        p for p in (run / "raw").rglob("*") if p.is_file() and p.suffix in {".json", ".csv"})]]
    return row, roles, sources


def export(sequence: Path) -> dict:
    sequence = physical(sequence, VERSION / "outputs")
    if read_json(sequence / "status.json")["status"] != "completed":
        raise ValueError("Require the completed five-seed sequence")
    rows, roles, inputs = [], [], []
    for run in sorted((VERSION / "outputs").iterdir()):
        path = run / "manifest.json"
        if not path.is_file() or "gate1" not in read_json(path):
            continue
        if read_json(run / "status.json")["status"] == "preflight_passed_no_training":
            continue
        manifest = read_json(path)
        category = "engineering" if (manifest["phase"] == "control" or
                                     manifest["gate1"]["branch"] == "control") else "new_research"
        row, part, sources = run_cost(physical(run), category)
        rows.append(row); roles.extend(part); inputs.extend(sources)
    manifest = read_json(sequence / "manifest.json")
    assets = read_json(Path(manifest["inputs"]["assets"]))
    for entry in assets["histories"].values():
        # Original server timestamps belong to the source, not its imported hardlinks.
        original = physical(Path(entry["source"]))
        row, part, sources = run_cost(original, "reused_v01_source_not_new_spending")
        rows.append(row); roles.extend(part); inputs.extend(sources)
    output = sequence / "processed"
    write_csv(output / "costs.csv", rows)
    if roles:
        write_csv(output / "role_costs.csv", roles)
    totals = {category: sum(row["recorded_run_selected_gpu_hours"] for row in rows
                           if row["category"] == category and row["recorded_run_selected_gpu_hours"] is not None)
              for category in sorted({row["category"] for row in rows})}
    receipt = {"allocated_gpu_hours_by_category": totals, "runs": len(rows), "raw_evidence": inputs,
               "source_sha256": file_digest(Path(__file__)),
               "tables_sha256": {name: file_digest(output / name) for name in ["costs.csv", "role_costs.csv"]
                                 if (output / name).exists()},
               "limitations": "Selected GPUs times recorded wall allocation, including restore/evaluation/IO; component and worker times overlap. Interrupted ledgers may be lower bounds. Reused v01 costs are reported separately, not charged as new Gate1 spending."}
    atomic_json(output / "cost_manifest.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    print(export(parser.parse_args().sequence))
