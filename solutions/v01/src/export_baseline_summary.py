"""Export completed D/E evidence as seed-level results and cost tables."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

from baseline_acceptance import costs
from baseline_costs import logical_role_counts
from pilot_utils import atomic_json, file_digest, read_json


def normal_results(reports: list[dict]) -> tuple[list[dict], dict]:
    if [report["seed"] for report in reports] != [20261011, 20261012, 20261013]:
        raise ValueError("Require all three predeclared seeds, once and in order")
    rows, deltas = [], []
    for report in reports:
        if report["complete_units"] != 32 or set(report["evaluations"]) != {"0", "8", "16", "24", "32"}:
            raise ValueError("Require all32 updates and all five evaluations")
        for step in (0, 8, 16, 24, 32):
            result = report["evaluations"][str(step)]
            if (result["cases"] != 256 or not 0 <= result["correct"] <= 256 or
                    result["accuracy"] != result["correct"] / 256):
                raise ValueError("Evaluation count and accuracy disagree")
            rows.append({"run_id": report["run_id"], "seed": report["seed"], "step": step,
                         **{key: result[key] for key in ("cases", "correct", "accuracy", "native_accuracy",
                                                       "explicit_answer_rate", "truncation_rate")}})
        deltas.append(100 * (report["evaluations"]["32"]["accuracy"] - report["evaluations"]["0"]["accuracy"]))
    return rows, {"seed_count": 3, "final_minus_initial_percentage_points": deltas,
                  "mean_delta_percentage_points": statistics.mean(deltas),
                  "range_delta_percentage_points": [min(deltas), max(deltas)],
                  "unit": "Independent training seed; repeated cases/trajectories are not additional seeds",
                  "inference": "Descriptive three-seed result; no significance or generalization claim"}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def export(sequence: Path) -> dict:
    sequence = sequence.resolve()
    manifest = read_json(sequence / "manifest.json")
    summary_path = sequence / "processed/summary.json"
    summary = read_json(summary_path)
    if (read_json(sequence / "status.json")["status"] != "completed" or
            summary["D_complete"] is not True or summary["E_passed"] is not True):
        raise ValueError("Do not export unfinished D/E as a completed baseline")
    reports = summary["normal_runs"]
    if [report["run_id"] for report in reports] != manifest["normal_run_ids"]:
        raise ValueError("Sequence and normal report identities disagree")
    rows, result = normal_results(reports)
    paired = summary["E"]
    if (paired["passed"] is not True or paired["full_GPU_update_completed"] is not True or
            paired["finite_updated_states"] is not True or paired["clean"] != manifest["normal_run_ids"][0] or
            paired["changed"] != manifest["E_run_id"]):
        raise ValueError("Require the declared actual finite paired GPU update")
    changed = sequence.parent / manifest["E_run_id"]
    entries = [(sequence.parent / report["run_id"], report["costs"], report["logical_rollout_costs"])
               for report in reports]
    entries.append((changed, costs(changed), logical_role_counts(changed)))
    run_rows, role_rows, inputs = [], [], []
    for run, cost, logical in entries:
        with (run / "raw/physical_gpu.csv").open(encoding="utf-8", newline="") as stream:
            samples = list(csv.DictReader(stream))
        selected = read_json(run / "manifest.json")["selected_gpus"]
        if {int(sample["gpu"]) for sample in samples} != set(selected):
            raise ValueError("Physical GPU samples do not cover the selected devices")
        run_rows.append({"run_id": run.name, **{key: cost[key] for key in (
            "completed_updates", "recorded_run_wall_seconds", "recorded_run_selected_gpu_hours",
            "native_step_wall_seconds", "checkpoint_wall_seconds", "evaluation_wall_seconds", "snapshot_bytes_written")},
            "physical_peak_sampled_mib": max(int(sample["device_used_mib"]) for sample in samples)})
        if set(cost["phase_role"]) != set(logical["phase_role"]):
            raise ValueError("Logical and worker role ledgers cover different phases")
        for role, worker in sorted(cost["phase_role"].items()):
            role_rows.append({"run_id": run.name, "phase_role": role,
                              **{"worker_" + key: value for key, value in worker.items()}, **logical["phase_role"][role]})
        inputs.append({"run_id": run.name, "physical_gpu_csv_sha256": file_digest(run / "raw/physical_gpu.csv")})
    output = sequence / "processed"
    write_csv(output / "results.csv", rows)
    write_csv(output / "costs.csv", run_rows)
    write_csv(output / "role_costs.csv", role_rows)
    write_csv(output / "paired_advantages.csv", paired["rows"])
    result.update({"D_complete": True, "E_passed": True, "six_gates_passed": False,
                   "summary_sha256": file_digest(summary_path), "source_sha256": file_digest(Path(__file__)),
                   "input_evidence": inputs, "costs": run_rows,
                   "limits": "Component/worker times overlap; GPU-hours use selected-device wall allocation. Physical peaks are2-second sampling lower bounds and include device-wide usage. Rebuild costs on server using original status timestamps."})
    atomic_json(output / "result_summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    args = parser.parse_args()
    print(__import__("json").dumps(export(args.sequence), ensure_ascii=False))
