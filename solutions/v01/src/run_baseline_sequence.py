"""Execute the predeclared three normal seeds and E serially on four GPUs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from analyze_baseline import paired_audit, summarize
from pilot_utils import ROOT, SRC, VERSION, atomic_json, file_digest, read_json
from preflight import inspect
from run_baseline import ACCEPTANCE_KEYS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=SRC / "configs/baseline_3b_candidate.json")
    parser.add_argument("--gpus", default="4,5,6,7")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,45}", args.sequence_id):
        parser.error("Use a new plain sequence ID up to45 characters")
    frozen = read_json(args.freeze)
    if (set(frozen["acceptance"]) != ACCEPTANCE_KEYS or not all(value is True for value in frozen["acceptance"].values()) or
            frozen["baseline_seeds"] != [20261011, 20261012, 20261013] or frozen["horizon"] != 32):
        raise ValueError("Require all six witnesses and the predeclared three full32 seeds")
    ids = [int(x) for x in args.gpus.split(",")]
    if len(ids) != 4 or len(set(ids)) != 4:
        parser.error("The frozen profile requires four distinct GPUs")
    sequence = VERSION / "outputs" / args.sequence_id
    (sequence / "raw").mkdir(parents=True, exist_ok=False)
    (sequence / "processed").mkdir()
    names = [f"{args.sequence_id}_seed_{seed}" for seed in frozen["baseline_seeds"]]
    corruption = args.sequence_id + "_E_delta1"
    atomic_json(sequence / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "normal_run_ids": names, "E_run_id": corruption, "seeds": frozen["baseline_seeds"],
        "freeze_path": str(args.freeze.resolve()), "freeze_sha256": file_digest(args.freeze),
        "entry_sha256": file_digest(Path(__file__)), "analysis_sha256": file_digest(SRC / "analyze_baseline.py"),
        "D_steps": 32, "E_step": 8, "E_clean_control_reused": names[0], "gpus": ids}, immutable=True)
    atomic_json(sequence / "status.json", {"status": "running", "normal_runs_completed": []})
    base = [sys.executable, str(SRC / "run_baseline.py"), "--model-path", str(args.model_path.resolve()),
            "--data", str(args.data.resolve()), "--profile", str(args.profile.resolve()), "--gpus", args.gpus]
    completed = []

    def execute(name, extra):
        # Wait without allocating GPUs; never signal other workloads.
        while True:
            report = inspect(ROOT, ids, minimum_disk_gib=0)
            if not report["problems"]:
                break
            if any(not problem.startswith("GPU ") for problem in report["problems"]):
                raise RuntimeError("Preflight failed: " + str(report["problems"]))
            with (sequence / "raw/resource_wait.jsonl").open("a", encoding="utf-8") as stream:
                __import__('json').dump({"utc": datetime.now(timezone.utc).isoformat(), "run_id": name,
                                        "problems": report["problems"]}, stream); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            time.sleep(60)
        command = [*base, "--run-id", name, *extra]
        run = VERSION / "outputs" / name
        if run.exists():
            raise FileExistsError("Run IDs are immutable; do not rerun an existing attempt")
        with (sequence / f"raw/{name}_launcher.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(sequence / f"raw/{name}_launch.json", {"argv": command, "pid": process.pid,
                "cwd": str(ROOT), "utc": datetime.now(timezone.utc).isoformat()}, immutable=True)
            print("START", name, "PID", process.pid, flush=True)
            # A restored parent is content-verified before the child creates raw/.
            # Start the sampler after raw/ exists, preserving its original PID check.
            while not (run / "raw").is_dir() and process.poll() is None:
                time.sleep(1)
            monitor = None
            monitor_log = sequence / f"raw/{name}_gpu_monitor.log"
            with monitor_log.open("x", encoding="utf-8") as sampling:
                if process.poll() is None:
                    monitor = subprocess.Popen([sys.executable, str(SRC / "record_gpu.py"), str(run),
                        "--pid", str(process.pid), "--gpus", args.gpus], cwd=ROOT, stdout=sampling, stderr=subprocess.STDOUT)
                    atomic_json(sequence / f"raw/{name}_monitor_launch.json", {"pid": monitor.pid,
                        "baseline_pid": process.pid, "gpus": ids}, immutable=True)
                result = process.wait()
                if monitor is not None:
                    monitor_result = monitor.wait()
                    if monitor_result != 0:
                        raise RuntimeError("Own GPU monitor failed: " + name)
            if result != 0:
                raise RuntimeError("Native baseline attempt failed; keep its complete failure scene: " + name)
        print("END", name, flush=True)
        return run

    try:
        reports = []
        for i, (seed, name) in enumerate(zip(frozen["baseline_seeds"], names)):
            extra = ["--phase", "baseline", "--freeze", str(args.freeze.resolve()), "--seed", str(seed), "--pin-step", "7"]
            if i == 0:
                extra += ["--pin-step", "8"]
            run = execute(name, extra)
            report = summarize(run)
            atomic_json(run / "processed/normal_baseline_report.json", report, immutable=True)
            reports.append(report); completed.append(name)
            atomic_json(sequence / "status.json", {"status": "running", "normal_runs_completed": completed})
        first = VERSION / "outputs" / names[0]
        changed = execute(corruption, ["--phase", "consistency", "--seed", str(frozen["baseline_seeds"][0]),
            "--resume-from", str(first / "checkpoints/global_step_7"), "--replay-from", str(first),
            "--replay-start", "8", "--replay-end", "8", "--injection-step", "8", "--reward-delta", "1",
            "--stop-after", "8", "--pin-step", "8"])
        pair = paired_audit(first, changed, 8)
        atomic_json(changed / "processed/paired_reward_audit.json", pair, immutable=True)
        atomic_json(sequence / "processed/summary.json", {"normal_runs": reports, "E": pair,
                    "D_complete": True, "E_passed": pair["passed"], "six_gates_passed": False}, immutable=True)
        atomic_json(sequence / "status.json", {"status": "completed" if pair["passed"] else "D_completed_E_no_eligible_target",
                                              "normal_runs_completed": completed, "E_passed": pair["passed"]})
        return 0
    except Exception as error:
        atomic_json(sequence / "status.json", {"status": "failed", "normal_runs_completed": completed,
                                              "error": type(error).__name__ + ": " + str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
