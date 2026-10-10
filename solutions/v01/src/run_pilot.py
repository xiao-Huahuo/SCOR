"""Launch a bounded native Agent0-VL pilot with immutable evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from apply_instrumentation import apply
from pilot_utils import ROOT, SRC, UPSTREAM_COMMIT, VERSION, atomic_json, check_profile, digest, file_digest, hydra_value, read_json
from preflight import inspect


def training_signature(overrides: dict, model: Path, dataset: dict) -> str:
    # Output paths, logging and checkpoint cadence do not change the update rule.
    ignored = {"trainer.default_local_dir", "trainer.resume_mode", "trainer.resume_from_path",
               "trainer.save_freq", "trainer.val_only", "trainer.experiment_name"}
    substantive = {key: value for key, value in overrides.items() if key not in ignored}
    return digest({"overrides": substantive, "upstream": UPSTREAM_COMMIT,
                   "hook_sha256": file_digest(SRC / "scor_hooks.py"),
                   "native_score_sha256": file_digest(ROOT / ".runtime/upstream/Agent0/Agent0-VL/verl/utils/reward_score/math_verify.py"),
                   "rollout_source_sha256": file_digest(ROOT / ".runtime/upstream/Agent0/Agent0-VL/verl/workers/rollout/vllm_rollout/vllm_agent0_rollout_spmd.py"),
                   "model_config": file_digest(model / "config.json"),
                   "model_revision": model.name, "data_signature": dataset["data_signature"]})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--profile", type=Path, default=SRC / "pilot_4090.json")
    parser.add_argument("--data", type=Path, default=VERSION / "inputs/visual_pilot_v2")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--no-new-checkpoint", action="store_true",
                        help="Bounded diagnostic run without a new checkpoint; final model is not persisted.")
    parser.add_argument("--corrupt-step", type=int)
    parser.add_argument("--corrupt-row", type=int, default=0)
    parser.add_argument("--corrupt-delta", type=float, default=-2.0)
    parser.add_argument("--replay-generation-from", type=Path)
    parser.add_argument("--replay-step", type=int, default=1)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", args.run_id):
        parser.error("Use a plain run identifier, not a path")
    if args.replay_generation_from and (args.replay_step != 1 or args.resume_from):
        parser.error("Paired replay in this pilot requires step 1 from the same fresh initial model")
    profile = read_json(args.profile)
    check_profile(profile)
    ids = [int(i) for i in args.gpus.split(",")]
    if len(ids) != profile["overrides"]["trainer.n_gpus_per_node"]:
        parser.error("GPU count must match the profile")
    run = VERSION / "outputs" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    for name in ["raw", "checkpoints", "processed"]:
        (run / name).mkdir()
    report = inspect(ROOT, ids, minimum_disk_gib=5 if args.no_new_checkpoint else 55)
    atomic_json(run / "raw/preflight.json", report, immutable=True)
    if report["problems"]:
        atomic_json(run / "status.json", {"status": "preflight_failed", "problems": report["problems"]})
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    model, data = args.model_path.resolve(), args.data.resolve()
    if not model.is_dir() or not (model / "config.json").exists():
        raise FileNotFoundError("Pass a complete local model snapshot")
    dataset = read_json(data / "manifest.json")
    for split in dataset["splits"].values():
        if file_digest(data / split["file"]) != split["sha256"]:
            raise ValueError("Input parquet hash mismatch")
    overrides = dict(profile["overrides"])
    overrides.update({"actor_rollout_ref.model.path": str(model),
                      "data.train_files": str(data / "train.parquet"),
                      "data.val_files": str(data / "val.parquet"),
                      "trainer.default_local_dir": str(run / "checkpoints"),
                      "trainer.project_name": "SCOR-v01",
                      "trainer.experiment_name": args.run_id})
    if args.resume_from:
        overrides["trainer.resume_mode"] = "resume_path"
        overrides["trainer.resume_from_path"] = str(args.resume_from.resolve())
    if args.no_new_checkpoint:
        overrides["trainer.save_freq"] = -1
    checkout = ROOT / ".runtime/upstream/Agent0"
    patch_hash = apply(checkout)
    signature = training_signature(overrides, model, dataset)
    checkpoint_count = 0 if args.no_new_checkpoint else (
        args.stop_after or overrides["trainer.total_training_steps"])
    # Conservatively estimate fp32 weights and two Adam moments from BF16 weight size.
    model_bytes = sum(path.stat().st_size for path in model.glob("*.safetensors"))
    needed = checkpoint_count * model_bytes * 6 + 5 * 2**30
    if shutil.disk_usage(ROOT).free < needed:
        problem = {"status": "storage_budget_failed", "needed_gib": round(needed / 2**30, 1),
                   "free_gib": report["free_disk_gib"], "checkpoint_count": checkpoint_count}
        atomic_json(run / "status.json", problem)
        print(json.dumps(problem))
        return 2
    manifest = {"run_id": args.run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": profile["purpose"], "upstream_commit": UPSTREAM_COMMIT,
                "instrumented_driver_sha256": patch_hash, "seed": profile["seed"],
                "dependency_notes": ["math-verify 0.7.0 matches the original native scoring API; no scoring source adaptation", "xgrammar 0.1.16 compiled from its official source tag because the exact PyPI release is unavailable"],
                "memory_adaptation": "Release inactive PyTorch allocator blocks immediately before native vLLM wake_up; parameters and update rule unchanged",
                "transport_fix": "Keep native rollout padded buffers and verification metadata at common configured lengths across DP ranks; generated tokens and validity masks unchanged",
                "inference_adaptation": "Disable V0 multimodal prefix caching and limit concurrent sequences to 16; full visual inputs retained",
                "project_source_sha256": {p.name: file_digest(p) for p in sorted(SRC.glob("*.py"))},
                "training_signature": signature, "overrides": overrides,
                "config_hash": digest(overrides), "data_signature": dataset["data_signature"],
                "model_snapshot": str(model), "selected_gpus": ids,
                "stop_after": args.stop_after, "final_checkpoint_disabled": args.no_new_checkpoint,
                "corruption": {"step": args.corrupt_step, "row": args.corrupt_row, "delta": args.corrupt_delta},
                "replay_generation": ({"run_dir": str(args.replay_generation_from.resolve()),
                                       "step": args.replay_step} if args.replay_generation_from else None),
                "limitations": ["Small synthetic dataset and 3B adaptation; not paper benchmark reproduction",
                                "vLLM sampling RNG bitwise recovery is not yet verified"]}
    if args.no_new_checkpoint:
        manifest["limitations"].append("No final model checkpoint is saved in this diagnostic attempt")
    if args.replay_generation_from:
        source = read_json(args.replay_generation_from / "manifest.json")
        if source["training_signature"] != signature:
            raise ValueError("Replayed generation belongs to a different training configuration")
    atomic_json(run / "manifest.json", manifest, immutable=True)
    environment_receipt = {"python": sys.version, "packages": subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True).splitlines()}
    xgrammar_source = ROOT / ".runtime/upstream/xgrammar"
    if (xgrammar_source / ".git").exists():
        environment_receipt["xgrammar_source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=xgrammar_source, text=True).strip()
    atomic_json(run / "raw/environment.json", environment_receipt, immutable=True)
    atomic_json(run / "status.json", {"status": "running"})
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": args.gpus, "SCOR_RUN_DIR": str(run),
                        "PYTHONPATH": str(SRC) + os.pathsep + str(checkout / "Agent0-VL"),
                        "RAY_ADDRESS": "local", "RAY_TMPDIR": str(ROOT / ".runtime/r"),
                        "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": str(2 * 2**30),
                        "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false",
                        "VLLM_USE_V1": "0",
                        "VERL_PPO_LOGGING_LEVEL": "INFO",
                        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                        "HF_HOME": str(ROOT / ".runtime/cache/huggingface"),
                        "TORCH_HOME": str(ROOT / ".runtime/cache/torch"),
                        "XDG_CACHE_HOME": str(ROOT / ".runtime/cache"),
                        "TRITON_CACHE_DIR": str(ROOT / ".runtime/cache/triton"),
                        "TORCHINDUCTOR_CACHE_DIR": str(ROOT / ".runtime/cache/torchinductor"),
                        "CUDA_CACHE_PATH": str(ROOT / ".runtime/cache/cuda"),
                        "NUMBA_CACHE_DIR": str(ROOT / ".runtime/cache/numba"),
                        "TMPDIR": str(ROOT / ".runtime/tmp"), "OMP_NUM_THREADS": "4"})
    command = [sys.executable, "-m", "verl.trainer.main_ppo", "--config-name=agent0_trainer"]
    command.extend(key + "=" + hydra_value(value) for key, value in overrides.items())
    command.extend(["hydra.run.dir=" + hydra_value(str(run / "raw/hydra")),
                    "hydra.output_subdir=null", "hydra.job.chdir=false"])
    atomic_json(run / "raw/command.json", {"argv": command}, immutable=True)
    try:
        with (run / "raw/console.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=checkout / "Agent0-VL", env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        stopped = (run / "raw/intentional_stop.json").exists()
        status = "intentionally_stopped" if stopped else ("completed" if result.returncode == 0 else "failed")
        atomic_json(run / "status.json", {"status": status, "returncode": result.returncode,
                                          "final_checkpoint_disabled": args.no_new_checkpoint})
        print(json.dumps({"run_id": args.run_id, "status": status, "returncode": result.returncode}))
        return 0 if stopped else result.returncode
    except BaseException as error:
        atomic_json(run / "status.json", {"status": "interrupted", "error_type": type(error).__name__})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
