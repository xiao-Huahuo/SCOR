"""Run Gate1 with an explicit legacy-state bridge and fixed-input audits."""
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

from apply_baseline_instrumentation import prepare as prepare_checkout
from checkpoint_store import plan, validate
from pilot_utils import ROOT, SRC, VERSION, UPSTREAM_COMMIT, atomic_json, digest, file_digest, hydra_value, read_json
from preflight import inspect
from prepare_baseline_data import verify as verify_data
from records import model_identity, signatures
from gate1_contract import compatible_learning

ENGINE_FILES = ["run_baseline.py", "native_adapter.py", "records.py", "checkpoint_store.py", "replay.py",
                "interventions.py", "evaluate.py", "probe.py", "baseline_acceptance.py", "apply_baseline_instrumentation.py", "apply_instrumentation.py", "pilot_utils.py",
                "run_gate1.py", "gate1_contract.py", "gate1_probes.py"]
ACCEPTANCE_KEYS = {"source_asset_import", "actual_parent_restore", "fixed_unit_replay",
                   "actual_interruption_recovery", "probe_inputs", "threshold_calibration",
                   "storage_budget", "independent_code_review"}


def check_profile(profile: dict, ids: list[int]) -> None:
    values = profile["overrides"]
    if profile["model_type"] not in ("qwen2_vl", "qwen2_5_vl"):
        raise ValueError("Use one of the pinned native Qwen-VL architectures")
    pixels = profile.get("processor_options", {})
    if pixels and (set(pixels) != {"min_pixels", "max_pixels"} or
                   any(type(v) is not int for v in pixels.values()) or not 0 < pixels["min_pixels"] <= pixels["max_pixels"]):
        raise ValueError("Use an explicit positive native image-processor pixel budget")
    required = {"algorithm.adv_estimator": "grpo", "trainer.balance_batch": False,
                "actor_rollout_ref.rollout.rollout_type": "agent0", "reward_model.reward_manager": "agent0",
                "actor_rollout_ref.rollout.enable_tool_execution": True,
                "actor_rollout_ref.rollout.enable_verification": True,
                "actor_rollout_ref.rollout.enable_self_repair": True,
                "actor_rollout_ref.actor.ppo_epochs": 1, "reward_model.alpha_out": 1.0}
    if len(ids) != len(set(ids)) or not 1 <= len(ids) <= 4 or len(ids) != values["trainer.n_gpus_per_node"]:
        raise ValueError("Select at most four distinct GPUs matching the profile")
    if any(values.get(k) != v for k, v in required.items()):
        raise ValueError("Baseline requires the declared complete native learning loop")
    n, batch, mini = (values[k] for k in ("actor_rollout_ref.rollout.n", "data.train_batch_size",
                                        "actor_rollout_ref.actor.ppo_mini_batch_size"))
    if not all(type(x) is int and x > 0 for x in (n, batch, mini)) or n < 2 or mini != batch:
        raise ValueError("Keep complete multi-response groups in one global optimizer minibatch")
    if (batch * n) % len(ids) or len(ids) % values["actor_rollout_ref.rollout.tensor_model_parallel_size"]:
        raise ValueError("Batch or tensor parallel topology is not divisible")
    if values.get("trainer.max_actor_ckpt_to_keep") is not None or values.get("trainer.del_local_ckpt_after_load"):
        raise ValueError("Native pre-save deletion and destructive loading are disabled")


def storage_peak(schedule: dict, full_bytes: int, initial_bytes: int) -> int:
    live, peak = [], 0
    for step in schedule["scheduled_steps"]:
        live.append(step)
        peak = max(peak, sum(initial_bytes if s == 0 else full_bytes for s in live))
        rolling = [s for s in live if s not in schedule["protected"]]
        live = [s for s in live if s in schedule["protected"] or s in rolling[-schedule["keep"]:]]
    return peak


def recovery_contract(previous: Path, snapshot: Path, *, seed: int, phase: str,
                      branch: str, probe: Path, assets: Path, freeze: Path | None) -> dict:
    """An explicit new attempt continues its predecessor's last committed state."""
    previous = previous.absolute()
    snapshot = snapshot.absolute()
    owned = (VERSION / "outputs").absolute()
    if (not previous.is_relative_to(owned) or previous.parent != owned or
            any(p.is_symlink() for p in [previous, snapshot, *snapshot.parents] if p.is_relative_to(owned)) or
            snapshot.parent.parent != previous):
        raise ValueError("Recovery must use a physical snapshot owned by its previous attempt")
    before = read_json(previous / "manifest.json")
    status = read_json(previous / "status.json")
    if before["run_id"] != previous.name or status.get("status") not in {"failed", "interrupted", "intentionally_stopped"}:
        raise ValueError("Previous attempt is active or already complete")
    launch = read_json(previous / "raw/workflow_launch.json")
    if launch["run_id"] != previous.name:
        raise ValueError("Recovery launch ownership differs")
    try:
        os.kill(launch["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("Previous launcher process is still alive")
    cert = validate(snapshot)
    committed = []
    for candidate in previous.glob("checkpoints/global_step_*"):
        try:
            item = validate(candidate, before["signatures"]["learning"], check_hashes=False)
            done = read_json(previous / f"raw/completed_step_{item['step']:06d}.json") if item["step"] else None
            if item["run_id"] == before["run_id"] and (not item["step"] or
                    (done["model_snapshot_committed"] is True and done["learning_signature"] == item["learning_signature"])):
                committed.append(item["step"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if (not committed or cert["step"] != max(committed) or cert["run_id"] != before["run_id"] or
            cert["learning_signature"] != before["signatures"]["learning"] or before["seed"] != seed or
            before["phase"] != phase or before["gate1"]["branch"] != branch or
            before["gate1"]["probe_manifest_sha256"] != file_digest(probe / "manifest.json") or
            before["gate1"]["import_manifest_sha256"] != file_digest(assets) or
            before["gate1"]["freeze_sha256"] != (file_digest(freeze) if freeze else None)):
        raise ValueError("Recovery changed its last committed state/seed/branch/probe/import/freeze")
    return {"previous_attempt": str(previous), "previous_manifest_sha256": file_digest(previous / "manifest.json"),
            "checkpoint": str(snapshot), "state_signature": cert["state_signature"],
            "completed_through": cert["step"], "origin_injection_step": before["gate1"].get(
                "recovery", {}).get("origin_injection_step", before.get("injection_step")),
            "origin_reward_delta": before["gate1"].get("recovery", {}).get(
                "origin_reward_delta", before.get("reward_delta"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=VERSION / "inputs/visual_baseline_08")
    parser.add_argument("--profile", type=Path, default=SRC / "configs/baseline_3b_candidate.json")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--phase", choices=["preparation", "control", "formal"], default="control")
    parser.add_argument("--evaluation-split", choices=["calibration", "evaluation"])
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--recovery-from", type=Path)
    parser.add_argument("--pin-step", type=int, action="append", default=[])
    parser.add_argument("--replay-from", type=Path)
    parser.add_argument("--replay-start", type=int, default=1)
    parser.add_argument("--replay-end", type=int)
    parser.add_argument("--injection-step", type=int)
    parser.add_argument("--reward-delta", type=int, choices=[0, 1], default=1)
    parser.add_argument("--repeat-initial-evaluation", action="store_true")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--assets", type=Path, default=VERSION / "inputs/import_manifest.json")
    parser.add_argument("--probe-set", type=Path, required=True)
    parser.add_argument("--branch", choices=["source", "prefix", "clean", "corrupt", "repeat", "control"], required=True)
    args = parser.parse_args()
    if args.evaluation_split is not None and args.phase != "preparation":
        parser.error("Only preparation may explicitly select its frozen auxiliary evaluation split")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.run_id):
        parser.error("Use a plain new run ID")
    for path in (args.assets, args.probe_set, args.data):
        if not path.resolve().is_relative_to((VERSION / "inputs").resolve()):
            parser.error("Use verified version-local input assets and probes")
    if args.resume_from and not any(args.resume_from.resolve().is_relative_to((VERSION / part).resolve())
                                    for part in ("inputs", "outputs")):
        parser.error("Materialize cross-version states into this version before restoring")
    profile = read_json(args.profile)
    ids = [int(x) for x in args.gpus.split(",")]
    check_profile(profile, ids)
    seed = profile["seed"] if args.seed is None else args.seed
    total = profile["overrides"]["trainer.total_training_steps"]
    parent = validate(args.resume_from.resolve()) if args.resume_from else None
    start = parent["step"] if parent else 0
    if args.recovery_from and not args.resume_from:
        parser.error("Explicit recovery requires its previous attempt's last complete snapshot")
    recovery = recovery_contract(args.recovery_from, args.resume_from, seed=seed, phase=args.phase,
        branch=args.branch, probe=args.probe_set, assets=args.assets, freeze=args.freeze) if args.recovery_from else None
    if parent and (parent["world_size"] != len(ids) or
                   read_json(args.resume_from / "baseline_state.json")["full_horizon"] != total):
        parser.error("Parent topology or full scheduler horizon differs")
    schedule = plan(total, profile["checkpoint_interval"], start=start, stop=args.stop_after,
                    protected=tuple(sorted(set(args.pin_step))), keep=profile["checkpoint_keep"])
    if parent and start == 0:
        schedule["scheduled_steps"].remove(0)
    if args.injection_step is not None and not start < args.injection_step <= schedule["end"]:
        parser.error("Injection must be in this run's scheduled future updates")
    if args.repeat_initial_evaluation and args.phase != "control":
        parser.error("Repeated initial evaluation is an explicit consistency control")
    consumed_noop = (recovery and recovery["origin_reward_delta"] == 0 and
                     type(recovery["origin_injection_step"]) is int and recovery["origin_injection_step"] <= start)
    if args.reward_delta == 0 and (args.phase != "control" or (args.injection_step is None and not consumed_noop)):
        parser.error("Delta0 requires an explicit consistency intervention")
    if args.phase == "formal" and (args.freeze is None or not args.replay_from):
        parser.error("Formal Gate1 requires a freeze and fixed source history")
    if args.branch == "corrupt" and args.injection_step != (None if recovery and start >= 8 else 8):
        parser.error("Corrupt branch must inject only at k=8")
    if recovery and ((args.branch == "corrupt" and
                      (recovery["origin_injection_step"] != 8 or recovery["origin_reward_delta"] != 1)) or
                     (args.branch in {"clean", "repeat", "source", "prefix"} and
                      recovery["origin_injection_step"] is not None)):
        parser.error("Recovery cannot change the original intervention")
    if args.branch != "corrupt" and args.reward_delta != 0 and args.injection_step is not None:
        parser.error("Only the corrupt branch may revise reward")
    run = VERSION / "outputs" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "checkpoints", "processed"):
        (run / name).mkdir()
    report = inspect(ROOT, ids, minimum_disk_gib=0)
    atomic_json(run / "raw/preflight.json", report, immutable=True)
    if report["problems"]:
        atomic_json(run / "status.json", {"status": "preflight_failed", "problems": report["problems"]})
        print(json.dumps(report, ensure_ascii=False))
        return 2
    try:
        data = verify_data(args.data.resolve())
        model = model_identity(args.model_path.resolve())
        if read_json(args.model_path / "config.json").get("model_type") != profile["model_type"]:
            raise ValueError("Model architecture differs from the declared native candidate")
        if (data["splits"]["train"]["rows"] // profile["overrides"]["data.train_batch_size"]) * profile["overrides"]["trainer.total_epochs"] < total:
            raise ValueError("Train data does not cover the full scheduler horizon")
        checkout = ROOT / (".runtime/upstream/Agent0_baseline_" + file_digest(SRC / "apply_baseline_instrumentation.py")[:12])
        patch = prepare_checkout(checkout, ROOT / ".runtime/upstream/Agent0")
        overrides = dict(profile["overrides"])
        evaluation_split = args.evaluation_split or ("evaluation" if args.phase == "formal" else "calibration")
        overrides.update({"actor_rollout_ref.model.path": str(args.model_path.resolve()),
                          "data.train_files": str(args.data.resolve() / "train.parquet"),
                          "data.val_files": str(args.data.resolve() / f"{evaluation_split}.parquet"),
                          "trainer.default_local_dir": str(run / "checkpoints"),
                          "trainer.project_name": "SCOR-v02-Gate1", "trainer.experiment_name": args.run_id,
                          "trainer.save_freq": profile["checkpoint_interval"]})
        if args.resume_from:
            overrides.update({"trainer.resume_mode": "resume_path", "trainer.resume_from_path": str(args.resume_from.resolve())})
        excluded = {"actor_rollout_ref.model.path", "data.train_files", "data.val_files", "trainer.default_local_dir",
                    "trainer.project_name", "trainer.experiment_name", "trainer.resume_mode", "trainer.resume_from_path",
                    "trainer.save_freq", "trainer.test_freq", "trainer.val_before_train"}
        environment = {"python": sys.version, "packages": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True).splitlines()}
        learning = {"native_parameters": {k: v for k, v in overrides.items() if k not in excluded},
                    "environment_signature": digest(environment),
                    "processor_options": profile.get("processor_options", {}),
                    "execution_env": {"FLASH_ATTENTION_DETERMINISTIC": "1", "vision_flash_attention_deterministic": True},
                    "seed": seed, "model_signature": model["signature"], "dataset_signature": data["data_signature"],
                    "upstream": UPSTREAM_COMMIT, "patch": patch,
                    "source_sha256": {name: file_digest(SRC / name) for name in ENGINE_FILES}}
        protocol = {"phase": args.phase, "injection_step": args.injection_step,
                    "reward_delta": args.reward_delta, "repeat_initial_evaluation": args.repeat_initial_evaluation,
                    "replay_mode": "native_reanchor" if args.replay_from else "native_online",
                    "evaluation_split": evaluation_split}
        if recovery:
            protocol["recovery"] = recovery
        identity = signatures(learning, protocol, parent)
        imported = read_json(args.assets)
        if imported["status"] != "verified" or imported["manifest_signature"] != digest(
                {k: v for k, v in imported.items() if k != "manifest_signature"}):
            raise ValueError("Require a complete signed source import")
        canonical = imported["canonical_source_sha256"]
        template = dict(imported["canonical_learning"])
        template["seed"] = seed
        compatibility = {"conditions": compatible_learning(learning, template, canonical)}
        parent_learning = identity["learning"]
        if recovery and read_json(args.recovery_from / "manifest.json")["learning"] != learning:
            raise ValueError("Recovery cannot change execution sources or native learning conditions")
        if recovery and read_json(args.recovery_from / "manifest.json")["evaluation_split"] != evaluation_split:
            raise ValueError("Recovery cannot change its evaluation split")
        if parent and parent["learning_signature"] != identity["learning"]:
            parent_manifest = read_json(args.resume_from.resolve().parent / "source_manifest.json")
            # Imported states carry their exact original run manifest next to the snapshot.
            compatibility["parent"] = compatible_learning(learning, parent_manifest["learning"], canonical)
            if parent["learning_signature"] != compatibility["parent"]["source_learning_signature"]:
                raise ValueError("Parent is not the verified original source state")
            if parent_manifest["run_id"] != parent["run_id"]:
                raise ValueError("Parent source run ownership differs")
            parent_learning = parent["learning_signature"]
        if args.freeze is not None:
            freeze = read_json(args.freeze)
            if (freeze["status"] != "accepted" or set(freeze["acceptance"]) != ACCEPTANCE_KEYS or
                    set(freeze["evidence"]) != ACCEPTANCE_KEYS or not all(v is True for v in freeze["acceptance"].values())):
                raise ValueError("Gate1 freeze lacks mandatory real evidence")
            if (freeze["profile_sha256"] != file_digest(args.profile) or
                    freeze["model_signature"] != model["signature"] or
                    freeze["dataset_signature"] != data["data_signature"] or
                    freeze["source_sha256"] != learning["source_sha256"] or
                    freeze["environment_signature"] != learning["environment_signature"] or
                    seed not in freeze["seeds"] or
                    freeze["probe_manifest_sha256"] != file_digest(args.probe_set / "manifest.json")):
                raise ValueError("Gate1 frozen condition/source/seed/probe differs")
            for key, evidence in freeze["evidence"].items():
                if file_digest(Path(evidence["path"])) != evidence["sha256"]:
                    raise ValueError("Gate1 evidence changed: " + key)
        full_estimate = model["weight_bytes"] * 13 // 2
        initial_estimate = model["weight_bytes"] * 11 // 5
        needed = storage_peak(schedule, full_estimate, initial_estimate) + 7 * 2**30
        if shutil.disk_usage(ROOT).free < needed:
            raise OSError(f"Full-state peak plus raw/margin requires {needed / 2**30:.2f} GiB free")
        replay = None
        if args.replay_from:
            source = args.replay_from.resolve()
            if not any(source.is_relative_to((VERSION / part).resolve()) for part in ("inputs", "outputs")):
                raise ValueError("Replay accepts only verified version-local histories")
            source_manifest = read_json(source / "manifest.json")
            source_learning = source_manifest["signatures"]["learning"]
            if source_learning != identity["learning"]:
                compatibility["history"] = compatible_learning(learning, source_manifest["learning"], canonical)
                if source_learning != compatibility["history"]["source_learning_signature"]:
                    raise ValueError("History source learning identity differs")
            end = args.replay_end or schedule["end"]
            if not start < args.replay_start <= end <= schedule["end"]:
                raise ValueError("Invalid replay interval")
            replay = {"source": str(source), "start": args.replay_start, "end": end,
                      "source_learning_signature": source_learning}
        if recovery:
            previous = read_json(args.recovery_from / "manifest.json")
            old_replay = previous.get("replay")
            if (schedule["end"] != previous["checkpoint_plan"]["end"] or bool(replay) != bool(old_replay) or
                    (replay and (any(replay[k] != old_replay[k] for k in ["source", "source_learning_signature"]) or
                                 replay["start"] != start + 1 or replay["end"] != old_replay["end"]))):
                raise ValueError("Recovery changed its horizon or fixed future history")
        manifest = {"run_id": args.run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "purpose": profile["purpose"], "phase": args.phase, "seed": seed,
                    "model_id": profile["model_id"], "model_revision": profile["model_revision"], "model": model,
                    "dataset_directory": str(args.data.resolve()), "data_signature": data["data_signature"],
                    "learning": learning, "protocol": protocol, "signatures": identity, "parent": parent,
                    "selected_gpus": ids, "overrides": overrides, "checkpoint_plan": schedule,
                    "checkpoint_estimate_bytes": full_estimate, "storage_required_bytes": needed,
                    "stop_after": args.stop_after, "replay": replay, "injection_step": args.injection_step,
                    "reward_delta": args.reward_delta,
                    "evaluation_split": evaluation_split, "sampler_bitwise_recovery_verified": False,
                    "gate1": {"branch": args.branch, "compatibility": compatibility,
                              "parent_learning_signature": parent_learning,
                              "import_manifest_sha256": file_digest(args.assets),
                              "probe_directory": str(args.probe_set.resolve()),
                              "probe_manifest_sha256": file_digest(args.probe_set / "manifest.json"),
                              "freeze_sha256": file_digest(args.freeze) if args.freeze else None}}
        if recovery:
            manifest["gate1"]["recovery"] = recovery
        if seed == 20261011 and replay and replay["start"] <= 8 <= replay["end"]:
            label = "seed_20261011_corrupt_S8" if args.branch == "corrupt" else "seed_20261011_clean_S8"
            manifest["gate1"]["reference_U8_worker_state"] = str(
                Path(imported["states"][label]["path"]) / "baseline_worker_state.json")
        atomic_json(run / "manifest.json", manifest, immutable=True)
        source = run / "raw/source"
        for path in SRC.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            copy = source / path.relative_to(SRC)
            copy.parent.mkdir(parents=True, exist_ok=True)
            with copy.open("xb") as stream:
                stream.write(path.read_bytes())
        for name, sha in learning["source_sha256"].items():
            if file_digest(source / name) != sha:
                raise ValueError("Engine changed while recording source identity")
        atomic_json(source / "profile.json", profile, immutable=True)
        if args.preflight_only:
            atomic_json(run / "status.json", {"status": "preflight_passed_no_training"})
            print(json.dumps({"run_id": args.run_id, "preflight": "passed", "required_disk_gib": needed / 2**30}))
            return 0
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": args.gpus, "BASELINE_RUN_DIR": str(run),
                    "PYTHONPATH": str(SRC) + os.pathsep + str(checkout / "Agent0-VL"),
                    "RAY_ADDRESS": "local", "RAY_TMPDIR": str(ROOT / ".runtime/b"),
                    "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": str(2 * 2**30),
                    "VLLM_USE_V1": "0", "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled",
                    "FLASH_ATTENTION_DETERMINISTIC": "1",
                    "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HOME": str(ROOT / ".runtime/cache/huggingface"),
                    "XDG_CACHE_HOME": str(ROOT / ".runtime/cache"), "TMPDIR": str(ROOT / ".runtime/tmp"),
                    "TORCH_HOME": str(ROOT / ".runtime/cache/torch"), "TRITON_CACHE_DIR": str(ROOT / ".runtime/cache/triton"),
                    "TORCHINDUCTOR_CACHE_DIR": str(ROOT / ".runtime/cache/torchinductor"),
                    "CUDA_CACHE_PATH": str(ROOT / ".runtime/cache/cuda"), "OMP_NUM_THREADS": "4"})
        env.pop("SCOR_RUN_DIR", None)
        command = [sys.executable, "-m", "verl.trainer.main_ppo", "--config-name=agent0_trainer"]
        command += [key + "=" + hydra_value(value) for key, value in overrides.items()]
        command += ["hydra.run.dir=" + hydra_value(str(run / "raw/hydra")), "hydra.output_subdir=null", "hydra.job.chdir=false"]
        atomic_json(run / "raw/command.json", {"argv": command}, immutable=True)
        atomic_json(run / "raw/environment.json", environment, immutable=True)
        atomic_json(run / "status.json", {"status": "running"})
        with (run / "raw/console.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, env=env, cwd=checkout / "Agent0-VL", stdout=log, stderr=subprocess.STDOUT)
        end = schedule["end"]
        if result.returncode == 0:
            validate(run / f"checkpoints/global_step_{end}", identity["learning"])
            if not (run / f"raw/completed_step_{end:06d}.json").exists():
                raise ValueError("Native process exited without a completed update receipt")
        status = "intentionally_stopped" if (run / "raw/intentional_stop.json").exists() else (
            "completed" if result.returncode == 0 else "failed")
        atomic_json(run / "status.json", {"status": status, "returncode": result.returncode})
        print(json.dumps({"run_id": args.run_id, "status": status, "returncode": result.returncode}))
        return result.returncode
    except BaseException as error:
        atomic_json(run / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
