"""Periodic full snapshots; retire owned snapshots only after a durable save."""
from __future__ import annotations

import functools
from pathlib import Path
import shutil
import time

from pilot_utils import atomic_json, file_digest, read_json


def plan_checkpoints(total: int, interval: int, *, start: int = 0,
                     stop: int | None = None, protected=(), keep: int | None = None,
                     enabled: bool = True) -> dict:
    if not 0 <= start < total or interval < 1:
        raise ValueError("Require a positive interval and unfinished training steps")
    if stop is not None and not start < stop <= total:
        raise ValueError("Stop step must be after the restored step and within the budget")
    if keep is not None and keep < 1:
        raise ValueError("Keep at least one full snapshot, or explicitly disable new saves")
    protected = sorted(set(protected))
    if any(not 1 <= step <= total for step in protected):
        raise ValueError("Protected checkpoint step outside the training budget")
    if not enabled and protected:
        raise ValueError("Protected snapshots cannot be combined with disabled saves")
    end = stop if stop is not None else total
    steps = [step for step in range(start + 1, end + 1)
             if enabled and (step % interval == 0 or step == end or step in protected)]
    live, peak = [], 0
    for step in steps:
        live.append(step)
        peak = max(peak, len(live))  # New snapshot coexists with old ones until certified.
        if keep is not None:
            retained = set(live[-keep:]) | set(protected)
            live = [item for item in live if item in retained]
    return {"enabled": enabled, "interval_steps": interval, "keep_latest": keep,
            "protected_steps": protected, "scheduled_steps": steps,
            "peak_snapshot_slots": peak,
            "commit_unit": "full checkpoint boundary; later updates are an uncommitted recovery segment"}


def required_disk_bytes(model_bytes: int, plan: dict) -> int:
    if not plan["enabled"]:
        return 5 * 2**30
    # Observed 3B full snapshot exceeds six times its BF16 tensor-file size.
    estimate = (model_bytes * 13 + 1) // 2
    return 55 * 2**30 + (plan["peak_snapshot_slots"] - 1) * estimate


def complete_checkpoint(path: Path, signature: str | None = None) -> bool:
    try:
        if path.is_symlink():
            return False
        cert = read_json(path / "scor_complete.json")
        world = cert["world_size"]
        if type(world) is not int or not 1 <= world <= 4 or path.name != f"global_step_{cert['step']}":
            return False
        if not isinstance(cert.get("training_signature"), str) or not cert["training_signature"]:
            return False
        if signature is not None and cert["training_signature"] != signature:
            return False
        shards = cert["worker_shards"]
        if len(set(shards)) != world * 3:
            return False
        for name in shards + ["data.pt", "scor_driver_rng.pt"]:
            file = path / name
            if not file.resolve().is_relative_to(path.resolve()):
                return False
            if not file.is_file() or file.stat().st_size == 0:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def retire_checkpoints(root: Path, newest: Path, signature: str,
                       keep: int | None, protected=()) -> list[str]:
    if newest.parent.resolve() != root.resolve() or not complete_checkpoint(newest, signature):
        raise ValueError("Refusing retirement before a complete owned replacement exists")
    if keep is None:
        return []
    if keep < 1:
        raise ValueError("Keep at least one snapshot")
    complete = sorted((path for path in root.glob("global_step_*")
                       if complete_checkpoint(path, signature)),
                      key=lambda path: read_json(path / "scor_complete.json")["step"])
    if not complete or complete[-1] != newest:
        raise ValueError("Replacement is not the newest complete snapshot")
    retained = set(complete[-keep:])
    deleted = []
    for path in complete:
        step = read_json(path / "scor_complete.json")["step"]
        if path not in retained and step not in protected:
            shutil.rmtree(path)  # Only certified snapshots within this new run directory.
            deleted.append(path.name)
    return deleted


def install(config) -> None:
    import os
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.tracking import Tracking

    if getattr(RayPPOTrainer, "_scor_checkpoint_policy", False):
        return
    run = Path(os.environ["SCOR_RUN_DIR"])
    manifest = read_json(run / "manifest.json")
    policy = manifest.get("checkpoint_policy", {"enabled": config.trainer.save_freq > 0,
                           "keep_latest": None, "protected_steps": []})
    if config.trainer.get("max_actor_ckpt_to_keep") is not None or config.trainer.get(
            "remove_previous_ckpt_in_save", False):
        raise ValueError("Native pre-save deletion must remain disabled")
    original_init, original_save, original_log = (
        RayPPOTrainer.__init__, RayPPOTrainer._save_checkpoint, Tracking.log)
    current = {}

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        current["trainer"] = self

    @functools.wraps(original_save)
    def save(self):
        result = original_save(self)  # Includes native shards and the existing fsynced certificate.
        root = Path(self.config.trainer.default_local_dir)
        directory = root / f"global_step_{self.global_steps}"
        if not complete_checkpoint(directory, manifest["training_signature"]):
            raise RuntimeError("Full snapshot certification failed; no old snapshot removed")
        tracker = root / "latest_checkpointed_iteration.txt"
        if tracker.is_file():
            with tracker.open("rb") as stream:
                os.fsync(stream.fileno())
        for parent in (directory / "actor", directory, root):
            fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        atomic_json(run / f"raw/durable_checkpoint_step_{self.global_steps:06d}.json",
                    {"step": self.global_steps, "training_signature": manifest["training_signature"],
                     "certificate_sha256": file_digest(directory / "scor_complete.json"),
                     "model_state_committed": True}, immutable=True)
        atomic_json(run / f"raw/retention_plan_step_{self.global_steps:06d}.json",
                    {"replacement_step": self.global_steps, "keep_latest": policy["keep_latest"],
                     "protected_steps": policy["protected_steps"]}, immutable=True)
        deleted = retire_checkpoints(root, directory, manifest["training_signature"],
                                     policy["keep_latest"], policy["protected_steps"])
        atomic_json(run / f"raw/retention_result_step_{self.global_steps:06d}.json",
                    {"removed_owned_snapshots": deleted}, immutable=True)
        return result

    @functools.wraps(original_log)
    def log(self, data, step, backend=None):
        trainer = current.get("trainer")
        forced = step in policy["protected_steps"] or manifest.get("stop_after") == step
        if trainer is not None and policy["enabled"] and forced and (
                run / f"raw/group_step_{step:06d}.json").exists():
            directory = Path(trainer.config.trainer.default_local_dir) / f"global_step_{step}"
            if not complete_checkpoint(directory, manifest["training_signature"]):
                started = time.monotonic()
                trainer._save_checkpoint()
                data["scor/extra_checkpoint_seconds"] = time.monotonic() - started
        return original_log(self, data, step, backend)

    RayPPOTrainer.__init__ = initialize
    RayPPOTrainer._save_checkpoint = save
    Tracking.log = log
    RayPPOTrainer._scor_checkpoint_policy = True
