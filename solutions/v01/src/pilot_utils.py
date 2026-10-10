"""Small, dependency-free helpers for immutable pilot receipts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
VERSION = SRC.parent
UPSTREAM_COMMIT = "f775b5101e62fe92976831adf4a21a38fcc0a767"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: object, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(temporary, path)  # fail rather than overwrite an existing receipt
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_profile(profile: dict) -> None:
    overrides = profile["overrides"]
    if not 1 <= overrides["trainer.n_gpus_per_node"] <= 4:
        raise ValueError("Server rules permit at most four GPUs")
    required = {
        "algorithm.adv_estimator": "grpo",
        "actor_rollout_ref.rollout.rollout_type": "agent0",
        "actor_rollout_ref.rollout.enable_tool_execution": True,
        "actor_rollout_ref.rollout.enable_verification": True,
        "actor_rollout_ref.rollout.enable_self_repair": True,
        "reward_model.reward_manager": "agent0",
    }
    for key, value in required.items():
        if overrides.get(key) != value:
            raise ValueError(f"Native-chain requirement missing: {key}={value}")
    group = overrides["actor_rollout_ref.rollout.n"]
    if group < 2:
        raise ValueError("GRPO requires multiple responses in this pilot")
    if overrides["trainer.balance_batch"]:
        raise ValueError("Keep GRPO groups contiguous")
    if overrides["trainer.total_training_steps"] > 8:
        raise ValueError("This entry is limited to small engineering pilots")


def hydra_value(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))
