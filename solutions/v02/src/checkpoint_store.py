"""Complete baseline state certificates, atomic commits and owned retention."""
from __future__ import annotations

import os
from pathlib import Path
import shutil

from pilot_utils import atomic_json, digest, file_digest, read_json
from records import fsync_directory


def plan(total: int, interval: int, *, start: int = 0, stop: int | None = None,
         protected: tuple = (0,), keep: int = 1) -> dict:
    end = total if stop is None else stop
    if not (0 <= start < end <= total) or interval < 1 or keep < 1:
        raise ValueError("Invalid horizon, checkpoint interval or retention")
    if any(type(p) is not int or not 0 <= p <= total for p in protected):
        raise ValueError("Protected step is outside the full horizon")
    steps = [s for s in range(start + 1, end + 1)
             if s % interval == 0 or s in protected or s == end]
    if start == 0:
        steps.insert(0, 0)
    live, peak = [], 0
    for step in steps:
        live.append(step)
        peak = max(peak, len(live))
        rolling = [s for s in live if s not in protected]
        live = [s for s in live if s in protected or s in rolling[-keep:]]
    return {"full_horizon": total, "start": start, "end": end, "interval": interval,
            "protected": list(protected), "keep": keep, "scheduled_steps": steps,
            "peak_owned_slots": peak}


def required_names(world: int) -> set[str]:
    if type(world) is not int or not 1 <= world <= 4:
        raise ValueError("World size must be between one and four")
    return {f"actor/{kind}_world_size_{world}_rank_{rank}.pt"
            for kind in ("model", "optim", "extra_state") for rank in range(world)} | {
                "data.pt", "baseline_driver_rng.pt", "baseline_state.json", "baseline_worker_state.json"}


def certify(path: Path, *, learning: str, run_id: str, step: int,
            world: int, parent: dict | None) -> dict:
    if path.is_symlink() or not required_names(world).issubset(
            str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()):
        raise ValueError("Incomplete snapshot; no completion certificate written")
    files, verified_metadata = {}, {}
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        if p.is_symlink() or not p.resolve().is_relative_to(path.resolve()) or p.stat().st_size == 0:
            raise ValueError("Invalid checkpoint member")
        with p.open("rb") as stream:
            os.fsync(stream.fileno())
        before = p.stat()
        sha = file_digest(p)
        after = p.stat()
        identity = lambda s: [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]
        if identity(before) != identity(after):
            raise ValueError("Checkpoint changed while its content was being certified")
        name = str(p.relative_to(path))
        files[name] = {"bytes": after.st_size, "sha256": sha}
        verified_metadata[name] = identity(after)
    certificate = {"learning_signature": learning, "run_id": run_id, "step": step,
                   "world_size": world, "parent": parent, "files": files,
                   "sampling_rng_bitwise_verified": False}
    certificate["state_signature"] = digest(certificate)
    atomic_json(path / "baseline_complete.json", certificate, immutable=True)
    fsync_directory(path)
    return {**certificate, "_verified_metadata": verified_metadata}


def validate(path: Path, learning: str | None = None, *, check_hashes: bool = True) -> dict:
    if path.is_symlink():
        raise ValueError("Snapshot directory must be a physical version-local asset")
    certificate = read_json(path / "baseline_complete.json")
    if certificate["state_signature"] != digest({k: v for k, v in certificate.items() if k != "state_signature"}):
        raise ValueError("Checkpoint certificate identity mismatch")
    if learning is not None and certificate["learning_signature"] != learning:
        raise ValueError("Snapshot learning conditions differ")
    if path.name != f"global_step_{certificate['step']}":
        raise ValueError("Snapshot step and folder differ")
    if not required_names(certificate["world_size"]).issubset(certificate["files"]):
        raise ValueError("Snapshot omits required training state")
    for name, info in certificate["files"].items():
        p = path / name
        if p.is_symlink() or not p.resolve().is_relative_to(path.resolve()):
            raise ValueError("Checkpoint member escapes snapshot")
        if not p.is_file() or p.stat().st_size != info["bytes"] or (check_hashes and file_digest(p) != info["sha256"]):
            raise ValueError("Snapshot missing, truncated or corrupted member")
    state = read_json(path / "baseline_state.json")
    if state["completed_step"] != certificate["step"] or state["next_step"] != certificate["step"] + 1:
        raise ValueError("Snapshot cursor differs from its certificate")
    return certificate


def verified_replacement(path: Path, learning: str | None, proof: dict | None) -> dict:
    if proof is None:
        return validate(path, learning)
    certificate = validate(path, learning, check_hashes=False)
    if certificate != {k: v for k, v in proof.items() if k != "_verified_metadata"}:
        raise ValueError("Replacement certification proof differs")
    if set(proof["_verified_metadata"]) != set(certificate["files"]):
        raise ValueError("Replacement proof omits members")
    for name, expected in proof["_verified_metadata"].items():
        st = (path / name).stat()
        if [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns] != expected:
            raise ValueError("Replacement changed after its complete content hash")
    return certificate


def commit(pending: Path, destination: Path, *, proof: dict | None = None) -> None:
    # A pending container can hold native tracker files; only the snapshot commits.
    verified_replacement(pending, None, proof)
    # Filesystems can give two writes the same coarse timestamp. The final
    # commit therefore checks content again; retirement can reuse this proof.
    validate(pending)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Snapshots are immutable")
    os.rename(pending, destination)
    fsync_directory(destination.parent)


def retire(root: Path, newest: Path, learning: str, run_id: str,
           keep: int, protected: list[int], *, proof: dict | None = None) -> list[str]:
    if newest.parent.resolve() != root.resolve() or keep < 1:
        raise ValueError("Invalid owned checkpoint retention")
    latest = verified_replacement(newest, learning, proof)
    if latest["run_id"] != run_id:
        raise ValueError("Replacement belongs to another run")
    candidates = []
    for path in root.glob("global_step_*"):
        if path.name in {f"global_step_{step}" for step in protected}:
            continue
        try:
            # Ownership/completeness is enough to retire an old owned unprotected replica.
            # Its bytes are never used to certify the new replacement.
            certificate = validate(path, learning, check_hashes=False)
        except (OSError, ValueError, KeyError, TypeError):
            continue  # Incomplete/foreign recovery assets remain untouched.
        if certificate["run_id"] == run_id and certificate["step"] not in protected:
            candidates.append((certificate["step"], path))
    candidates.sort()
    deleted = []
    for step, path in candidates[:-keep]:
        if path == newest:
            raise ValueError("Newest snapshot cannot be retired")
        shutil.rmtree(path)
        deleted.append(path.name)
    fsync_directory(root)
    return deleted
