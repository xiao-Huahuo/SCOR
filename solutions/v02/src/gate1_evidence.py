"""Physical paths and immutable evidence shared by Gate1 workflow checks."""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath

from pilot_utils import ROOT, SRC, VERSION, atomic_json, digest, file_digest, read_json

SEEDS = [20261011, 20261012, 20261013, 20261014, 20261015]
ACCEPTANCE_KEYS = {"source_asset_import", "actual_parent_restore", "fixed_unit_replay",
                   "actual_interruption_recovery", "probe_inputs", "threshold_calibration",
                   "storage_budget", "independent_code_review"}


def physical(path: Path, root: Path | None = None) -> Path:
    path = Path(path).absolute()
    root = (ROOT if root is None else root).absolute()
    if ".." in path.parts or not path.is_relative_to(root):
        raise ValueError("Evidence path is outside its owned physical root")
    if root.is_symlink() or any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)):
        raise ValueError("Symlink evidence or ancestor is forbidden")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes its root")
    return path


def member(root: Path, name: str) -> Path:
    parsed = PurePosixPath(name)
    if (not name or parsed.is_absolute() or ".." in parsed.parts or "\\" in name or
            str(parsed) != name or name == "."):
        raise ValueError("Invalid certified member path")
    return physical(root / name, root)


def evidence(path: Path) -> dict:
    path = physical(path)
    if not path.is_file():
        raise ValueError("Evidence file is missing")
    return {"path": str(path), "sha256": file_digest(path)}


def verify_evidence(items: list[dict]) -> None:
    if not items:
        raise ValueError("Acceptance has no durable evidence")
    for item in items:
        if evidence(Path(item["path"]))["sha256"] != item["sha256"]:
            raise ValueError("Acceptance evidence changed")


def immutable_json(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Existing immutable receipt differs")
    else:
        atomic_json(path, value, immutable=True)


def finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def validate_freeze(path: Path, *, check_workflow: bool = True) -> dict:
    frozen = read_json(physical(path))
    if (frozen.get("status") != "accepted" or frozen.get("seeds") != SEEDS or
            frozen.get("horizon") != 32 or frozen.get("k", frozen.get("injection_step")) != 8 or
            set(frozen.get("acceptance", {})) != ACCEPTANCE_KEYS or
            set(frozen.get("evidence", {})) != ACCEPTANCE_KEYS or
            not all(v is True for v in frozen["acceptance"].values()) or
            not finite(frozen.get("epsilon_D")) or type(frozen.get("epsilon_D")) not in (int, float) or
            frozen["epsilon_D"] <= 0):
        raise ValueError("Require the accepted five-seed T32/k8 freeze and all real witnesses")
    verify_evidence(list(frozen["evidence"].values()))
    for item in frozen["evidence"].values():
        if read_json(Path(item["path"])).get("passed") is not True:
            raise ValueError("Frozen acceptance witness is not passed")
    sources = dict(frozen["source_sha256"])
    from run_gate1 import ENGINE_FILES
    if set(sources) != set(ENGINE_FILES):
        raise ValueError("Freeze omits the complete reviewed native execution source")
    if check_workflow:
        workflow = frozen.get("workflow_source_sha256", {})
        required = {"run_gate1_sequence.py", "gate1_state.py", "gate1_validation.py",
                    "gate1_references.py", "gate1_evidence.py", "export_gate1.py"}
        if set(workflow) != required:
            raise ValueError("Freeze omits reviewed workflow sources")
        sources.update(workflow)
    for name, sha in sources.items():
        if file_digest(member(SRC, name)) != sha:
            raise ValueError("Frozen execution or workflow source changed")
    return frozen
