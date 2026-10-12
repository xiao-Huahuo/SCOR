"""Certified full-state comparisons and resumable temporary-weight retirement."""
from __future__ import annotations

from pathlib import Path
import re

from checkpoint_store import required_names, retire, validate
from gate1_evidence import evidence, finite, immutable_json, member, physical, verify_evidence
from pilot_utils import VERSION, digest, file_digest, read_json
from records import fsync_directory, model_identity, value_hash


def certificate_metadata(path: Path) -> dict:
    """Check certificate and retained metadata; this never proves restorability."""
    path = physical(path)
    certificate = read_json(member(path, "baseline_complete.json"))
    if (certificate["state_signature"] != digest({k: v for k, v in certificate.items() if k != "state_signature"}) or
            type(certificate["step"]) is not int or path.name != f"global_step_{certificate['step']}" or
            not required_names(certificate["world_size"]).issubset(certificate["files"])):
        raise ValueError("Checkpoint certificate identity/schema differs")
    for name, info in certificate["files"].items():
        p = member(path, name)
        if (type(info["bytes"]) is not int or info["bytes"] <= 0 or
                not re.fullmatch(r"[0-9a-f]{64}", info["sha256"])):
            raise ValueError("Invalid certified member size/hash")
        if name in {"baseline_state.json", "baseline_worker_state.json", "baseline_driver_rng.pt", "data.pt"}:
            if not p.is_file() or p.stat().st_size != info["bytes"] or file_digest(p) != info["sha256"]:
                raise ValueError("Retained checkpoint metadata changed")
    state = read_json(path / "baseline_state.json")
    workers = read_json(path / "baseline_worker_state.json")
    keys = {"rank", "model", "buffers", "optimizer", "optimizer_groups", "scheduler", "rng_hash",
            "all_parameters_and_optimizer_finite"}
    if (state["completed_step"] != certificate["step"] or state["next_step"] != certificate["step"] + 1 or
            state["full_horizon"] != 32 or not finite(state) or not finite(workers) or
            len(workers) != certificate["world_size"] or
            {w.get("rank") for w in workers} != set(range(certificate["world_size"])) or
            any(not keys.issubset(w) or w["all_parameters_and_optimizer_finite"] is not True for w in workers)):
        raise ValueError("Checkpoint cursor/worker identities/finite-state evidence differs")
    return certificate


def _retirement_for(path: Path, *, pending: bool = False) -> tuple[dict, dict]:
    run = path.parent.parent
    receipt_path = run / ("raw/temporary_weight_retirement_plan.json" if pending else
                          "raw/temporary_weight_retirement_complete.json")
    receipt = read_json(physical(receipt_path))
    proof = physical(Path(receipt["group_proof"]))
    if file_digest(proof) != receipt["group_proof_sha256"]:
        raise ValueError("Retirement proof changed")
    verify_evidence(receipt["evidence"])
    if (receipt["run_id"] != run.name or receipt["directly_restorable_after_retirement"] is not False or
            read_json(proof).get("passed") is not True):
        raise ValueError("Invalid weight retirement ownership or proof")
    anchor = receipt["reconstruction"]
    if anchor["kind"] == "retained_full_state_and_fixed_history":
        restored = certified_snapshot(physical(Path(anchor["path"])))
        if restored["state_signature"] != anchor["state_signature"]:
            raise ValueError("Retirement reconstruction anchor changed")
    elif anchor["kind"] == "base_model_seed_and_recorded_native_history":
        if model_identity(physical(Path(anchor["path"])))["signature"] != anchor["model_signature"]:
            raise ValueError("Retirement base model changed")
    else:
        raise ValueError("Retirement has no valid reconstruction strategy")
    return receipt, receipt["snapshots"][path.name]


def certified_snapshot(path: Path, *, allow_retired: bool = False, pending: bool = False) -> dict:
    certificate = certificate_metadata(path)
    if not allow_retired:
        return validate(path)
    missing = [name for name in certificate["files"] if not member(path, name).is_file()]
    retirement = None
    if missing or (path.parent.parent / "raw/temporary_weight_retirement_complete.json").exists():
        _, retirement = _retirement_for(path, pending=pending)
        if (retirement["certificate_sha256"] != file_digest(path / "baseline_complete.json") or
                retirement["state_signature"] != certificate["state_signature"] or
                retirement["retired_members"] != {name: info for name, info in certificate["files"].items()
                    if name.startswith(("actor/model_", "actor/optim_"))}):
            raise ValueError("Retirement receipt does not certify these snapshot members")
    for name, info in certificate["files"].items():
        p = member(path, name)
        if not p.is_file():
            if retirement is None or name not in retirement["retired_members"]:
                raise ValueError("Missing state lacks an explicit retirement receipt")
        elif p.stat().st_size != info["bytes"] or file_digest(p) != info["sha256"]:
            raise ValueError("Snapshot member changed")
    return certificate


def state_fingerprint(path: Path, *, allow_retired: bool = False) -> dict:
    certified_snapshot(path, allow_retired=allow_retired)
    return metadata_fingerprint(path)


def metadata_fingerprint(path: Path) -> dict:
    """Compare certified small metadata, without a model restoration claim."""
    import torch
    certificate = certificate_metadata(path)
    state = read_json(path / "baseline_state.json")
    return {"step": certificate["step"], "workers": read_json(path / "baseline_worker_state.json"),
            "driver_rng": value_hash(torch.load(path / "baseline_driver_rng.pt", map_location="cpu", weights_only=False)),
            "data_cursor": value_hash(torch.load(path / "data.pt", map_location="cpu", weights_only=False)),
            "position": {key: state[key] for key in ["completed_step", "next_step", "full_horizon", "zero_signal_streak"]}}


def compare_states(left: Path, right: Path, *, allow_retired: bool = False) -> dict:
    a, b = state_fingerprint(left, allow_retired=allow_retired), state_fingerprint(right, allow_retired=allow_retired)
    checks = {key: a[key] == b[key] for key in a}
    return {"passed": all(checks.values()), "checks": checks,
            "left_certificate_sha256": file_digest(left / "baseline_complete.json"),
            "right_certificate_sha256": file_digest(right / "baseline_complete.json"),
            "scope": "full parameter/buffer/Adam/group/scheduler/workerRNG plus driverRNG/cursor; no sampler claim"}


def _anchor(parent: dict | None, anchor: dict) -> bool:
    while parent:
        if parent["state_signature"] == anchor["state_signature"]:
            return True
        parent = parent.get("parent")
    return False


def retire_prefix_initial(run: Path, reference_zero: Path, reference_seven: Path) -> dict:
    """Retire only a successfully rebuilt new prefix's unprotected S0; full S7 stays."""
    from gate1_references import compare_rebuilt, validate_reference
    from gate1_validation import validate_run
    run = physical(run, VERSION / "outputs")
    data, status = read_json(run / "manifest.json"), read_json(run / "status.json")
    if (run.parent != (VERSION / "outputs").absolute() or data["run_id"] != run.name or
            data["phase"] != "preparation" or data["gate1"]["branch"] != "prefix" or
            status.get("returncode") != 0 or status.get("status") != "intentionally_stopped" or
            data["checkpoint_plan"]["end"] != 7 or data["checkpoint_plan"]["protected"] != [7]):
        raise ValueError("Only a terminal new prefix with protected S7 and unprotected S0 may retire its initial point")
    root = run / "checkpoints"
    anchor = certified_snapshot(root / "global_step_7")
    if anchor["run_id"] != run.name or anchor["learning_signature"] != data["signatures"]["learning"]:
        raise ValueError("Prefix S7 anchor ownership/learning differs")
    plan_path = run / "raw/prefix_initial_retirement_plan.json"
    complete_path = run / "raw/prefix_initial_retirement_complete.json"
    if plan_path.exists():
        receipt = read_json(plan_path)
        verify_evidence(receipt["evidence"])
        if receipt["anchor_state_signature"] != anchor["state_signature"]:
            raise ValueError("Prefix S7 changed after its retirement plan")
    else:
        validated = validate_run(run)
        source = root / "global_step_0"
        cert = certified_snapshot(source)
        zero = compare_rebuilt(source, reference_zero)
        seven = compare_rebuilt(root / "global_step_7", reference_seven)
        captured = run / "raw/initial_state/global_step_0"
        captured_certificate = certificate_metadata(captured)
        if (cert["run_id"] != run.name or cert["learning_signature"] != data["signatures"]["learning"] or
                not zero["passed"] or not seven["passed"] or captured_certificate != cert or
                metadata_fingerprint(captured) != state_fingerprint(source) or
                {p.name for p in root.glob("global_step_*")} != {"global_step_0", "global_step_7"}):
            raise ValueError("Initial/prefix state or retained small metadata failed its actual historical audit")
        names = [name for name in cert["files"] if name.startswith(("actor/model_", "actor/optim_"))]
        if any(member(source, name).stat().st_nlink != 1 for name in names):
            raise ValueError("Prefix initial weights have unexpected shared physical ownership")
        receipt = {"run_id": run.name, "initial_state_signature": cert["state_signature"],
            "anchor_state_signature": anchor["state_signature"], "initial_checkpoint_bytes": sum(i["bytes"] for i in cert["files"].values()),
            "retired_large_shard_bytes": sum(cert["files"][name]["bytes"] for name in names),
            "initial_directly_restorable_after_retirement": False, "full_S7_retained": True,
            "checks": {"S0": zero, "S7": seven}, "evidence": [*validated["evidence"], *zero["evidence"], *seven["evidence"],
                *[evidence(member(captured, name)) for name in ["baseline_complete.json", *cert["files"]] if name not in names]]}
        immutable_json(plan_path, receipt)
    if (root / "global_step_0").exists():
        deleted = retire(root, root / "global_step_7", data["signatures"]["learning"], run.name, 1, [])
        if deleted != ["global_step_0"]:
            raise ValueError("Prefix initial retirement removed an unexpected owned point")
    certified_snapshot(root / "global_step_7")
    immutable_json(complete_path, receipt)
    return receipt


def retire_temporary_weights(run: Path, group_proof: Path, retained_prefix: Path | None = None) -> dict:
    """Only accepted terminal runs with explicit no-pending-dependency proof."""
    from gate1_validation import validate_run
    run = physical(run, VERSION / "outputs")
    if run.parent != (VERSION / "outputs").absolute():
        raise ValueError("Temporary weights must belong to a direct version-local run")
    manifest = read_json(run / "manifest.json")
    status = read_json(run / "status.json")
    proof = read_json(physical(group_proof))
    temporary = (manifest["gate1"]["branch"] in {"control", "clean", "repeat"} or
                 (manifest["phase"] in {"control", "preparation"} and manifest["gate1"]["branch"] == "corrupt"))
    if (manifest["run_id"] != run.name or not temporary or status.get("returncode") != 0 or
            status.get("status") not in {"completed", "intentionally_stopped"} or
            proof.get("passed") is not True or run.name not in proof.get("accepted_run_ids", []) or
            proof.get("pending_dependents") != []):
        raise ValueError("Run is foreign, unaccepted or has undeclared pending dependencies")
    verify_evidence(proof["evidence"])
    plan_path = run / "raw/temporary_weight_retirement_plan.json"
    complete_path = run / "raw/temporary_weight_retirement_complete.json"
    if complete_path.exists():
        receipt = read_json(complete_path)
        if receipt["group_proof_sha256"] != file_digest(group_proof):
            raise ValueError("Completed retirement belongs to another proof")
        for name in receipt["snapshots"]:
            certified_snapshot(run / "checkpoints" / name, allow_retired=True)
        return receipt
    pending = plan_path.exists()
    validation = validate_run(run, allow_retired=pending, pending_retirement=pending)
    covered = {item["path"]: item["sha256"] for item in proof["evidence"]}
    if any(covered.get(item["path"]) != item["sha256"] for item in validation["evidence"]):
        raise ValueError("Proof does not cover all actual run/unit/evaluation evidence")
    original = validation["origin_manifest"]
    if original.get("parent"):
        if retained_prefix is None:
            raise ValueError("A full retained reconstruction anchor is required")
        retained_prefix = physical(retained_prefix)
        anchor = certified_snapshot(retained_prefix)
        if not _anchor(original["parent"], anchor):
            raise ValueError("Retained anchor is not this run's actual state ancestry")
        reconstruction = {"kind": "retained_full_state_and_fixed_history", "path": str(retained_prefix),
                          "state_signature": anchor["state_signature"], "step": anchor["step"]}
        anchor_evidence = [evidence(retained_prefix / "baseline_complete.json")]
        replay = original.get("replay")
        source = physical(Path(replay["source"])) if replay else None
        if source is None or any(not (source / f"raw/completed_step_{s:06d}.json").is_file()
                                 for s in range(anchor["step"] + 1, original["checkpoint_plan"]["end"] + 1)):
            raise ValueError("Reconstruction history does not cover the retained anchor-to-end interval")
        from gate1_validation import validate_unit
        for step in range(anchor["step"] + 1, original["checkpoint_plan"]["end"] + 1):
            anchor_evidence.extend(validate_unit(source, step, expected_injected=False))
    else:
        model = physical(Path(original["overrides"]["actor_rollout_ref.model.path"]))
        if model_identity(model) != original["model"]:
            raise ValueError("Base-model reconstruction asset changed")
        reconstruction = {"kind": "base_model_seed_and_recorded_native_history", "path": str(model),
                          "model_signature": original["model"]["signature"], "step": 0}
        anchor_evidence = [evidence(member(model, name)) for name in original["model"]["files"]]
    reconstruction["protocol"] = {key: original.get(key) for key in ["phase", "injection_step", "reward_delta"]}
    reconstruction["protocol"]["branch"] = original["gate1"]["branch"]
    snapshots = {}
    for snapshot in sorted((run / "checkpoints").glob("global_step_*")):
        cert = certified_snapshot(snapshot, allow_retired=pending, pending=pending)
        if cert["run_id"] != run.name or cert["learning_signature"] != manifest["signatures"]["learning"]:
            raise ValueError("Temporary checkpoint belongs to another run/learning identity")
        snapshots[snapshot.name] = {"state_signature": cert["state_signature"],
            "certificate_sha256": file_digest(snapshot / "baseline_complete.json"),
            "retired_members": {name: info for name, info in cert["files"].items()
                if name.startswith(("actor/model_", "actor/optim_"))}}
    signatures = {item["state_signature"] for item in snapshots.values()}
    for other in (VERSION / "outputs").iterdir():
        if other == run or not (other / "manifest.json").is_file() or other.is_symlink():
            continue
        data = read_json(other / "manifest.json")
        parent = data.get("parent")
        if parent and parent.get("state_signature") in signatures:
            terminal = read_json(other / "status.json") if (other / "status.json").exists() else {}
            if (terminal.get("returncode") != 0 or terminal.get("status") not in {"completed", "intentionally_stopped"} or
                    not (other / "raw/temporary_weight_retirement_complete.json").exists()):
                raise ValueError("A dependent run still requires these recovery assets")
    receipt = {"run_id": run.name, "group_proof": str(group_proof.absolute()),
        "group_proof_sha256": file_digest(group_proof), "snapshots": snapshots,
        "reconstruction": reconstruction, "seed": manifest["seed"],
        "evidence": [*validation["evidence"], *anchor_evidence],
        "directly_restorable_after_retirement": False, "raw_and_full_fingerprints_retained": True}
    if pending and read_json(plan_path) != receipt:
        raise ValueError("Interrupted retirement plan or reconstruction evidence changed")
    immutable_json(plan_path, receipt)
    for name, snapshot in snapshots.items():
        directory = run / "checkpoints" / name
        for filename in snapshot["retired_members"]:
            p = member(directory, filename)
            if p.exists():
                p.unlink()
        fsync_directory(directory / "actor")
    immutable_json(complete_path, receipt)
    return receipt
