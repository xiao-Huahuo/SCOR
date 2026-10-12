"""Recheck native raw evidence and explicit recovery chains before acceptance."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from evaluate import aggregate
from gate1_evidence import evidence, finite, physical, verify_evidence
from pilot_utils import digest, file_digest, read_json
from records import signatures


def attempt_chain(run: Path, *, allow_retired: bool = False) -> list[Path]:
    chain = []
    while True:
        run = physical(run)
        if run in chain:
            raise ValueError("Recovery chain contains a cycle")
        data = read_json(run / "manifest.json")
        if data["run_id"] != run.name or data["signatures"] != signatures(data["learning"], data["protocol"], data.get("parent")):
            raise ValueError("Attempt owner or signed learning/protocol identity differs")
        chain.append(run)
        recovery = data.get("gate1", {}).get("recovery")
        if recovery is None:
            return list(reversed(chain))
        if data["protocol"].get("recovery") != recovery:
            raise ValueError("Recovery protocol and lineage differ")
        previous = physical(Path(recovery["previous_attempt"]))
        if file_digest(previous / "manifest.json") != recovery["previous_manifest_sha256"]:
            raise ValueError("Recovery predecessor manifest changed")
        before = read_json(previous / "manifest.json")
        for key in ["phase", "seed", "learning", "selected_gpus", "data_signature", "model"]:
            if before[key] != data[key]:
                raise ValueError("Recovery changed seed/model/data/learning/topology")
        for key in ["branch", "freeze_sha256", "probe_manifest_sha256", "import_manifest_sha256"]:
            if before["gate1"][key] != data["gate1"][key]:
                raise ValueError("Recovery changed branch/freeze/probe/source import")
        a, b = before.get("replay"), data.get("replay")
        if bool(a) != bool(b) or (a and any(a[k] != b[k] for k in ["source", "source_learning_signature"])):
            raise ValueError("Recovery changed its fixed history")
        checkpoint = physical(Path(recovery["checkpoint"]))
        if checkpoint.parent.parent != previous or data["parent"]["state_signature"] != recovery["state_signature"]:
            raise ValueError("Recovery parent belongs to another attempt")
        from gate1_state import certified_snapshot
        cert = certified_snapshot(checkpoint, allow_retired=allow_retired)
        boundary = recovery["completed_through"]
        completed = read_json(previous / f"raw/completed_step_{boundary:06d}.json") if boundary else None
        if (cert != data["parent"] or cert["step"] != boundary or cert["run_id"] != before["run_id"] or
                cert["state_signature"] != recovery["state_signature"] or
                data["checkpoint_plan"]["start"] != boundary or
                data["checkpoint_plan"]["end"] != before["checkpoint_plan"]["end"] or
                (completed and completed.get("model_snapshot_committed") is not True)):
            raise ValueError("Recovery checkpoint lacks its actual committed-unit boundary")
        if data.get("injection_step") is not None and data["injection_step"] <= boundary:
            raise ValueError("Recovery repeats an already committed intervention")
        run = previous


def record_run(run: Path, step: int, *, allow_retired: bool = False) -> Path:
    for attempt in attempt_chain(run, allow_retired=allow_retired):
        data = read_json(attempt / "manifest.json")
        recovery = data.get("gate1", {}).get("recovery")
        # The previous committed score/update stays authoritative at the boundary.
        if (recovery is None or step > recovery["completed_through"] or
                (step == 0 and not all((selected / f"raw/{stem}_step_000000.json").exists()
                                      for stem in ["gate1_scores", "evaluation"]) and
                 data["checkpoint_plan"]["start"] == 0)):
            selected = attempt
    return selected


def initial_metadata_path(run: Path) -> Path:
    from gate1_state import certificate_metadata
    for attempt in attempt_chain(run, allow_retired=True):
        for path in [attempt / "raw/initial_state/global_step_0", attempt / "checkpoints/global_step_0"]:
            if (path / "baseline_complete.json").exists():
                certificate_metadata(path)
                return path
    raise ValueError("No certified S0 metadata was retained")


def legacy_uniform_masks(run: Path, directory: Path, probe: dict) -> dict:
    """Verify the actual old constant-candidate masks; only historical engineering controls."""
    import torch
    from records import prompt_hash
    data = read_json(run / "manifest.json")
    if (data["phase"] != "control" or data["gate1"]["branch"] != "control" or
            data["gate1"]["freeze_sha256"] is not None or
            len(probe.get("solver_candidates", [])) != 2 or
            {re.sub(r"\s+", "", text) for text in probe["solver_candidates"]} != {"Finalanswer:0", "Finalanswer:1"} or
            probe["source_sha256"] != data["learning"]["source_sha256"]["gate1_probes.py"] or
            file_digest(run / "raw/source/gate1_probes.py") != probe["source_sha256"]):
        raise ValueError("Missing candidate masks are allowed only for the recorded old uniform engineering control")
    batches = torch.load(directory / "fixed_inputs.pt", map_location="cpu", weights_only=False)
    if digest([(batch["input_hash"], batch["cases"]) for batch in batches]) != digest(probe["batch_identities"]):
        raise ValueError("Legacy cached cases differ from their signed identities")
    masks = {}
    for batch in batches:
        actual = batch["data"]
        response = actual.batch["responses"]
        mask = actual.batch["multiturn_mask"]
        if (prompt_hash(actual) != batch["input_hash"] or response.ndim != 2 or not response.shape[1] or
                mask.shape != response.shape or not mask.bool().all() or
                not actual.batch["attention_mask"][:, -response.shape[1]:].bool().all() or
                not torch.equal(response, response[0].expand_as(response)) or len(batch["cases"]) != len(actual)):
            raise ValueError("Legacy masks contain padding or nonuniform candidate rows")
        for row in batch["cases"]:
            key = (row["case_id"], row["role"], row["candidate_id"])
            if (row["role"] not in {"solver", "verifier"} or type(row["candidate_id"]) is not int or
                    row["candidate_id"] not in {0, 1} or "candidate_mask" in row or key in masks or row["candidate"] != probe[
                    "solver_candidates" if row["role"] == "solver" else "verifier_candidates"][row["candidate_id"]]):
                raise ValueError("Legacy uniform candidate schema/identity differs")
            masks[key] = response.shape[1]
    return masks


def validate_probe(run: Path, step: int) -> list[dict]:
    data = read_json(run / "manifest.json")
    directory = physical(Path(data["gate1"]["probe_directory"]))
    probe = read_json(directory / "manifest.json")
    unsigned = {k: v for k, v in probe.items() if k not in {"probe_signature", "payload_sha256"}}
    if (digest(unsigned) != probe["probe_signature"] or
            file_digest(directory / "manifest.json") != data["gate1"]["probe_manifest_sha256"] or
            file_digest(directory / "fixed_inputs.pt") != probe["payload_sha256"]):
        raise ValueError("Probe input manifest/payload changed")
    score_path = run / f"raw/gate1_scores_step_{step:06d}.json"
    score = read_json(score_path)
    expected = [item for _, rows in probe["batch_identities"] for item in rows]
    legacy = legacy_uniform_masks(run, directory, probe) if any("candidate_mask" not in row for row in expected) else None
    if (score["step"] != step or score["split"] != probe["split"] or
            score["probe_signature"] != probe["probe_signature"] or
            [{k: v for k, v in row.items() if k != "token_logp"} for row in score["cases"]] != expected):
        raise ValueError("Scores do not cover the complete fixed probe identities")
    for row in score["cases"]:
        values = row["token_logp"]
        count = legacy[(row["case_id"], row["role"], row["candidate_id"])] if legacy is not None else sum(row["candidate_mask"])
        if (not values or not finite(values) or len(values) != count or
                any(type(v) not in (int, float) for v in values)):
            raise ValueError("Probe has missing/nonfinite/padded token scores")
    evaluation_path = run / f"raw/evaluation_step_{step:06d}.json"
    evaluation = read_json(evaluation_path)
    summary = aggregate(evaluation["cases"])
    split = data["evaluation_split"]
    dataset_path = physical(Path(data["dataset_directory"]) / "manifest.json")
    dataset = read_json(dataset_path)
    if (dataset["data_signature"] != data["data_signature"] or
            digest({k: v for k, v in dataset.items() if k != "data_signature"}) != dataset["data_signature"]):
        raise ValueError("Evaluation dataset identity differs")
    expected_identity = {row["case_id"]: (row["image_sha256"], int(row["answer"]), row["family"], row["scene_id"])
                         for row in dataset["samples"] if row["split"] == split}
    actual_identity = {row["case_id"]: (row["image_sha256"], row["ground_truth"], row["family"], row["scene_id"])
                       for row in evaluation["cases"]}
    expected_cases = 256 if split == "evaluation" else 64
    if (evaluation["step"] != step or evaluation["split"] != split or len(evaluation["cases"]) != expected_cases or
            actual_identity != expected_identity or evaluation["summary"] != summary or not finite(evaluation) or
            any(evaluation.get(key) is not True for key in ["worker_rng_restored", "driver_rng_restored", "data_cursor_unchanged"]) or
            any(evaluation[f"{key}_before"] != evaluation[f"{key}_after"] for key in ["worker_rng", "driver_rng", "data_cursor"]) or
            any(row["correct"] != int(row["explicit_answer"] is not None and row["explicit_answer"] == row["ground_truth"])
                for row in evaluation["cases"])):
        raise ValueError("Held-out cases/summary or evaluation RNG isolation differs")
    return [evidence(score_path), evidence(evaluation_path), evidence(directory / "manifest.json"),
            evidence(directory / "fixed_inputs.pt"), evidence(dataset_path), *([evidence(run / "raw/source/gate1_probes.py")]
                if legacy is not None else [])]


def validate_unit(run: Path, step: int, *, expected_injected: bool | None = None) -> list[dict]:
    import torch
    from baseline_acceptance import role_masks_match
    from gate1_contract import fixed_unit_audit, historical_output_matches
    from records import prompt_hash
    manifest = read_json(run / "manifest.json")
    stems = ["completed", "metrics", "group", "reward"]
    paths = [run / f"raw/{stem}_step_{step:06d}.json" for stem in stems]
    completed, metrics, group, reward = (read_json(p) for p in paths)
    ids = group["group_ids"]
    if (any(x["step"] != step for x in [completed, group, reward]) or
            completed["learning_signature"] != manifest["signatures"]["learning"] or
            completed["group_sha256"] != file_digest(paths[2]) or group["groups"] != 4 or group["trajectories"] != 16 or
            len(ids) != 16 or sorted(Counter(ids).values()) != [4] * 4 or
            group["trainable_tokens"] <= 0 or metrics["baseline/optimizer_step"] != step or
            metrics["baseline/parameter_probe_finite"] != 1. or not finite(metrics) or not finite(reward)):
        raise ValueError("Unit/group/optimizer/finite-state evidence differs")
    group_path = run / f"raw/training_group_step_{step:06d}.pt"
    generation_path = run / f"raw/generation_step_{step:06d}.pt"
    actual = torch.load(group_path, map_location="cpu", weights_only=False)
    generated = torch.load(generation_path, map_location="cpu", weights_only=False)
    if (prompt_hash(actual) != group["payload_hash"] or len(actual) != 16 or
            actual.non_tensor_batch["uid"].tolist() != ids or
            generated["step"] != step or generated["learning_signature"] != manifest["signatures"]["learning"] or
            not role_masks_match(actual) or
            not all(torch.isfinite(v).all() for v in actual.batch.values())):
        raise ValueError("Recorded native tensor/group/role content differs")
    historical_output_matches(generated["output"], actual)
    used = actual.batch["token_level_scores"].sum(-1).tolist()
    if used != reward["used_returns"] or len(reward["native_returns"]) != 16 or len(reward["native_info"]["acc"]) != 16:
        raise ValueError("Recorded reward differs from actual training scores")
    injected = reward["revision"]["status"] == "applied"
    if expected_injected is not None and injected != expected_injected:
        raise ValueError("Missing or unexpected intervention")
    replay = manifest.get("replay")
    if replay and replay["start"] <= step <= replay["end"]:
        audit_path = run / f"raw/gate1_unit_step_{step:06d}.json"
        audit = fixed_unit_audit(actual, physical(Path(replay["source"])), step,
                                 replay["source_learning_signature"], injected=injected)
        if read_json(audit_path) != audit:
            raise ValueError("Recorded fixed-unit audit differs from recomputed native evidence")
        paths.append(audit_path)
        source = physical(Path(replay["source"]))
        paths.extend(source / f"raw/{stem}_step_{step:06d}.{suffix}" for stem, suffix in [
            ("training_group", "pt"), ("group", "json"), ("completed", "json"), ("reward", "json")])
        native = read_json(source / f"raw/reward_step_{step:06d}.json")
        if reward["native_returns"] != native["native_returns"] or reward["native_info"] != native["native_info"]:
            raise ValueError("Native objective reward changed during fixed replay")
    elif reward["used_returns"] != reward["native_returns"]:
        raise ValueError("Online source unexpectedly changed its objective reward")
    return [evidence(p) for p in [*paths, group_path, generation_path]]


def validate_run(run: Path, *, allow_retired: bool = False, pending_retirement: bool = False) -> dict:
    from gate1_state import certificate_metadata, certified_snapshot, state_fingerprint
    chain = attempt_chain(run, allow_retired=allow_retired)
    manifest = read_json(run / "manifest.json")
    origin = read_json(chain[0] / "manifest.json")
    status = read_json(run / "status.json")
    if status.get("returncode") != 0 or status.get("status") not in {"completed", "intentionally_stopped"}:
        raise ValueError("Run did not terminate successfully")
    evidence_rows = []
    for attempt in chain:
        data = read_json(attempt / "manifest.json")
        evidence_rows.append(evidence(attempt / "manifest.json"))
        for name, sha in data["learning"]["source_sha256"].items():
            p = attempt / "raw/source" / name
            if evidence(p)["sha256"] != sha:
                raise ValueError("Recorded execution source changed")
            evidence_rows.append(evidence(p))
        evidence_rows.append(evidence(attempt / "raw/source/profile.json"))
        captured = attempt / "raw/initial_state/global_step_0/metadata_complete.json"
        if captured.exists():
            metadata = read_json(captured)
            verify_evidence(metadata["members"])
            zero = certificate_metadata(captured.parent)
            checkpoint_receipt = attempt / "raw/checkpoint_step_000000.json"
            if (metadata["run_id"] != attempt.name or zero["run_id"] != attempt.name or
                    zero["state_signature"] != metadata["state_signature"] or
                    (checkpoint_receipt.exists() and read_json(checkpoint_receipt)["state_signature"] != zero["state_signature"])):
                raise ValueError("Captured initial metadata differs from its actual native committed S0")
            evidence_rows.extend([evidence(captured), *metadata["members"]])
            if checkpoint_receipt.exists():
                evidence_rows.append(evidence(checkpoint_receipt))
        if data.get("parent"):
            path = attempt / "raw/resume_loaded.json"
            restored = read_json(path)
            parent = data["parent"]
            if (restored["step"] != parent["step"] or restored["parent_state"] != parent["state_signature"] or
                    any(restored.get(key) is not True for key in ["model_optimizer_scheduler_worker_rng_loaded",
                        "driver_rng_restored", "dataloader_loaded", "full_worker_state_fingerprints_match"])):
                raise ValueError("Actual full parent restore evidence is incomplete")
            parent_path = Path(data["overrides"]["trainer.resume_from_path"])
            if certificate_metadata(parent_path) != parent:
                raise ValueError("Actual restore path is not the declared complete parent state")
            fingerprint = state_fingerprint(parent_path, allow_retired=allow_retired)
            if (restored["driver_rng_sha256"] != fingerprint["driver_rng"] or
                    restored["dataloader_sha256"] != fingerprint["data_cursor"]):
                raise ValueError("Actual restored driver RNG/data cursor differs from the parent")
            evidence_rows.append(evidence(path))
    evidence_rows.append(evidence(run / "status.json"))
    start, end = origin["checkpoint_plan"]["start"], manifest["checkpoint_plan"]["end"]
    if start == 0:
        initial_metadata_path(run)
    for step in range(start + 1, end + 1):
        selected = record_run(run, step, allow_retired=allow_retired)
        data = read_json(selected / "manifest.json")
        injection = data.get("injection_step") == step and data.get("reward_delta") == 1
        evidence_rows.extend(validate_unit(selected, step, expected_injected=injection))
    evaluation_steps = sorted({start, end, *range(((start // 8) + 1) * 8, end + 1, 8)})
    for step in evaluation_steps:
        evidence_rows.extend(validate_probe(record_run(run, step, allow_retired=allow_retired), step))
    snapshot = run / f"checkpoints/global_step_{end}"
    cert = certified_snapshot(snapshot, allow_retired=allow_retired, pending=pending_retirement)
    completed = read_json(run / f"raw/completed_step_{end:06d}.json")
    if (cert["run_id"] != run.name or cert["learning_signature"] != manifest["signatures"]["learning"] or
            cert["parent"] != manifest["parent"] or
            completed["model_snapshot_committed"] is not True):
        raise ValueError("Terminal full state is not the actual committed end")
    evidence_rows.extend(evidence(snapshot / name) for name in [
        "baseline_complete.json", "baseline_state.json", "baseline_worker_state.json", "baseline_driver_rng.pt", "data.pt"])
    return {"passed": True, "run_id": run.name, "units": end - start, "evidence": evidence_rows,
            "learning_signature": manifest["signatures"]["learning"],
            "parent_state_signature": origin["parent"]["state_signature"] if origin.get("parent") else None,
            "replay": origin.get("replay"), "origin_manifest": origin, "attempts": [str(p) for p in chain]}


def validate_branch(run: Path, seed: int, branch: str, freeze_sha: str, *, allow_retired: bool = False) -> dict:
    validation = validate_run(run, allow_retired=allow_retired)
    manifest = validation["origin_manifest"]
    if (read_json(run / "status.json").get("status") != "completed" or manifest["phase"] != "formal" or
            manifest["seed"] != seed or manifest["gate1"]["branch"] != branch or
            manifest["gate1"]["freeze_sha256"] != freeze_sha or manifest["checkpoint_plan"]["start"] != 7 or
            manifest["checkpoint_plan"]["end"] != 32 or manifest["selected_gpus"] != [4, 5, 6, 7] or
            manifest["replay"]["start"] != 8 or manifest["replay"]["end"] != 32 or validation["units"] != 25 or
            manifest["injection_step"] != (8 if branch == "corrupt" else None)):
        raise ValueError("Branch terminal/protocol/seed/topology/intervention differs")
    return validation
