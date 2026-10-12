"""Execute the frozen five Gate1 blocks serially, resuming immutable attempts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from gate1_contract import score_distance
from gate1_evidence import SEEDS, evidence, immutable_json, member, physical, validate_freeze, verify_evidence
from gate1_references import compare_rebuilt, validate_reference
from gate1_state import certificate_metadata, certified_snapshot, compare_states, metadata_fingerprint, retire_prefix_initial, retire_temporary_weights
from gate1_validation import attempt_chain, initial_metadata_path, record_run, validate_branch, validate_probe, validate_run, validate_unit
from pilot_utils import ROOT, SRC, VERSION, atomic_json, digest, file_digest, read_json
from preflight import inspect
from records import fsync_directory

# Native DataProto unpickling in this parent needs the child's patched package.
sys.path.insert(1, str(ROOT / (".runtime/upstream/Agent0_baseline_" +
                             file_digest(SRC / "apply_baseline_instrumentation.py")[:12]) / "Agent0-VL"))


def event(sequence: Path, kind: str, **values):
    import json
    with (sequence / "raw/events.jsonl").open("a", encoding="utf-8") as stream:
        json.dump({"utc": datetime.now(timezone.utc).isoformat(), "kind": kind, **values}, stream)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def eligibility(source: Path):
    import torch
    from interventions import false_positive
    selected = record_run(source, 8) if ("gate1" in read_json(source / "manifest.json") and
                                       not (source / "history_provenance.json").exists()) else source
    group = torch.load(selected / "raw/training_group_step_000008.pt", map_location="cpu", weights_only=False)
    native = read_json(selected / "raw/reward_step_000008.json")["native_info"]["acc"]
    _, revision = false_positive(group, group.batch["token_level_scores"], native, step=8, alpha_out=1., delta=1)
    return revision


def _option(values: list[str], name: str, default=None):
    return values[values.index(name) + 1] if name in values else default


def _replace(values: list[str], name: str, value=None) -> list[str]:
    result = list(values)
    if name in result:
        i = result.index(name)
        del result[i:i + 2]
    if value is not None:
        result += [name, str(value)]
    return result


def last_committed(run: Path) -> Path | None:
    data = read_json(run / "manifest.json")
    found = []
    for path in (run / "checkpoints").glob("global_step_*"):
        try:
            cert = certified_snapshot(path)
            step = cert["step"]
            completed = read_json(run / f"raw/completed_step_{step:06d}.json") if step else None
            if (cert["run_id"] != run.name or cert["learning_signature"] != data["signatures"]["learning"] or
                    (step and (completed["model_snapshot_committed"] is not True or
                               completed["learning_signature"] != cert["learning_signature"]))):
                continue
            found.append((step, path))
        except (OSError, KeyError, ValueError, TypeError):
            continue
    if not found:
        return None
    step, path = max(found)
    for position in range(data["checkpoint_plan"]["start"] + 1, step + 1):
        validate_unit(run, position)
    return path


def _inactive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def capture_initial_metadata(run: Path) -> None:
    run = physical(run, VERSION / "outputs")
    source = physical(run / "checkpoints/global_step_0", run)
    target = physical(run / "raw/initial_state/global_step_0", run)
    if (target / "metadata_complete.json").exists() or not (source / "baseline_complete.json").is_file():
        return
    cert = certificate_metadata(source)
    target.mkdir(parents=True, exist_ok=True)
    names = ["baseline_complete.json", *[name for name in cert["files"]
             if not name.startswith(("actor/model_", "actor/optim_"))]]
    for name in names:
        destination = member(target, name)
        if destination.exists():
            if file_digest(destination) != file_digest(source / name):
                raise ValueError("Initial metadata capture changed")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source / name, destination)
    if (target / "actor").exists():
        fsync_directory(target / "actor")
    fsync_directory(target)
    fsync_directory(target.parent)
    fsync_directory(target.parent.parent)
    immutable_json(target / "metadata_complete.json", {"run_id": run.name,
        "state_signature": cert["state_signature"], "members": [evidence(target / name) for name in names],
        "metadata_only": True, "directly_restorable": False})


def execute(sequence: Path, name: str, base: list[str], extra: list[str], ids: list[int]) -> Path:
    plan_path = sequence / f"raw/stage_{name}_plan.json"
    ledger_path = sequence / f"raw/stage_{name}.json"
    accepted_path = sequence / f"raw/stage_{name}_accepted.json"
    immutable_json(plan_path, {"stage": name, "base": base, "extra": extra, "gpus": ids})
    ledger = read_json(ledger_path) if ledger_path.exists() else {"stage": name, "attempts": []}
    future = list(extra)
    if ledger["attempts"]:
        last = ledger["attempts"][-1]
        run = VERSION / "outputs" / last["run_id"]
        if last.get("pid") and not _inactive(last["pid"]):
            raise RuntimeError("Previous owned launcher is still running; do not launch another stage")
        if (run / "manifest.json").exists():
            data = read_json(run / "manifest.json")
            origin = read_json(attempt_chain(run)[0] / "manifest.json")
            if (origin["seed"] != int(_option(extra, "--seed")) or origin["phase"] != _option(extra, "--phase") or
                    origin["gate1"]["branch"] != _option(extra, "--branch")):
                raise ValueError("Recorded stage is not the requested seed/phase/branch")
            freeze_path = Path(_option(base, "--freeze"))
            frozen = read_json(freeze_path)
            if (origin["gate1"]["freeze_sha256"] != file_digest(freeze_path) or any(
                    origin["learning"][key] != frozen[frozen_key] for key, frozen_key in [
                        ("source_sha256", "source_sha256"), ("environment_signature", "environment_signature"),
                        ("model_signature", "model_signature"), ("dataset_signature", "dataset_signature")])):
                raise ValueError("Existing stage differs from frozen source/learning conditions")
            status = read_json(run / "status.json")
            if status.get("returncode") == 0 and status.get("status") in {"completed", "intentionally_stopped"}:
                receipts = [read_json(path) for suffix in ["plan", "complete"]
                            if (path := run / f"raw/temporary_weight_retirement_{suffix}.json").exists()]
                if receipts:
                    branch = origin["gate1"]["branch"]
                    proof_path = sequence / f"raw/{'clean_source' if branch == 'clean' else 'paired'}_seed_{origin['seed']}.json"
                    if len(receipts) == 2 and receipts[0] != receipts[1]:
                        raise ValueError("Retirement plan and completion differ")
                    for retirement in receipts:
                        anchor = retirement["reconstruction"]
                        if (branch not in {"clean", "repeat"} or origin["phase"] != "formal" or
                                retirement["run_id"] != run.name or retirement["seed"] != origin["seed"] or
                                physical(Path(retirement["group_proof"])) != physical(proof_path) or
                                anchor["kind"] != "retained_full_state_and_fixed_history" or
                                physical(Path(anchor["path"])) != physical(Path(_option(extra, "--resume-from")))):
                            raise ValueError("Recorded retirement is not this sequence's branch proof/common S7")
                    if branch == "clean":
                        proof = read_json(proof_path)
                        imported = read_json(Path(read_json(sequence / "manifest.json")["inputs"]["assets"]))
                        source = physical(Path(origin["replay"]["source"]))
                        if str(origin["seed"]) in imported["histories"]:
                            source_run = physical(Path(imported["histories"][str(origin["seed"])]["source"]))
                        else:
                            source_run = Path(read_json(source / "history_provenance.json")["attempts"][-1]) if (
                                source / "history_provenance.json").exists() else source
                        retained = proof["retained_source_snapshot"]
                        source_snapshot = source_run / "checkpoints/global_step_32"
                        source_cert = certified_snapshot(source_snapshot)
                        if (proof["kind"] != "clean_source_dedup" or proof["seed"] != origin["seed"] or
                                physical(Path(proof["source"])) != source or physical(Path(retained["path"])) != source_snapshot or
                                source_cert["state_signature"] != retained["state_signature"] or
                                file_digest(source_snapshot / "baseline_complete.json") != retained["certificate_sha256"]):
                            raise ValueError("Retained full source S32 differs from the pre-retirement clean acceptance")
                    retire_temporary_weights(run, proof_path, Path(receipts[0]["reconstruction"]["path"]))
                validated = validate_run(run, allow_retired=True)
                if accepted_path.exists():
                    accepted = read_json(accepted_path)
                    verify_evidence(accepted["validation"]["evidence"])
                    if accepted != {"run_id": run.name, "validation": validated}:
                        raise ValueError("Accepted stage evidence changed")
                else:
                    immutable_json(accepted_path, {"run_id": run.name, "validation": validated})
                ledger["status"] = "completed"
                atomic_json(ledger_path, ledger)
                event(sequence, "stage_reused", run_id=run.name)
                return run
            if status.get("status") == "running":
                immutable_json(run / "raw/workflow_interruption.json", {"previous_status": status,
                    "launch": last, "reason": "owned launcher is no longer alive"})
                atomic_json(run / "status.json", {"status": "interrupted", "previous_status": status})
            snapshot = last_committed(run)
            if snapshot is not None:
                boundary = certified_snapshot(snapshot)["step"]
                if boundary == data["checkpoint_plan"]["end"]:
                    raise RuntimeError("End state exists without successful terminal receipt; preserve failed finalization")
                future = _replace(future, "--resume-from", snapshot)
                future = _replace(future, "--recovery-from", run)
                if data.get("replay"):
                    future = _replace(future, "--replay-start", boundary + 1)
                if _option(future, "--injection-step") and int(_option(future, "--injection-step")) <= boundary:
                    future = _replace(future, "--injection-step")
                event(sequence, "recovery_planned", previous_attempt=run.name, completed_through=boundary)
    number = len(ledger["attempts"]) + 1
    attempt_name = f"{name}_a{number:03d}"
    run = VERSION / "outputs" / attempt_name
    if run.exists() or run.is_symlink():
        raise FileExistsError("New attempt ID already exists")
    while True:
        resource = inspect(ROOT, ids, minimum_disk_gib=0)
        if not resource["problems"]:
            break
        if any(not item.startswith("GPU ") for item in resource["problems"]):
            raise RuntimeError(str(resource["problems"]))
        event(sequence, "resource_wait", run_id=attempt_name, problems=resource["problems"])
        time.sleep(30)
    command = [*base, "--run-id", attempt_name, *future]
    last = {"run_id": attempt_name, "argv": command}
    ledger["attempts"].append(last)
    ledger["status"] = "launch_planned"
    atomic_json(ledger_path, ledger)
    with (sequence / f"raw/{attempt_name}_launcher.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        last["pid"] = process.pid
        ledger["status"] = "running"
        atomic_json(ledger_path, ledger)
        event(sequence, "stage_started", **last)
        while not (run / "raw").is_dir() and process.poll() is None:
            time.sleep(1)
        if (run / "raw").is_dir():
            immutable_json(run / "raw/workflow_launch.json", last)
        while process.poll() is None:
            capture_initial_metadata(run)
            time.sleep(1)
        capture_initial_metadata(run)
        code = process.wait()
    if code != 0:
        ledger["status"] = "failed"
        last["returncode"] = code
        atomic_json(ledger_path, ledger)
        raise RuntimeError(f"Stage {attempt_name} failed with {code}; preserve its last complete recovery state")
    validated = validate_run(run)
    immutable_json(accepted_path, {"run_id": run.name, "validation": validated})
    ledger["status"] = "completed"
    atomic_json(ledger_path, ledger)
    event(sequence, "stage_completed", run_id=attempt_name)
    return run


def materialize_history(sequence: Path, source: Path, seed: int) -> Path:
    """A recovered online source exposes committed original units without recomputing them."""
    if len(attempt_chain(source)) == 1:
        return source
    target = sequence / f"histories/seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for relative in ["manifest.json", "status.json", *[
            f"raw/{stem}_step_{step:06d}.{suffix}" for step in range(1, 33)
            for stem, suffix in [("generation", "pt"), ("training_group", "pt"), ("completed", "json"),
                                 ("group", "json"), ("reward", "json"), ("metrics", "json")]]]:
        step = int(re.search(r"step_(\d+)", relative).group(1)) if "step_" in relative else None
        actual = (record_run(source, step) if step else source) / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_digest(destination) != file_digest(actual):
                raise ValueError("Consolidated source history changed")
        else:
            os.link(actual, destination)
        rows.append({"target": relative, **evidence(actual)})
    immutable_json(target / "history_provenance.json", {"seed": seed, "attempts": [str(p) for p in attempt_chain(source)],
        "files": rows, "scope": "committed source units linked from their actual attempts; no model restore claim"})
    return target


def validate_learning(sequence: Path, seed: int, validations: dict) -> None:
    from gate1_contract import compatible_learning
    sequence_manifest = read_json(sequence / "manifest.json")
    frozen = read_json(Path(sequence_manifest["freeze_path"]))
    if file_digest(Path(sequence_manifest["freeze_path"])) != sequence_manifest["freeze_sha256"]:
        raise ValueError("Sequence's frozen learning evidence changed")
    imported = read_json(Path(sequence_manifest["inputs"]["assets"]))
    template = dict(imported["canonical_learning"])
    template["seed"] = seed
    for value in validations.values():
        learning = value["origin_manifest"]["learning"]
        compatible_learning(learning, template, imported["canonical_source_sha256"])
        if any(learning[key] != frozen[frozen_key] for key, frozen_key in [
                ("source_sha256", "source_sha256"), ("environment_signature", "environment_signature"),
                ("model_signature", "model_signature"), ("dataset_signature", "dataset_signature")]):
            raise ValueError("Paired learning conditions differ from the accepted freeze")


def clean_source_proof(sequence: Path, seed: int, source: Path, prefix: Path, clean: Path,
                       freeze_sha: str, validation: dict, reference: Path | None = None) -> dict:
    """Accept clean's physical weight dependencies before later S7-based branches."""
    early_path = sequence / f"raw/clean_source_seed_{seed}.json"
    for suffix in ["plan", "complete"]:
        retirement_path = clean / f"raw/temporary_weight_retirement_{suffix}.json"
        if retirement_path.exists():
            receipt = read_json(retirement_path)
            if (not early_path.is_file() or physical(Path(receipt["group_proof"])) != physical(early_path) or
                    receipt["group_proof_sha256"] != file_digest(early_path)):
                raise ValueError("Retired clean lacks this sequence's original pre-retirement acceptance")
    validate_learning(sequence, seed, {"clean": validation})
    anchor = certified_snapshot(prefix)
    source_manifest = read_json(source / "manifest.json")
    sequence_manifest = read_json(sequence / "manifest.json")
    if (anchor["step"] != 7 or anchor["state_signature"] != validation["parent_state_signature"] or
            physical(Path(validation["replay"]["source"])) != source or source_manifest["seed"] != seed or
            sequence_manifest["freeze_sha256"] != freeze_sha):
        raise ValueError("Clean's source/common S7/freeze differs")
    snapshot = clean / "checkpoints/global_step_32"
    rows = [*validation["evidence"], evidence(source / "manifest.json"),
            evidence(prefix / "baseline_complete.json"), evidence(sequence / "manifest.json"),
            evidence(Path(sequence_manifest["freeze_path"]))]
    if reference:
        retained_source = validate_reference(reference)["entry"]
        source_snapshot = Path(retained_source["source"])
        full_source = certified_snapshot(source_snapshot)
        if full_source["state_signature"] != retained_source["state_signature"]:
            raise ValueError("Historical source S32 is not the preserved full state")
        source_state = compare_rebuilt(snapshot, reference, allow_retired=True)
        rows += [*source_state["evidence"], evidence(Path(retained_source["source"]) / "baseline_complete.json")]
    else:
        actual_source = Path(read_json(source / "history_provenance.json")["attempts"][-1]) if (
            source / "history_provenance.json").exists() else source
        source_snapshot = actual_source / "checkpoints/global_step_32"
        full_source = certified_snapshot(source_snapshot)
        source_state = compare_states(snapshot, source_snapshot, allow_retired=True)
        rows.append(evidence(source_snapshot / "baseline_complete.json"))
    if not source_state["passed"]:
        raise ValueError("Clean-fixed S32 differs from the preserved source; retain all weights and re-budget")
    return {"passed": True, "kind": "clean_source_dedup", "seed": seed, "source": str(source), "prefix": str(prefix),
            "freeze_sha256": freeze_sha, "accepted_run_ids": [clean.name], "pending_dependents": [],
            "dependency_scope": "physical clean weights only; later branches restore common S7 and use retained history/scores",
            "retained_source_snapshot": {"path": str(source_snapshot), "state_signature": full_source["state_signature"],
                                         "certificate_sha256": file_digest(source_snapshot / "baseline_complete.json")},
            "clean_source_state": source_state, "branch_unit_validation": validation, "evidence": rows}


def paired_proof(sequence: Path, seed: int, source: Path, prefix: Path, branches: dict,
                 revision: dict, freeze_sha: str, reference: Path | None = None) -> dict:
    validations = {branch: validate_branch(run, seed, branch, freeze_sha, allow_retired=True)
                   for branch, run in branches.items()}
    validate_learning(sequence, seed, validations)
    identities = [(x["learning_signature"], x["parent_state_signature"], x["replay"]) for x in validations.values()]
    if any(x != identities[0] for x in identities[1:]):
        raise ValueError("Paired learning, parent or fixed history differs")
    source_manifest = read_json(source / "manifest.json")
    if source_manifest["seed"] != seed or physical(Path(identities[0][2]["source"])) != source:
        raise ValueError("Paired source training seed/path differs")
    anchor = certified_snapshot(prefix)
    if anchor["step"] != 7 or anchor["state_signature"] != identities[0][1]:
        raise ValueError("Paired common full S7 anchor differs")
    actual_revision = read_json(record_run(branches["corrupt"], 8) / "raw/reward_step_000008.json")["revision"]
    if actual_revision != revision or eligibility(source) != revision:
        raise ValueError("Predeclared first eligible U8 target differs")
    rows = [item for value in validations.values() for item in value["evidence"]]
    rows += [evidence(source / "manifest.json"), evidence(prefix / "baseline_complete.json"),
             evidence(sequence / f"raw/eligibility_{seed}.json")]
    initial = initial_point(sequence, seed, source, prefix)
    rows += initial["evidence"]
    curve = [initial["curve"]]
    for step in [8, 16, 24, 32]:
        scores = {branch: read_json(record_run(run, step) / f"raw/gate1_scores_step_{step:06d}.json")
                  for branch, run in branches.items()}
        difference = score_distance(scores["clean"], scores["corrupt"])
        floor = score_distance(scores["clean"], scores["repeat"])
        evaluations = [read_json(record_run(run, step) / f"raw/evaluation_step_{step:06d}.json") for run in branches.values()]
        case_ids = [{row["case_id"]: (row["image_sha256"], row["ground_truth"], row["family"])
                    for row in result["cases"]} for result in evaluations]
        if any(x != case_ids[0] for x in case_ids[1:]):
            raise ValueError("Paired held-out case/image/answer identities differ")
        curve.append({"step": step, "D_score": difference["D_score"], "floor": floor["D_score"],
                      "corrected_D": difference["D_score"] - floor["D_score"], "roles": difference["roles"]})
    clean_snapshot = branches["clean"] / "checkpoints/global_step_32"
    early_path = sequence / f"raw/clean_source_seed_{seed}.json"
    early = clean_source_proof(sequence, seed, source, prefix, branches["clean"], freeze_sha,
                               validations["clean"], reference)
    stored_early = read_json(early_path)
    verify_evidence(stored_early["evidence"])
    if stored_early != early:
        raise ValueError("Early clean/source acceptance differs from rebuilt actual evidence")
    clean_retirement_path = branches["clean"] / "raw/temporary_weight_retirement_complete.json"
    clean_retirement = read_json(clean_retirement_path)
    if (physical(Path(clean_retirement["group_proof"])) != physical(early_path) or
            clean_retirement["group_proof_sha256"] != file_digest(early_path)):
        raise ValueError("Clean retirement is not bound to its early acceptance")
    rows += [*early["evidence"], evidence(early_path), evidence(clean_retirement_path)]
    repeat_state = compare_states(clean_snapshot, branches["repeat"] / "checkpoints/global_step_32", allow_retired=True)
    return {"passed": True, "seed": seed, "source": str(source), "prefix": str(prefix),
            "accepted_run_ids": [p.name for p in branches.values()], "pending_dependents": [],
            "curve": curve, "repeat_full_state": repeat_state, "clean_source_state": early["clean_source_state"],
            "revision": revision, "evidence": rows, "branch_unit_validations": validations}


def initial_point(sequence: Path, seed: int, source: Path, prefix: Path) -> dict:
    manifest = read_json(sequence / "manifest.json")
    freeze = read_json(Path(manifest["freeze_path"]))
    control = freeze["initial_state_control"]
    control_run = physical(Path(control["run"]))
    control_manifest = read_json(control_run / "manifest.json")
    if (control_manifest["seed"] != SEEDS[0] or control_manifest["learning"]["source_sha256"] != freeze["source_sha256"] or
            control_manifest["model"]["signature"] != freeze["model_signature"] or
            read_json(control_run / "status.json").get("returncode") != 0 or
            physical(Path(control["scores"]["path"])) != control_run / "raw/gate1_scores_step_000000.json" or
            physical(Path(control["evaluation"]["path"])) != control_run / "raw/evaluation_step_000000.json"):
        raise ValueError("Initial control is not the frozen current-source base model")
    verify_evidence([control["scores"], control["evaluation"]])
    control_state = metadata_fingerprint(physical(Path(control["snapshot"])))
    if control_state["step"] != 0 or certificate_metadata(Path(control["snapshot"]))["run_id"] != control_run.name:
        raise ValueError("Initial control metadata is not S0")
    rows = validate_probe(control_run, 0)
    reference_directory = Path(manifest["inputs"]["references"])
    references = read_json(reference_directory / "manifest.json")
    canonical_zero = validate_reference(Path(references["references"][f"{SEEDS[0]}:0"]["path"]))
    if control_state != metadata_fingerprint(Path(canonical_zero["entry"]["path"])):
        raise ValueError("Current initial control differs from historical certified seed11 S0")
    rows += [*canonical_zero["evidence"], evidence(control_run / "manifest.json"), control["scores"], control["evaluation"],
            evidence(Path(control["snapshot"]) / "baseline_complete.json")]
    if seed == SEEDS[0]:
        score_path = physical(Path(control["scores"]["path"]))
        eval_path = physical(Path(control["evaluation"]["path"]))
        actual_state = control_state
        initial_run = control_run
    elif seed in SEEDS[:3]:
        run = prefix.parent.parent
        initial_run = record_run(run, 0)
        score_path = initial_run / "raw/gate1_scores_step_000000.json"
        actual_state = metadata_fingerprint(initial_metadata_path(run))
        initial_evaluation = freeze["initial_evaluations"][str(seed)]
        verify_evidence([initial_evaluation])
        for key in ["source_manifest", "import_manifest"]:
            if key in initial_evaluation:
                verify_evidence([initial_evaluation[key]])
                rows.append(initial_evaluation[key])
        if "source_manifest" in initial_evaluation:
            origin = read_json(Path(initial_evaluation["source_manifest"]["path"]))
            if origin["seed"] != seed or origin["data_signature"] != freeze["dataset_signature"]:
                raise ValueError("Imported initial evaluation source identity differs")
        eval_path = physical(Path(initial_evaluation["path"]))
        reference = validate_reference(Path(references["references"][f"{seed}:0"]["path"]))
        reference_state = metadata_fingerprint(Path(reference["entry"]["path"]))
        if actual_state != reference_state:
            raise ValueError("Rebuilt S0 differs from its seed's historical certified initial state")
        rows += reference["evidence"]
    else:
        actual_source = Path(read_json(source / "history_provenance.json")["attempts"][-1]) if (
            source / "history_provenance.json").exists() else source
        initial_run = record_run(actual_source, 0)
        score_path = initial_run / "raw/gate1_scores_step_000000.json"
        eval_path = initial_run / "raw/evaluation_step_000000.json"
        actual_state = metadata_fingerprint(initial_metadata_path(actual_source))
    model_fields = ["model", "buffers", "optimizer", "optimizer_groups", "scheduler"]
    if (actual_state["step"] != 0 or [[worker[key] for key in model_fields] for worker in actual_state["workers"]] !=
            [[worker[key] for key in model_fields] for worker in control_state["workers"]]):
        raise ValueError("Seed S0 parameters/Adam/scheduler differ from the shared pretrained model")
    scores, evaluation = read_json(score_path), read_json(eval_path)
    rows += validate_probe(initial_run, 0)
    from evaluate import aggregate
    dataset_path = Path(manifest["inputs"]["assets"]).parent / "visual_baseline_08/manifest.json"
    dataset = read_json(dataset_path)
    expected_cases = {row["case_id"]: (row["image_sha256"], int(row["answer"]), row["family"], row["scene_id"])
                      for row in dataset["samples"] if row["split"] == "evaluation"}
    actual_cases = {row["case_id"]: (row["image_sha256"], row["ground_truth"], row["family"], row["scene_id"])
                    for row in evaluation["cases"]}
    if (actual_cases != expected_cases or evaluation["summary"] != aggregate(evaluation["cases"]) or
            dataset["data_signature"] != freeze["dataset_signature"] or
            any(evaluation.get(key) is not True for key in ["worker_rng_restored", "driver_rng_restored", "data_cursor_unchanged"])):
        raise ValueError("Initial held-out P256 cases/checker/RNG evidence differs from the frozen dataset")
    rows.append(evidence(dataset_path))
    if (scores["step"] != 0 or evaluation["step"] != 0 or evaluation["split"] != "evaluation" or
            len(evaluation["cases"]) != 256 or scores["probe_signature"] != read_json(
                Path(manifest["inputs"]["probe_set"]) / "manifest.json")["probe_signature"]):
        raise ValueError("Initial score/held-out256 witness differs from the frozen protocol")
    distance = score_distance(scores, scores)
    floor = score_distance(scores, scores)
    rows += [evidence(score_path), evidence(eval_path)]
    return {"curve": {"step": 0, "D_score": distance["D_score"], "floor": floor["D_score"],
        "corrected_D": distance["D_score"] - floor["D_score"], "roles": distance["roles"],
        "provenance": "same actually scored common S0 for all paths before the S7 fork",
        "scores": str(score_path), "evaluation": str(eval_path)}, "evidence": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--resume", action="store_true")
    for option in ["freeze", "assets", "references", "model-path", "probe-set"]:
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=SRC / "configs/baseline_3b_candidate.json")
    parser.add_argument("--gpus", default="4,5,6,7")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", args.sequence_id):
        parser.error("Use a short physical sequence ID")
    freeze = validate_freeze(args.freeze)
    assets = read_json(physical(args.assets))
    references = read_json(physical(args.references / "manifest.json"))
    if (assets["status"] != "verified" or assets["manifest_signature"] != digest(
            {k: v for k, v in assets.items() if k != "manifest_signature"}) or
            references["import_manifest_sha256"] != file_digest(args.assets) or
            references["manifest_signature"] != digest({k: v for k, v in references.items() if k != "manifest_signature"}) or
            file_digest(args.profile) != freeze["profile_sha256"] or
            file_digest(args.probe_set / "manifest.json") != freeze["probe_manifest_sha256"]):
        raise ValueError("Frozen input/profile/probe/reference provenance differs")
    for name, info in assets["files"].items():
        if file_digest(physical(args.assets.parent / name)) != info["sha256"]:
            raise ValueError("Imported source data/history/state changed")
    ids = [int(x) for x in args.gpus.split(",")]
    if ids != [4, 5, 6, 7]:
        parser.error("Use the frozen four-card topology")
    sequence = physical(VERSION / "outputs" / args.sequence_id, VERSION / "outputs")
    manifest = {"sequence_id": args.sequence_id, "seeds": SEEDS, "freeze_sha256": file_digest(args.freeze),
        "freeze_path": str(args.freeze.absolute()), "gpus": ids, "horizon": 32, "k": 8,
        "inputs": {key: str(getattr(args, key).absolute()) for key in [
            "assets", "references", "model_path", "probe_set", "profile"]},
        "input_sha256": {"assets": file_digest(args.assets), "references": file_digest(args.references / "manifest.json")}}
    if sequence.exists() and not args.resume:
        raise FileExistsError("Use --resume to verify and reuse this sequence's immutable attempts")
    (sequence / "raw").mkdir(parents=True, exist_ok=args.resume)
    (sequence / "processed").mkdir(exist_ok=args.resume)
    immutable_json(sequence / "manifest.json", manifest)
    common = [sys.executable, str(SRC / "run_gate1.py"), "--model-path", str(args.model_path.absolute()),
        "--assets", str(args.assets.absolute()), "--data", str(args.assets.parent / "visual_baseline_08"),
        "--probe-set", str(args.probe_set.absolute()), "--profile", str(args.profile.absolute()),
        "--freeze", str(args.freeze.absolute()), "--gpus", args.gpus]
    results = []
    atomic_json(sequence / "status.json", {"status": "running", "completed_seed_blocks": results})
    try:
        for seed in SEEDS:
            seed_args = ["--seed", str(seed)]
            if str(seed) in assets["histories"]:
                source = physical(Path(assets["histories"][str(seed)]["path"]))
                source_proof = {"seed": seed, "source": str(source), "kind": "verified_import",
                                "import_manifest_sha256": file_digest(args.assets)}
            else:
                actual_source = execute(sequence, f"{args.sequence_id}_source_{seed}", common,
                    [*seed_args, "--phase", "preparation", "--evaluation-split", "evaluation", "--branch", "source", "--pin-step", "7"], ids)
                source = materialize_history(sequence, actual_source, seed)
                source_proof = {"seed": seed, "source": str(source), "actual_source": str(actual_source),
                    "kind": "accepted_preparation", "acceptance": str(sequence / f"raw/stage_{args.sequence_id}_source_{seed}_accepted.json")}
            immutable_json(sequence / f"raw/source_seed_{seed}.json", source_proof)
            revision = eligibility(source)
            immutable_json(sequence / f"raw/eligibility_{seed}.json", revision)
            if revision["status"] != "applied":
                results.append({"seed": seed, "valid": False, "status": "no_eligible_target", "source": str(source)})
                atomic_json(sequence / "status.json", {"status": "running", "completed_seed_blocks": results})
                continue
            if seed == SEEDS[0]:
                prefix = physical(Path(assets["states"]["seed_20261011_S7"]["path"]))
            elif str(seed) in assets["histories"]:
                rebuilt = execute(sequence, f"{args.sequence_id}_prefix_{seed}", common,
                    [*seed_args, "--phase", "preparation", "--branch", "prefix", "--stop-after", "7", "--pin-step", "7",
                     "--replay-from", str(source), "--replay-start", "1", "--replay-end", "7"], ids)
                prefix = rebuilt / "checkpoints/global_step_7"
            else:
                prefix = record_run(actual_source, 7) / "checkpoints/global_step_7"
            if str(seed) in assets["histories"]:
                reference = physical(Path(references["references"][f"{seed}:7"]["path"]))
                audit = compare_rebuilt(prefix, reference)
                if not audit["passed"]:
                    raise ValueError("Rebuilt/imported S7 differs from historically certified full metadata")
                immutable_json(sequence / f"raw/prefix_rebuild_{seed}.json", audit)
                if seed != SEEDS[0]:
                    retire_prefix_initial(prefix.parent.parent,
                        Path(references["references"][f"{seed}:0"]["path"]), reference)
            reference = physical(Path(references["references"][f"{seed}:32"]["path"])) if str(seed) in assets["histories"] else None
            branches = {}
            for branch in ["clean", "corrupt", "repeat"]:
                extra = [*seed_args, "--phase", "formal", "--branch", branch, "--resume-from", str(prefix),
                         "--replay-from", str(source), "--replay-start", "8", "--replay-end", "32"]
                if branch == "corrupt":
                    extra += ["--injection-step", "8", "--reward-delta", "1"]
                branches[branch] = execute(sequence, f"{args.sequence_id}_{branch}_{seed}", common, extra, ids)
                if branch == "clean":
                    early = clean_source_proof(sequence, seed, source, prefix, branches["clean"], file_digest(args.freeze),
                        validate_branch(branches["clean"], seed, "clean", file_digest(args.freeze), allow_retired=True), reference)
                    early_path = sequence / f"raw/clean_source_seed_{seed}.json"
                    immutable_json(early_path, early)
                    retire_temporary_weights(branches["clean"], early_path, prefix)
            proof = paired_proof(sequence, seed, source, prefix, branches, revision, file_digest(args.freeze), reference)
            proof_path = sequence / f"raw/paired_seed_{seed}.json"
            immutable_json(proof_path, proof)
            retire_temporary_weights(branches["repeat"], proof_path, prefix)
            results.append({"seed": seed, "valid": True, "source": str(source), "prefix": str(prefix),
                            "branches": {key: str(value) for key, value in branches.items()}, "proof": str(proof_path)})
            atomic_json(sequence / "status.json", {"status": "running", "completed_seed_blocks": results})
        atomic_json(sequence / "status.json", {"status": "completed", "completed_seed_blocks": results})
        return 0
    except BaseException as error:
        atomic_json(sequence / "status.json", {"status": "failed", "completed_seed_blocks": results,
                    "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
