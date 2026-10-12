"""Adversarial workflow evidence and small-file retirement tests; no model/GPU allocation."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gate1_evidence as guard
import gate1_state as state
import gate1_validation as validation
import run_gate1 as launcher
import run_gate1_sequence as workflow
from checkpoint_store import certify, required_names, validate
from evaluate import aggregate
from pilot_utils import atomic_json, digest, file_digest, read_json
from records import model_identity, signatures


@pytest.fixture
def owned(tmp_path, monkeypatch):
    version = tmp_path / "solutions/v02"
    (version / "outputs").mkdir(parents=True)
    (version / "inputs").mkdir()
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(state, "VERSION", version)
    monkeypatch.setattr(launcher, "VERSION", version)
    monkeypatch.setattr(workflow, "VERSION", version)
    return version


def snapshot(run, step, learning="learning", parent=None):
    path = run / f"checkpoints/global_step_{step}"
    path.mkdir(parents=True)
    for name in required_names(4):
        p = path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((name + "-payload").encode("utf-8"))
    atomic_json(path / "baseline_state.json", {"completed_step": step, "next_step": step + 1,
        "full_horizon": 32, "zero_signal_streak": 0, "parent": parent})
    workers = [{"rank": rank, "model": {"w": "a" * 64}, "buffers": {}, "optimizer": {"w": "b" * 64},
                "optimizer_groups": [{"lr": 5e-7}], "scheduler": {"last_epoch": step}, "rng_hash": "c" * 64,
                "all_parameters_and_optimizer_finite": True} for rank in range(4)]
    atomic_json(path / "baseline_worker_state.json", workers)
    return path, certify(path, learning=learning, run_id=run.name, step=step, world=4, parent=parent)


@pytest.mark.parametrize("damage", ["traversal", "absolute", "symlink", "ancestor", "missing_model", "worker_rank", "nonfinite"])
def test_certificate_cannot_restore_missing_or_escaping_state(owned, damage):
    run = owned / "outputs/control"
    path, cert = snapshot(run, 8)
    target = path / "actor/model_world_size_4_rank_0.pt"
    if damage == "missing_model":
        target.unlink()
    elif damage == "symlink":
        external = owned / "inputs/keep.pt"
        external.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(external)
    elif damage == "ancestor":
        actor = path / "actor"
        actor.rename(path / "physical_actor")
        actor.symlink_to(path / "physical_actor")
    elif damage in {"worker_rank", "nonfinite"}:
        workers = read_json(path / "baseline_worker_state.json")
        if damage == "worker_rank":
            workers[1]["rank"] = 0
        else:
            workers[0]["all_parameters_and_optimizer_finite"] = False
        atomic_json(path / "baseline_worker_state.json", workers)
        cert["files"]["baseline_worker_state.json"] = {"bytes": (path / "baseline_worker_state.json").stat().st_size,
            "sha256": file_digest(path / "baseline_worker_state.json")}
    else:
        external = owned / "inputs/keep.pt"
        external.write_bytes(b"must not delete")
        name = "../../inputs/keep.pt" if damage == "traversal" else str(external)
        cert["files"][name] = {"bytes": external.stat().st_size, "sha256": file_digest(external)}
    if damage in {"traversal", "absolute", "worker_rank", "nonfinite"}:
        cert = {k: v for k, v in cert.items() if k not in {"state_signature", "_verified_metadata"}}
        cert["state_signature"] = digest(cert)
        atomic_json(path / "baseline_complete.json", cert)
    with pytest.raises((ValueError, FileNotFoundError)):
        state.certified_snapshot(path)
    if (owned / "inputs/keep.pt").exists():
        assert (owned / "inputs/keep.pt").read_bytes() in {b"must not delete", b"actor/model_world_size_4_rank_0.pt-payload"}


def recovery_pair(owned):
    learning = {"seed": 20261011, "source_sha256": {}}
    previous = owned / "outputs/corrupt_a001"
    parent, cert = snapshot(previous, 8, digest(learning))
    cert = {k: v for k, v in cert.items() if k != "_verified_metadata"}
    common = {"seed": 20261011, "phase": "formal", "learning": learning, "selected_gpus": [4, 5, 6, 7],
              "data_signature": "data", "model": {"signature": "model"},
              "replay": {"source": str(owned / "inputs/history"), "start": 8, "end": 32, "source_learning_signature": "legacy"}}
    gate = {"branch": "corrupt", "freeze_sha256": "freeze", "probe_manifest_sha256": "probe",
            "import_manifest_sha256": "import"}
    before = {**common, "run_id": previous.name, "parent": None, "protocol": {"phase": "formal", "injection_step": 8},
              "injection_step": 8, "reward_delta": 1, "gate1": gate, "checkpoint_plan": {"start": 7, "end": 32}}
    before["signatures"] = signatures(learning, before["protocol"], None)
    atomic_json(previous / "manifest.json", before)
    atomic_json(previous / "status.json", {"status": "failed", "returncode": -9})
    atomic_json(previous / "raw/completed_step_000008.json", {"step": 8, "learning_signature": digest(learning),
                "model_snapshot_committed": True})
    recovery = {"previous_attempt": str(previous), "previous_manifest_sha256": file_digest(previous / "manifest.json"),
        "checkpoint": str(parent), "state_signature": cert["state_signature"], "completed_through": 8,
        "origin_injection_step": 8, "origin_reward_delta": 1}
    current = owned / "outputs/corrupt_a002"
    current.mkdir()
    after = {**common, "run_id": current.name, "parent": cert, "protocol": {"phase": "formal", "recovery": recovery},
             "injection_step": None, "reward_delta": 1, "gate1": {**gate, "recovery": recovery},
             "checkpoint_plan": {"start": 8, "end": 32},
             "replay": {**common["replay"], "start": 9}}
    after["signatures"] = signatures(learning, after["protocol"], cert)
    atomic_json(current / "manifest.json", after)
    return previous, current, parent


def test_recovery_routes_committed_U8_to_original_and_future_to_new_attempt(owned):
    before, after, _ = recovery_pair(owned)
    assert validation.record_run(after, 8) == before
    assert validation.record_run(after, 9) == after


@pytest.mark.parametrize("damage", ["seed", "branch", "freeze", "probe", "history", "parent", "completion", "reinject"])
def test_recovery_chain_rejects_changed_identity_or_partial_commit(owned, damage):
    previous, current, _ = recovery_pair(owned)
    data = read_json(current / "manifest.json")
    if damage == "seed":
        data["seed"] += 1
    elif damage == "branch":
        data["gate1"]["branch"] = "clean"
    elif damage in {"freeze", "probe"}:
        data["gate1"]["freeze_sha256" if damage == "freeze" else "probe_manifest_sha256"] = "changed"
    elif damage == "history":
        data["replay"]["source"] = str(owned / "inputs/other_history")
    elif damage == "parent":
        data["parent"]["run_id"] = "other_run"
    elif damage == "completion":
        atomic_json(previous / "raw/completed_step_000008.json", {"model_snapshot_committed": False})
    else:
        data["injection_step"] = 8
    data["signatures"] = signatures(data["learning"], data["protocol"], data["parent"])
    atomic_json(current / "manifest.json", data)
    with pytest.raises((ValueError, KeyError)):
        validation.attempt_chain(current)


@pytest.mark.parametrize("damage", ["none", "seed", "branch", "probe", "freeze", "stale", "active"])
def test_launcher_recovery_requires_latest_owned_commit_and_unchanged_inputs(owned, monkeypatch, damage):
    previous, _, parent = recovery_pair(owned)
    probe = owned / "inputs/probe"
    probe.mkdir()
    atomic_json(probe / "manifest.json", {"fixed": True})
    assets, freeze = owned / "inputs/assets.json", owned / "inputs/freeze.json"
    atomic_json(assets, {"source": True})
    atomic_json(freeze, {"frozen": True})
    data = read_json(previous / "manifest.json")
    data["gate1"].update({"probe_manifest_sha256": file_digest(probe / "manifest.json"),
                         "import_manifest_sha256": file_digest(assets), "freeze_sha256": file_digest(freeze)})
    atomic_json(previous / "manifest.json", data)
    atomic_json(previous / "raw/workflow_launch.json", {"run_id": previous.name, "pid": 99999999})
    def dead(_pid, _signal):
        raise ProcessLookupError()
    monkeypatch.setattr(launcher.os, "kill", (lambda *_: None) if damage == "active" else dead)
    if damage == "stale":
        newer, _ = snapshot(previous, 16, data["signatures"]["learning"])
        atomic_json(previous / "raw/completed_step_000016.json", {"model_snapshot_committed": True,
                    "learning_signature": data["signatures"]["learning"]})
    if damage == "probe":
        atomic_json(probe / "manifest.json", {"fixed": False})
    if damage == "freeze":
        atomic_json(freeze, {"frozen": False})
    kwargs = {"seed": 20261012 if damage == "seed" else 20261011, "phase": "formal",
              "branch": "clean" if damage == "branch" else "corrupt", "probe": probe, "assets": assets, "freeze": freeze}
    if damage == "none":
        receipt = launcher.recovery_contract(previous, parent, **kwargs)
        assert receipt["completed_through"] == 8
        assert receipt["origin_injection_step"] == 8
    else:
        with pytest.raises(ValueError):
            launcher.recovery_contract(previous, parent, **kwargs)


def probe_run(owned):
    run = owned / "outputs/probe_control"
    probe = owned / "inputs/fixed_probe"
    data = owned / "inputs/data"
    probe.mkdir()
    data.mkdir()
    inputs = probe / "fixed_inputs.pt"
    inputs.write_bytes(b"independent fixed input")
    identity = [{"case_id": "probe", "base_case_id": "probe", "role": role, "candidate_id": candidate,
                 "candidate_mask": [True, True, False], "image_sha256": "image", "batch_prefix_hash": "prefix",
                 "batch_scoring_hash": "tokens", "candidate": str(candidate)}
                for role in ["solver", "verifier"] for candidate in [0, 1]]
    receipt = {"split": "probe", "batch_identities": [["input", identity]]}
    receipt["probe_signature"] = digest(receipt)
    receipt["payload_sha256"] = file_digest(inputs)
    atomic_json(probe / "manifest.json", receipt)
    cases = [{"case_id": f"case_{i}", "image_sha256": f"image_{i}", "ground_truth": i, "scene_id": f"scene_{i}",
              "family": i % 3, "explicit_answer": i, "correct": 1, "explicit_answer_present": True,
              "native_accuracy": 1., "truncated": False} for i in range(256)]
    dataset = {"samples": [{"case_id": row["case_id"], "image_sha256": row["image_sha256"], "answer": str(row["ground_truth"]),
                "family": row["family"], "scene_id": row["scene_id"], "split": "evaluation"} for row in cases]}
    dataset["data_signature"] = digest(dataset)
    atomic_json(data / "manifest.json", dataset)
    atomic_json(run / "manifest.json", {"evaluation_split": "evaluation", "dataset_directory": str(data),
        "data_signature": dataset["data_signature"], "gate1": {"probe_directory": str(probe),
        "probe_manifest_sha256": file_digest(probe / "manifest.json")}})
    scores = {"step": 32, "split": "probe", "probe_signature": receipt["probe_signature"],
              "cases": [{**row, "token_logp": [-1., -2.]} for row in identity]}
    evaluation = {"step": 32, "split": "evaluation", "cases": cases, "summary": aggregate(cases),
        "worker_rng_restored": True, "driver_rng_restored": True, "data_cursor_unchanged": True}
    for name in ["worker_rng", "driver_rng", "data_cursor"]:
        evaluation[name + "_before"] = evaluation[name + "_after"] = "stable"
    atomic_json(run / "raw/gate1_scores_step_000032.json", scores)
    atomic_json(run / "raw/evaluation_step_000032.json", evaluation)
    return run


@pytest.mark.parametrize("damage", ["none", "subset", "padding", "nan", "image", "case_id", "answer", "rng", "summary"])
def test_probe_and_heldout_receipts_must_match_frozen_inputs_and_dataset(owned, damage):
    run = probe_run(owned)
    score_path = run / "raw/gate1_scores_step_000032.json"
    eval_path = run / "raw/evaluation_step_000032.json"
    scores, evaluation = read_json(score_path), read_json(eval_path)
    if damage == "subset":
        scores["cases"] = scores["cases"][:-1]
    elif damage == "padding":
        scores["cases"][0]["token_logp"].append(-3.)
    elif damage == "nan":
        scores["cases"][0]["token_logp"][0] = float("nan")
    elif damage == "image":
        evaluation["cases"][0]["image_sha256"] = "different"
    elif damage == "case_id":
        evaluation["cases"][0]["case_id"] = "foreign"
    elif damage == "answer":
        evaluation["cases"][0]["ground_truth"] = -1
        evaluation["cases"][0]["correct"] = 0
        evaluation["summary"] = aggregate(evaluation["cases"])
    elif damage == "rng":
        evaluation["driver_rng_after"] = "consumed"
    elif damage == "summary":
        evaluation["summary"]["accuracy"] = 0.
    score_path.write_text(json.dumps(scores, allow_nan=True), encoding="utf-8")
    atomic_json(eval_path, evaluation)
    if damage == "none":
        assert len(validation.validate_probe(run, 32)) == 5
    else:
        with pytest.raises(ValueError):
            validation.validate_probe(run, 32)


def temporary_control(owned, monkeypatch):
    model = owned / "inputs/base_model"
    model.mkdir()
    for name in ["config.json", "tokenizer_config.json", "preprocessor_config.json", "model.safetensors"]:
        (model / name).write_bytes(b"base-model-source")
    run = owned / "outputs/temporary_control"
    snapshot(run, 0)
    snapshot(run, 1)
    manifest = {"run_id": run.name, "seed": 20261011, "gate1": {"branch": "control"},
                "signatures": {"learning": "learning"}, "parent": None, "checkpoint_plan": {"start": 0, "end": 1},
                "model": model_identity(model), "overrides": {"actor_rollout_ref.model.path": str(model)}}
    atomic_json(run / "manifest.json", manifest)
    atomic_json(run / "status.json", {"status": "intentionally_stopped", "returncode": 0})
    rows = [guard.evidence(run / "manifest.json"), guard.evidence(run / "status.json")]
    # Only native unit rechecking is replaced: these tests exercise physical deletion/receipt recovery.
    monkeypatch.setattr(validation, "validate_run", lambda *_args, **_kw: {
        "passed": True, "run_id": run.name, "origin_manifest": manifest, "evidence": rows})
    proof = owned / "outputs/accepted_controls.json"
    atomic_json(proof, {"passed": True, "accepted_run_ids": [run.name], "pending_dependents": [], "evidence": rows})
    return run, proof, model


def test_S0_control_retires_all_owned_snapshots_and_keeps_base_model_reconstruction(owned, monkeypatch):
    run, proof, model = temporary_control(owned, monkeypatch)
    receipt = state.retire_temporary_weights(run, proof)
    assert receipt["reconstruction"]["kind"] == "base_model_seed_and_recorded_native_history"
    assert receipt["reconstruction"]["step"] == 0
    assert set(receipt["snapshots"]) == {"global_step_0", "global_step_1"}
    assert (model / "model.safetensors").is_file()
    for step in [0, 1]:
        path = run / f"checkpoints/global_step_{step}"
        assert not list((path / "actor").glob("model_*.pt"))
        assert (path / "actor/extra_state_world_size_4_rank_0.pt").is_file()
        assert state.certified_snapshot(path, allow_retired=True)["step"] == step
        with pytest.raises(ValueError):
            validate(path)
    assert state.retire_temporary_weights(run, proof) == receipt


def test_retirement_resumes_after_single_unlink_interruption_without_false_completion(owned, monkeypatch):
    run, proof, _ = temporary_control(owned, monkeypatch)
    original = Path.unlink
    calls = []
    def interrupted(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path.name.startswith("model_world"):
            calls.append(path)
            if len(calls) == 1:
                raise RuntimeError("simulated interruption after durable plan")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupted)
        with pytest.raises(RuntimeError):
            state.retire_temporary_weights(run, proof)
    assert (run / "raw/temporary_weight_retirement_plan.json").is_file()
    assert not (run / "raw/temporary_weight_retirement_complete.json").exists()
    with pytest.raises(FileNotFoundError):
        state.certified_snapshot(run / "checkpoints/global_step_0", allow_retired=True)
    receipt = state.retire_temporary_weights(run, proof)
    assert (run / "raw/temporary_weight_retirement_complete.json").is_file()
    assert receipt["directly_restorable_after_retirement"] is False


@pytest.mark.parametrize("damage", ["unaccepted", "pending", "evidence", "legacy_alias", "active_dependent", "failed_dependent", "proof_seed"])
def test_retirement_refuses_unaccepted_changed_or_dependent_assets(owned, monkeypatch, damage):
    run, proof, _ = temporary_control(owned, monkeypatch)
    before = {p: p.read_bytes() for p in (run / "checkpoints").rglob("*") if p.is_file()}
    payload = read_json(proof)
    if damage == "unaccepted":
        payload["accepted_run_ids"] = []
    elif damage == "pending":
        payload["pending_dependents"] = ["future_control"]
    elif damage == "evidence":
        atomic_json(run / "status.json", {"status": "failed"})
    elif damage == "legacy_alias":
        alias = owned / "outputs/alias"
        alias.symlink_to(run)
        run = alias
    elif damage == "proof_seed":
        payload["evidence"] = []
    else:
        dependent = owned / "outputs/dependent"
        cert = read_json(run / "checkpoints/global_step_1/baseline_complete.json")
        atomic_json(dependent / "manifest.json", {"parent": cert})
        atomic_json(dependent / "status.json", {"status": "running" if damage == "active_dependent" else "failed"})
    atomic_json(proof, payload)
    with pytest.raises(ValueError):
        state.retire_temporary_weights(run, proof)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("damage", ["none", "seeds", "missing_witness", "false_witness", "changed_evidence", "missing_engine", "source", "workflow", "boolean_epsilon"])
def test_accepted_freeze_binds_all_seeds_engine_workflow_and_real_witnesses(owned, monkeypatch, damage):
    source = owned / "src"
    source.mkdir()
    monkeypatch.setattr(guard, "SRC", source)
    engine = {name: None for name in launcher.ENGINE_FILES}
    workflow_names = ["run_gate1_sequence.py", "gate1_state.py", "gate1_validation.py",
                      "gate1_references.py", "gate1_evidence.py", "export_gate1.py"]
    for name in [*engine, *workflow_names]:
        (source / name).write_text("# independently frozen " + name, encoding="utf-8")
    witnesses = {}
    for name in guard.ACCEPTANCE_KEYS:
        witness = owned / "inputs" / f"{name}.json"
        atomic_json(witness, {"passed": True})
        witnesses[name] = guard.evidence(witness)
    freeze = {"status": "accepted", "seeds": guard.SEEDS, "horizon": 32, "k": 8, "epsilon_D": .01,
        "acceptance": {k: True for k in witnesses}, "evidence": witnesses,
        "source_sha256": {name: file_digest(source / name) for name in engine},
        "workflow_source_sha256": {name: file_digest(source / name) for name in workflow_names}}
    if damage == "seeds":
        freeze["seeds"] = guard.SEEDS[:3]
    elif damage == "missing_witness":
        freeze["acceptance"].pop("actual_interruption_recovery")
    elif damage in {"false_witness", "changed_evidence"}:
        p = Path(witnesses["actual_interruption_recovery"]["path"])
        atomic_json(p, {"passed": False})
        if damage == "false_witness":
            witnesses["actual_interruption_recovery"] = guard.evidence(p)
    elif damage == "missing_engine":
        freeze["source_sha256"].pop("native_adapter.py")
    elif damage == "source":
        (source / "native_adapter.py").write_text("# changed", encoding="utf-8")
    elif damage == "workflow":
        (source / "gate1_state.py").write_text("# changed", encoding="utf-8")
    elif damage == "boolean_epsilon":
        freeze["epsilon_D"] = True
    path = owned / "inputs/freeze.json"
    atomic_json(path, freeze)
    if damage == "none":
        assert guard.validate_freeze(path)["seeds"] == guard.SEEDS
    else:
        with pytest.raises(ValueError):
            guard.validate_freeze(path)


def test_actual_process_failure_recovery_and_completed_stage_reuse(owned, monkeypatch):
    """Actual tiny subprocess IO; native tensor/GPU acceptance remains a separate test."""
    sequence = owned / "outputs/sequence"
    (sequence / "raw").mkdir(parents=True)
    learning = {"seed": 20261011, "source_sha256": {}, "environment_signature": "environment",
                "model_signature": "model", "dataset_signature": "data"}
    prefix, _ = snapshot(owned / "inputs/prefix", 7, digest(learning))
    freeze = owned / "inputs/frozen.json"
    atomic_json(freeze, {"source_sha256": {}, "environment_signature": "environment",
                        "model_signature": "model", "dataset_signature": "data"})
    script = owned / "inputs/small_process.py"
    script.write_text(
        "import argparse,sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0,{str(Path(workflow.__file__).parent)!r})\n"
        "from pilot_utils import atomic_json,digest,read_json,file_digest\n"
        "from records import signatures\nfrom checkpoint_store import certify,required_names,validate\n"
        "p=argparse.ArgumentParser()\n"
        "for key in ['root','run-id','freeze','seed','phase','branch','resume-from','recovery-from','replay-from','replay-start','replay-end','injection-step','reward-delta']: p.add_argument('--'+key)\n"
        "a=p.parse_args();run=Path(a.root)/a.run_id;(run/'raw').mkdir(parents=True);(run/'checkpoints').mkdir()\n"
        f"learning={learning!r}\n"
        "parent=validate(Path(a.resume_from));protocol={'phase':a.phase,'injection_step':int(a.injection_step) if a.injection_step else None}\n"
        "gate={'branch':a.branch,'freeze_sha256':file_digest(Path(a.freeze)),'probe_manifest_sha256':'probe','import_manifest_sha256':'import'}\n"
        "replay={'source':a.replay_from,'start':int(a.replay_start),'end':int(a.replay_end),'source_learning_signature':'source'}\n"
        "if a.recovery_from:\n"
        " assert a.injection_step is None and int(a.replay_start)==9\n"
        " old=Path(a.recovery_from);recovery={'previous_attempt':str(old),'previous_manifest_sha256':file_digest(old/'manifest.json'),'checkpoint':a.resume_from,'state_signature':parent['state_signature'],'completed_through':8,'origin_injection_step':8,'origin_reward_delta':1}\n"
        " gate['recovery']=recovery;protocol['recovery']=recovery\n"
        "manifest={'run_id':run.name,'phase':a.phase,'seed':int(a.seed),'learning':learning,'selected_gpus':[4,5,6,7],'data_signature':'data','model':{'signature':'model'},'parent':parent,'protocol':protocol,'gate1':gate,'replay':replay,'checkpoint_plan':{'start':parent['step'],'end':32},'injection_step':protocol['injection_step'],'reward_delta':1}\n"
        "manifest['signatures']=signatures(learning,protocol,parent);atomic_json(run/'manifest.json',manifest)\n"
        "step=32 if a.recovery_from else 8;dest=run/f'checkpoints/global_step_{step}';dest.mkdir()\n"
        "for name in required_names(4):\n"
        " out=dest/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b'small-state')\n"
        "atomic_json(dest/'baseline_state.json',{'completed_step':step,'next_step':step+1,'full_horizon':32,'zero_signal_streak':0,'parent':parent})\n"
        "atomic_json(dest/'baseline_worker_state.json',[{'rank':r,'model':{'w':'hash'},'buffers':{},'optimizer':{'w':'hash'},'optimizer_groups':[],'scheduler':{},'rng_hash':'rng','all_parameters_and_optimizer_finite':True} for r in range(4)])\n"
        "certify(dest,learning=digest(learning),run_id=run.name,step=step,world=4,parent=parent)\n"
        "atomic_json(run/f'raw/completed_step_{step:06d}.json',{'step':step,'learning_signature':digest(learning),'model_snapshot_committed':True})\n"
        "if not a.recovery_from: atomic_json(run/'raw/group_step_000009.json',{'partial':True})\n"
        "atomic_json(run/'status.json',{'status':'completed' if a.recovery_from else 'failed','returncode':0 if a.recovery_from else 3})\n"
        "sys.exit(0 if a.recovery_from else 3)\n", encoding="utf-8")
    monkeypatch.setattr(workflow, "inspect", lambda *_args, **_kw: {"problems": []})
    monkeypatch.setattr(workflow, "validate_unit", lambda *_args, **_kw: [])
    monkeypatch.setattr(workflow, "validate_run", lambda run, **_kw: {
        "passed": True, "evidence": [guard.evidence(run / "manifest.json")]})
    base = [sys.executable, str(script), "--root", str(owned / "outputs"), "--freeze", str(freeze)]
    extra = ["--seed", "20261011", "--phase", "formal", "--branch", "corrupt", "--resume-from", str(prefix),
             "--replay-from", str(owned / "inputs/history"), "--replay-start", "8", "--replay-end", "32",
             "--injection-step", "8", "--reward-delta", "1"]
    with pytest.raises(RuntimeError):
        workflow.execute(sequence, "paired_corrupt", base, extra, [4, 5, 6, 7])
    first = owned / "outputs/paired_corrupt_a001"
    old_hash = file_digest(first / "checkpoints/global_step_8/baseline_complete.json")
    second = workflow.execute(sequence, "paired_corrupt", base, extra, [4, 5, 6, 7])
    assert second.name == "paired_corrupt_a002"
    assert read_json(first / "raw/group_step_000009.json") == {"partial": True}
    assert file_digest(first / "checkpoints/global_step_8/baseline_complete.json") == old_hash
    assert validation.record_run(second, 8) == first
    assert validation.record_run(second, 9) == second
    with monkeypatch.context() as patch:
        patch.setattr(workflow.subprocess, "Popen", lambda *_args, **_kw: pytest.fail("completed stage was recomputed"))
        assert workflow.execute(sequence, "paired_corrupt", base, extra, [4, 5, 6, 7]) == second
    assert len(read_json(sequence / "raw/stage_paired_corrupt.json")["attempts"]) == 2


def test_S0_capture_survives_native_rolling_removal_without_a_restore_claim(owned):
    import shutil
    run = owned / "outputs/source"
    native, cert = snapshot(run, 0)
    workflow.capture_initial_metadata(run)
    shutil.rmtree(native)
    captured = run / "raw/initial_state/global_step_0"
    assert state.certificate_metadata(captured)["state_signature"] == cert["state_signature"]
    marker = read_json(captured / "metadata_complete.json")
    assert marker["metadata_only"] is True and marker["directly_restorable"] is False
    assert not list((captured / "actor").glob("model_*.pt"))
    assert len(list((captured / "actor").glob("extra_state_*.pt"))) == 4
    guard.verify_evidence(marker["members"])
    with pytest.raises(ValueError):
        state.certified_snapshot(captured)


def test_S0_capture_refuses_foreign_symlink_before_writing(owned):
    run = owned / "outputs/source"
    snapshot(run, 0)
    (run / "raw").mkdir()
    foreign = owned / "inputs/foreign"
    foreign.mkdir()
    (run / "raw/initial_state").symlink_to(foreign)
    with pytest.raises(ValueError):
        workflow.capture_initial_metadata(run)
    assert not list(foreign.iterdir())


@pytest.mark.parametrize("damage", ["none", "summary", "duplicate", "unknown_context"])
def test_auxiliary_false_positive_summary_uses_independent_wrong_context_cases(owned, damage):
    from export_gate1 import auxiliary_diagnostics
    run = probe_run(owned)
    evaluation = read_json(run / "raw/evaluation_step_000032.json")
    for row in evaluation["cases"]:
        row.update({"tool_calls": 2, "tool_successes": 1, "verified_steps": 1, "repair_steps": 1})
    probes = [
        {"probe_id": "a:wrong", "expected_correct": False, "parsed": True, "accepted": True},
        {"probe_id": "b:wrong", "expected_correct": False, "parsed": False, "accepted": False},
        {"probe_id": "a:correct", "expected_correct": True, "parsed": True, "accepted": True},
        {"probe_id": "b:correct", "expected_correct": True, "parsed": True, "accepted": False}]
    summary = {"cases": 4, "parse_rate": .75, "incorrect_acceptance_rate": .5, "correct_acceptance_rate": .5}
    if damage == "summary":
        summary["incorrect_acceptance_rate"] = 1.
    elif damage == "duplicate":
        probes[1]["probe_id"] = probes[0]["probe_id"]
    elif damage == "unknown_context":
        probes[0]["expected_correct"] = None
    evaluation["verifier_probes"] = {"split": "probe", "cases": probes, "summary": summary}
    if damage == "none":
        result, families = auxiliary_diagnostics(evaluation, 20261011, 32, "clean")
        assert result["repeated_false_positive_rate"] == .5
        assert result["solver_explicit_answer_rate"] == 1
        assert result["tool_calls"] == 512
        assert len(families) == 3
    else:
        with pytest.raises(ValueError):
            auxiliary_diagnostics(evaluation, 20261011, 32, "clean")


def exporter_fixture(owned, monkeypatch):
    import export_gate1 as exporter
    monkeypatch.setattr(exporter, "VERSION", owned)
    sequence = owned / "outputs/sequence_export"
    (sequence / "raw").mkdir(parents=True)
    witnesses = {}
    for key in guard.ACCEPTANCE_KEYS:
        path = owned / "inputs" / f"witness_{key}.json"
        atomic_json(path, {"passed": True})
        witnesses[key] = guard.evidence(path)
    workflow_names = ["run_gate1_sequence.py", "gate1_state.py", "gate1_validation.py",
                      "gate1_references.py", "gate1_evidence.py", "export_gate1.py"]
    frozen = {"status": "accepted", "seeds": guard.SEEDS, "horizon": 32, "k": 8, "epsilon_D": .01,
              "acceptance": {k: True for k in witnesses}, "evidence": witnesses,
              "source_sha256": {k: file_digest(guard.SRC / k) for k in launcher.ENGINE_FILES},
              "workflow_source_sha256": {k: file_digest(guard.SRC / k) for k in workflow_names}}
    freeze_path = owned / "inputs/export_freeze.json"
    atomic_json(freeze_path, frozen)
    assets = {"status": "verified", "histories": {}, "files": {}}
    assets["manifest_signature"] = digest(assets)
    asset_path = owned / "inputs/assets.json"
    atomic_json(asset_path, assets)
    references = owned / "inputs/references"
    references.mkdir()
    reference_manifest = {"references": {}}
    reference_manifest["manifest_signature"] = digest(reference_manifest)
    atomic_json(references / "manifest.json", reference_manifest)
    atomic_json(sequence / "manifest.json", {"freeze_path": str(freeze_path), "freeze_sha256": file_digest(freeze_path),
        "seeds": guard.SEEDS, "horizon": 32, "k": 8, "inputs": {"assets": str(asset_path), "references": str(references)},
        "input_sha256": {"assets": file_digest(asset_path), "references": file_digest(references / "manifest.json")}})
    blocks = []
    revision = {"status": "no_eligible_target", "step": 8, "requested_delta": 1}
    for seed in guard.SEEDS:
        source = owned / "outputs" / f"source_{seed}"
        learning = {"seed": seed}
        snapshot(source, 32, digest(learning))
        manifest = {"run_id": source.name, "seed": seed, "learning": learning,
                    "signatures": {"learning": digest(learning)}, "checkpoint_plan": {"end": 32}}
        atomic_json(source / "manifest.json", manifest)
        atomic_json(source / "status.json", {"status": "completed", "returncode": 0})
        atomic_json(source / "raw/source_metric_evidence.json", {"optimizer_step": 32})
        accepted = sequence / f"raw/source_{seed}_accepted.json"
        rows = [guard.evidence(source / "manifest.json"), guard.evidence(source / "raw/source_metric_evidence.json")]
        atomic_json(accepted, {"run_id": source.name, "validation": {"origin_manifest": manifest, "evidence": rows}})
        atomic_json(sequence / f"raw/source_seed_{seed}.json", {"seed": seed, "source": str(source),
            "actual_source": str(source), "kind": "accepted_preparation", "acceptance": str(accepted)})
        atomic_json(sequence / f"raw/eligibility_{seed}.json", revision)
        blocks.append({"seed": seed, "valid": False, "status": "no_eligible_target", "source": str(source)})
    atomic_json(sequence / "status.json", {"status": "completed", "completed_seed_blocks": blocks})
    # Statistics/provenance tests only; the separate native tests recheck DataProto units.
    monkeypatch.setattr(exporter, "eligibility", lambda _source: deepcopy(revision))
    monkeypatch.setattr(exporter, "validate_unit", lambda source, *_a, **_kw: [guard.evidence(source / "raw/source_metric_evidence.json")])
    return exporter, sequence


def test_export_rebuilds_all_five_no_eligible_rows_from_raw_without_processed_outputs(owned, monkeypatch):
    import csv
    import shutil
    exporter, sequence = exporter_fixture(owned, monkeypatch)
    first = exporter.export(sequence)
    assert first["primary"]["decision"] == "uncertain" and first["primary"]["n"] == 0
    with (sequence / "processed/paired_seeds.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["seed"]) for row in rows] == guard.SEEDS
    assert all(row["status"] == "no_eligible_target" for row in rows)
    shutil.rmtree(sequence / "processed")
    assert exporter.export(sequence)["primary"] == first["primary"]


@pytest.mark.parametrize("damage", ["missing_seed", "duplicate_seed", "wrong_source", "source_metrics", "failed_source", "missing_source_weight"])
def test_export_refuses_selected_seeds_or_changed_source_evidence(owned, monkeypatch, damage):
    exporter, sequence = exporter_fixture(owned, monkeypatch)
    status = read_json(sequence / "status.json")
    source = Path(status["completed_seed_blocks"][0]["source"])
    if damage == "missing_seed":
        status["completed_seed_blocks"].pop()
    elif damage == "duplicate_seed":
        status["completed_seed_blocks"][1]["seed"] = guard.SEEDS[0]
    elif damage == "wrong_source":
        status["completed_seed_blocks"][0]["source"] = status["completed_seed_blocks"][1]["source"]
    elif damage == "source_metrics":
        atomic_json(source / "raw/source_metric_evidence.json", {"optimizer_step": 31})
    elif damage == "failed_source":
        atomic_json(source / "status.json", {"status": "failed", "returncode": 1})
    else:
        (source / "checkpoints/global_step_32/actor/model_world_size_4_rank_0.pt").unlink()
    atomic_json(sequence / "status.json", status)
    with pytest.raises(ValueError):
        exporter.export(sequence)
    assert not (sequence / "processed/result_summary.json").exists()


@pytest.mark.parametrize("damage", ["none", "failed", "protected0", "S0_mismatch", "S7_mismatch", "foreign_owner"])
def test_prefix_initial_retirement_requires_both_historical_states_and_preserves_full_S7(owned, monkeypatch, damage):
    import gate1_references as references
    run = owned / "outputs/prefix"
    zero, _ = snapshot(run, 0)
    seven, cert = snapshot(run, 7)
    manifest = {"run_id": run.name, "phase": "preparation", "seed": 20261012,
        "gate1": {"branch": "prefix"}, "checkpoint_plan": {"end": 7, "protected": [7]},
        "signatures": {"learning": "learning"}}
    if damage == "protected0":
        manifest["checkpoint_plan"]["protected"] = [0, 7]
    if damage == "foreign_owner":
        manifest["run_id"] = "foreign"
    atomic_json(run / "manifest.json", manifest)
    atomic_json(run / "status.json", {"status": "failed" if damage == "failed" else "intentionally_stopped", "returncode": 0})
    workflow.capture_initial_metadata(run)
    old_hash = file_digest(seven / "baseline_complete.json")
    rows = [guard.evidence(run / "manifest.json"), guard.evidence(run / "status.json")]
    monkeypatch.setattr(validation, "validate_run", lambda *_args, **_kw: {"evidence": rows})
    def comparison(actual, reference):
        mismatched = (damage == "S0_mismatch" and actual.name == "global_step_0") or (
            damage == "S7_mismatch" and actual.name == "global_step_7")
        return {"passed": not mismatched, "evidence": rows}
    monkeypatch.setattr(references, "compare_rebuilt", comparison)
    monkeypatch.setattr(state, "metadata_fingerprint", lambda *_args, **_kw: {"same_actual_metadata": True})
    monkeypatch.setattr(state, "state_fingerprint", lambda *_args, **_kw: {"same_actual_metadata": True})
    if damage == "none":
        receipt = state.retire_prefix_initial(run, owned / "inputs/ref0", owned / "inputs/ref7")
        assert not zero.exists()
        assert validate(seven)["state_signature"] == cert["state_signature"]
        assert file_digest(seven / "baseline_complete.json") == old_hash
        assert (run / "raw/initial_state/global_step_0/actor/extra_state_world_size_4_rank_0.pt").is_file()
        assert receipt["initial_directly_restorable_after_retirement"] is False
        assert state.retire_prefix_initial(run, owned / "inputs/ref0", owned / "inputs/ref7") == receipt
    else:
        with pytest.raises(ValueError):
            state.retire_prefix_initial(run, owned / "inputs/ref0", owned / "inputs/ref7")
        assert zero.exists() and file_digest(seven / "baseline_complete.json") == old_hash
