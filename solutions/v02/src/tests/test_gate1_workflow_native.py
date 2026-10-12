"""Actual native DataProto counterexamples for Gate1 acceptance, with no model or GPU."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")
TensorDict = pytest.importorskip("tensordict").TensorDict
DataProto = pytest.importorskip("verl").DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

import gate1_evidence as guard
from gate1_contract import fixed_unit_audit
from gate1_validation import validate_unit
from interventions import false_positive
from pilot_utils import atomic_json, file_digest, read_json
from records import prompt_hash, tensor_save


def objects(items):
    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def group():
    ids = objects([f"group_{i // 4}" for i in range(16)])
    scores = torch.tensor([[0., 0., float(i % 2)] for i in range(16)])
    mask = torch.ones_like(scores)
    advantages, returns = compute_grpo_outcome_advantage(scores, mask, ids)
    responses = torch.tensor([[11, 12, 13]] * 16)
    inputs = torch.cat([torch.ones(16, 2, dtype=torch.long), responses], -1)
    segments = [[{"role": "solver", "start": 0, "end": 3, "trainable": True,
                 "token_ids": [11, 12, 13], "text": "\\boxed{1}" if i % 2 else "\\boxed{2}",
                 "truncated": False}] for i in range(16)]
    metadata = {"uid": ids, "extra_info": objects([{"case_id": f"case_{i // 4}", "answer": "1"} for i in range(16)]),
        "baseline_case_id": objects([f"case_{i // 4}" for i in range(16)]),
        "baseline_slot": np.arange(16) % 4, "baseline_segments": objects(segments), "step_data": objects([[] for _ in range(16)]),
        "verify_probs": np.zeros(16), "num_steps": np.ones(16, dtype=np.int64),
        "num_repairs": np.zeros(16, dtype=np.int64), "final_answers": objects(["1" if i % 2 else "2" for i in range(16)]),
        "final_generation_step": np.ones(16, dtype=np.int64)}
    tensors = {"prompts": inputs[:, :2], "responses": responses, "input_ids": inputs,
        "attention_mask": torch.ones(16, 5, dtype=torch.long), "position_ids": torch.arange(5).expand(16, 3, 5),
        "multiturn_mask": mask.bool(), "response_mask": mask, "token_level_scores": scores,
        "token_level_rewards": scores.clone(), "advantages": advantages, "returns": returns,
        "old_log_probs": -torch.ones_like(scores), "ref_log_prob": -torch.ones_like(scores)}
    return DataProto(batch=TensorDict(tensors, batch_size=[16]), non_tensor_batch=metadata)


def record(run, data, learning, native_returns, revision):
    generation = DataProto(batch=TensorDict({k: data.batch[k].clone() for k in [
        "prompts", "responses", "input_ids", "attention_mask", "position_ids", "multiturn_mask"]}, batch_size=[16]),
        non_tensor_batch=deepcopy(data.non_tensor_batch))
    tensor_save(data, run / "raw/training_group_step_000008.pt")
    tensor_save({"step": 8, "learning_signature": learning, "output": generation}, run / "raw/generation_step_000008.pt")
    mask = data.batch["multiturn_mask"]
    atomic_json(run / "raw/group_step_000008.json", {"step": 8, "trajectories": 16, "groups": 4,
        "group_ids": data.non_tensor_batch["uid"].tolist(), "trainable_tokens": int(mask.sum()),
        "nonzero_advantage_tokens": int(((data.batch["advantages"] != 0) & mask).sum()), "payload_hash": prompt_hash(data)})
    atomic_json(run / "raw/completed_step_000008.json", {"step": 8, "learning_signature": learning,
        "group_sha256": file_digest(run / "raw/group_step_000008.json"), "model_snapshot_committed": True})
    atomic_json(run / "raw/reward_step_000008.json", {"step": 8, "native_returns": native_returns,
        "used_returns": data.batch["token_level_scores"].sum(-1).tolist(), "native_info": {"acc": native_returns},
        "revision": revision})
    atomic_json(run / "raw/metrics_step_000008.json", {"baseline/optimizer_step": 8., "baseline/parameter_probe_finite": 1.})


@pytest.mark.parametrize("damage", ["none", "current_anchor", "group_multiplicity", "optimizer", "nonfinite",
                                   "reward", "partial", "native_group_advantage", "nonterminal_injection"])
def test_acceptance_recomputes_real_typed_group_reward_and_native_GRPO(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    source, run = tmp_path / "source", tmp_path / "actual"
    original = group()
    native = original.batch["token_level_scores"].sum(-1).tolist()
    atomic_json(source / "manifest.json", {"signatures": {"learning": "source-learning"}})
    record(source, original, "source-learning", native, {"status": "none", "step": 8})
    actual = deepcopy(original)
    injected = damage in {"native_group_advantage", "nonterminal_injection"}
    if injected:
        actual.batch["token_level_scores"], revision = false_positive(actual, actual.batch["token_level_scores"],
            native, step=8, alpha_out=1., delta=1)
        actual.batch["token_level_rewards"] = actual.batch["token_level_scores"].clone()
        actual.batch["advantages"], actual.batch["returns"] = compute_grpo_outcome_advantage(
            actual.batch["token_level_rewards"], actual.batch["response_mask"], actual.non_tensor_batch["uid"])
    else:
        revision = {"status": "none", "step": 8}
    atomic_json(run / "manifest.json", {"signatures": {"learning": "actual-learning"}, "replay": {
        "source": str(source), "source_learning_signature": "source-learning", "start": 8, "end": 8}})
    if damage == "current_anchor":
        actual.batch["old_log_probs"] -= .3
    audit = fixed_unit_audit(actual, source, 8, "source-learning", injected=injected)
    if damage == "group_multiplicity":
        actual.non_tensor_batch["uid"][0] = "group_1"
    elif damage == "nonfinite":
        actual.batch["old_log_probs"][0, 0] = float("nan")
    elif damage == "reward":
        actual.batch["token_level_scores"][0, -1] += 1
    elif damage == "native_group_advantage":
        actual.batch["advantages"][1:4] = original.batch["advantages"][1:4]
    elif damage == "nonterminal_injection":
        row = revision["row"]
        actual.batch["token_level_scores"][row, -1] -= 1
        actual.batch["token_level_scores"][row, 0] += 1
        actual.batch["token_level_rewards"] = actual.batch["token_level_scores"].clone()
    record(run, actual, "actual-learning", native, revision)
    atomic_json(run / "raw/gate1_unit_step_000008.json", audit)
    if damage == "optimizer":
        atomic_json(run / "raw/metrics_step_000008.json", {"baseline/optimizer_step": 7., "baseline/parameter_probe_finite": 1.})
    if damage == "partial":
        (run / "raw/completed_step_000008.json").unlink()
    if damage in {"none", "current_anchor"}:
        assert validate_unit(run, 8, expected_injected=False)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            validate_unit(run, 8, expected_injected=injected)


def test_actual_native_metadata_materialization_fingerprints_and_restore_boundary(tmp_path, monkeypatch):
    import gate1_references as references
    from checkpoint_store import certify, required_names
    from gate1_state import certified_snapshot, metadata_fingerprint, state_fingerprint
    from pilot_utils import digest
    from records import value_hash
    version = tmp_path / "solutions/v02"
    (version / "inputs").mkdir(parents=True)
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(references, "VERSION", version)
    source = tmp_path / "historical"
    learning = {"seed": 20261011, "source_sha256": {}}
    atomic_json(source / "manifest.json", {"run_id": source.name, "seed": 20261011,
        "learning": learning, "signatures": {"learning": digest(learning)}})
    for step in [0, 7, 32]:
        snapshot = source / f"checkpoints/global_step_{step}"
        snapshot.mkdir(parents=True)
        for name in required_names(4):
            if name in {"baseline_state.json", "baseline_worker_state.json", "baseline_driver_rng.pt", "data.pt"}:
                continue
            tensor_save({"small_tensor": torch.tensor([step, 1.])}, snapshot / name)
        tensor_save({"python": ("typed", 1), "torch": torch.tensor([1, 2], dtype=torch.uint8)}, snapshot / "baseline_driver_rng.pt")
        tensor_save({"cursor": step, "data_order": torch.tensor([1, 2])}, snapshot / "data.pt")
        atomic_json(snapshot / "baseline_state.json", {"completed_step": step, "next_step": step + 1,
            "full_horizon": 32, "zero_signal_streak": 0})
        atomic_json(snapshot / "baseline_worker_state.json", [{"rank": rank,
            "model": {"w": value_hash(torch.tensor([1., 2.]))}, "buffers": {}, "optimizer": {},
            "optimizer_groups": [{"lr": 5e-7}], "scheduler": {"last_epoch": step}, "rng_hash": "rng",
            "all_parameters_and_optimizer_finite": True} for rank in range(4)])
        certify(snapshot, learning=digest(learning), run_id=source.name, step=step, world=4, parent=None)
    assets = {"status": "verified", "histories": {"20261011": {"source": str(source),
        "learning_signature": digest(learning)}}}
    assets["manifest_signature"] = digest(assets)
    path = version / "inputs/import.json"
    atomic_json(path, assets)
    output = version / "inputs/references"
    references.materialize(path, output)
    metadata = output / "seed_20261011/global_step_0"
    checked = references.validate_reference(metadata)
    assert checked["entry"]["metadata_only"] is True
    assert not (metadata / "actor").exists()
    original = source / "checkpoints/global_step_0"
    assert metadata_fingerprint(metadata) == state_fingerprint(original)
    assert references.compare_rebuilt(original, metadata)["passed"] is True
    with pytest.raises(ValueError):
        certified_snapshot(metadata)
    # Real native .pt fingerprints, physical capture and canonical retirement;
    # native training-unit validation itself is covered by the nine tests above.
    import shutil
    import gate1_state as state
    import gate1_validation as validation
    import run_gate1_sequence as workflow
    monkeypatch.setattr(state, "VERSION", version)
    monkeypatch.setattr(workflow, "VERSION", version)
    prefix = version / "outputs/prefix"
    for step in [0, 7]:
        target = prefix / f"checkpoints/global_step_{step}"
        shutil.copytree(source / f"checkpoints/global_step_{step}", target)
        (target / "baseline_complete.json").unlink()
        certify(target, learning=digest(learning), run_id=prefix.name, step=step, world=4, parent=None)
    atomic_json(prefix / "manifest.json", {"run_id": prefix.name, "seed": 20261011, "phase": "preparation",
        "gate1": {"branch": "prefix"}, "checkpoint_plan": {"end": 7, "protected": [7]},
        "signatures": {"learning": digest(learning)}})
    atomic_json(prefix / "status.json", {"status": "intentionally_stopped", "returncode": 0})
    rows = [guard.evidence(prefix / "manifest.json"), guard.evidence(prefix / "status.json")]
    monkeypatch.setattr(validation, "validate_run", lambda *_a, **_kw: {"evidence": rows})
    workflow.capture_initial_metadata(prefix)
    seven_sha = file_digest(prefix / "checkpoints/global_step_7/baseline_complete.json")
    receipt = state.retire_prefix_initial(prefix, metadata, output / "seed_20261011/global_step_7")
    assert not (prefix / "checkpoints/global_step_0").exists()
    assert file_digest(prefix / "checkpoints/global_step_7/baseline_complete.json") == seven_sha
    assert (prefix / "raw/initial_state/global_step_0/actor/extra_state_world_size_4_rank_0.pt").exists()
    assert state.retire_prefix_initial(prefix, metadata, output / "seed_20261011/global_step_7") == receipt
    # Corrupt one retained field while leaving the signed import/certificate intact.
    (metadata / "baseline_driver_rng.pt").write_bytes(b"truncated")
    with pytest.raises(ValueError):
        references.validate_reference(metadata)


@pytest.mark.parametrize("damage", ["none", "padding", "formal", "wrong_source"])
def test_legacy_uniform_control_masks_are_verified_from_actual_cached_tensors(tmp_path, monkeypatch, damage):
    from gate1_validation import legacy_uniform_masks
    from pilot_utils import digest
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    run, directory = tmp_path / "old_control", tmp_path / "old_inputs"
    script = run / "raw/source/gate1_probes.py"
    script.parent.mkdir(parents=True)
    script.write_text('SOLVER_CANDIDATES = ["Final answer: 0", "Final answer: 1"]\n', encoding="utf-8")
    data = group()
    scoring = DataProto(batch=TensorDict({key: data.batch[key].clone() for key in [
        "prompts", "responses", "input_ids", "attention_mask", "position_ids", "multiturn_mask"]}, batch_size=[16]),
        non_tensor_batch=deepcopy(data.non_tensor_batch))
    if damage == "padding":
        scoring.batch["multiturn_mask"][0, -1] = False
        scoring.batch["attention_mask"][0, -1] = 0
    cases = [{"case_id": f"p_{i}", "base_case_id": f"p_{i}", "role": "solver", "candidate_id": 0,
              "candidate": "Final answer: 0"} for i in range(16)]
    batch = {"data": scoring, "input_hash": prompt_hash(scoring), "cases": cases}
    tensor_save([batch], directory / "fixed_inputs.pt")
    receipt = {"solver_candidates": ["Final answer: 0", "Final answer: 1"],
               "verifier_candidates": ["accept", "reject"], "source_sha256": file_digest(script),
               "batch_identities": [[batch["input_hash"], cases]]}
    atomic_json(run / "manifest.json", {"phase": "formal" if damage == "formal" else "control",
        "gate1": {"branch": "control", "freeze_sha256": None}, "learning": {"source_sha256": {
        "gate1_probes.py": "wrong-source" if damage == "wrong_source" else file_digest(script)}}})
    if damage == "none":
        result = legacy_uniform_masks(run, directory, receipt)
        assert len(result) == 16 and set(result.values()) == {3}
        assert all("candidate_mask" not in row for row in cases)
    else:
        with pytest.raises(ValueError):
            legacy_uniform_masks(run, directory, receipt)


def early_clean_block(tmp_path, monkeypatch):
    """Real tiny torch states/IO; training-unit validation is tested independently above."""
    import gate1_state as state
    import gate1_validation as validation
    import gate1_references as references
    import run_gate1_sequence as workflow
    from checkpoint_store import certify, required_names
    from records import signatures, value_hash
    from pilot_utils import digest
    from run_gate1 import ENGINE_FILES
    version = tmp_path / "solutions/v02"
    (version / "outputs").mkdir(parents=True)
    (version / "inputs").mkdir()
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    for module in [state, references, workflow]:
        monkeypatch.setattr(module, "VERSION", version)
    src = Path(workflow.__file__).parent
    canonical = {name: file_digest(src.parents[1] / "v01/src" / name) for name in ENGINE_FILES[:12]}
    learning = {"seed": 20261011, "source_sha256": {name: file_digest(src / name) for name in ENGINE_FILES},
                "environment_signature": "environment", "model_signature": "model", "dataset_signature": "data"}
    old_learning = {**learning, "source_sha256": canonical}

    def small_snapshot(run, step, identity, parent=None):
        path = run / f"checkpoints/global_step_{step}"
        path.mkdir(parents=True)
        for name in required_names(4) - {"baseline_state.json", "baseline_worker_state.json"}:
            tensor_save({"step": torch.tensor([step, 1])}, path / name)
        atomic_json(path / "baseline_state.json", {"completed_step": step, "next_step": step + 1,
            "full_horizon": 32, "zero_signal_streak": 0})
        atomic_json(path / "baseline_worker_state.json", [{"rank": rank,
            "model": {"w": value_hash(torch.tensor([1., 2.]))}, "buffers": {}, "optimizer": {"Adam": "same"},
            "optimizer_groups": [{"lr": 5e-7}], "scheduler": {"last_epoch": step}, "rng_hash": "same",
            "all_parameters_and_optimizer_finite": True} for rank in range(4)])
        certify(path, learning=digest(identity), run_id=run.name, step=step, world=4, parent=parent)
        return path

    source = version / "outputs/source"
    atomic_json(source / "manifest.json", {"run_id": source.name, "seed": 20261011, "learning": old_learning,
        "signatures": {"learning": digest(old_learning)}})
    for step in [0, 7, 32]:
        small_snapshot(source, step, old_learning)
    prefix = source / "checkpoints/global_step_7"
    parent = state.certified_snapshot(prefix)
    for step in range(8, 33):
        atomic_json(source / f"raw/completed_step_{step:06d}.json", {"step": step, "fixture": "native unit tested separately"})
    assets = version / "inputs/assets.json"
    imported = {"status": "verified", "canonical_learning": old_learning, "canonical_source_sha256": canonical,
        "histories": {"20261011": {"source": str(source), "learning_signature": digest(old_learning)}}}
    imported["manifest_signature"] = digest(imported)
    atomic_json(assets, imported)
    freeze = version / "inputs/freeze.json"
    atomic_json(freeze, {key: learning[key] for key in ["source_sha256", "environment_signature", "model_signature", "dataset_signature"]})
    sequence = version / "outputs/sequence"
    atomic_json(sequence / "manifest.json", {"freeze_path": str(freeze), "freeze_sha256": file_digest(freeze),
                                            "inputs": {"assets": str(assets)}})
    references.materialize(assets, version / "inputs/references")
    branches = {}
    for branch in ["clean", "corrupt", "repeat"]:
        run = version / "outputs" / branch
        manifest = {"run_id": branch, "seed": 20261011, "phase": "formal", "selected_gpus": [4, 5, 6, 7],
            "learning": learning, "parent": parent, "protocol": {"phase": "formal", "injection_step": 8 if branch == "corrupt" else None},
            "gate1": {"branch": branch, "freeze_sha256": file_digest(freeze)},
            "checkpoint_plan": {"start": 7, "end": 32}, "injection_step": 8 if branch == "corrupt" else None,
            "replay": {"source": str(source), "start": 8, "end": 32, "source_learning_signature": digest(old_learning)}}
        manifest["signatures"] = signatures(learning, manifest["protocol"], parent)
        atomic_json(run / "manifest.json", manifest)
        atomic_json(run / "status.json", {"status": "completed", "returncode": 0})
        small_snapshot(run, 32, learning, parent)
        for step in [8, 16, 24, 32]:
            atomic_json(run / f"raw/gate1_scores_step_{step:06d}.json", {"probe_signature": "tiny-fixed",
                "cases": [{"case_id": "case", "base_case_id": "case", "role": role, "candidate_id": 0,
                           "token_logp": [-1.]} for role in ["solver", "verifier"]]})
            atomic_json(run / f"raw/evaluation_step_{step:06d}.json", {"cases": [{"case_id": "case", "image_sha256": "image",
                                                                                "ground_truth": 1, "family": "family"}]})
        branches[branch] = run

    def validated(run, **_kwargs):
        manifest = read_json(run / "manifest.json")
        point = run / "checkpoints/global_step_32"
        return {"passed": True, "run_id": run.name, "units": 25, "origin_manifest": manifest,
            "learning_signature": manifest["signatures"]["learning"], "parent_state_signature": parent["state_signature"],
            "replay": manifest["replay"], "attempts": [str(run)], "evidence": [guard.evidence(run / "manifest.json"),
                guard.evidence(run / "status.json"), *[guard.evidence(point / name) for name in [
                    "baseline_complete.json", "baseline_state.json", "baseline_worker_state.json", "baseline_driver_rng.pt", "data.pt"]]]}
    monkeypatch.setattr(validation, "validate_run", validated)
    monkeypatch.setattr(workflow, "validate_run", validated)
    monkeypatch.setattr(validation, "validate_unit", lambda run, step, **_kw: [guard.evidence(run / f"raw/completed_step_{step:06d}.json")])
    return workflow, state, validation, sequence, source, prefix, branches, freeze


@pytest.mark.parametrize("reference_kind", ["new_source", "historical"])
@pytest.mark.parametrize("retirement_mode", ["complete", "interrupted"])
def test_early_clean_retirement_rebuilds_same_proof_and_final_pair_after_repeat_retirement(tmp_path, monkeypatch, reference_kind, retirement_mode):
    workflow, state, validation, sequence, source, prefix, branches, freeze = early_clean_block(tmp_path, monkeypatch)
    reference = source.parents[1] / "inputs/references/seed_20261011/global_step_32" if reference_kind == "historical" else None
    clean = branches["clean"]
    proof_path = sequence / "raw/clean_source_seed_20261011.json"
    before = workflow.clean_source_proof(sequence, 20261011, source, prefix, clean, file_digest(freeze),
        validation.validate_branch(clean, 20261011, "clean", file_digest(freeze)), reference)
    guard.immutable_json(proof_path, before)
    state.retire_temporary_weights(clean, proof_path, prefix)
    assert not list((clean / "checkpoints/global_step_32/actor").glob("model_*.pt"))
    assert state.certified_snapshot(source / "checkpoints/global_step_32")["step"] == 32
    assert workflow.clean_source_proof(sequence, 20261011, source, prefix, clean, file_digest(freeze),
        validation.validate_branch(clean, 20261011, "clean", file_digest(freeze), allow_retired=True), reference) == before
    revision = {"status": "applied", "fixture": "selection tested separately"}
    atomic_json(sequence / "raw/eligibility_20261011.json", revision)
    atomic_json(branches["corrupt"] / "raw/reward_step_000008.json", {"revision": revision})
    monkeypatch.setattr(workflow, "eligibility", lambda _source: revision)
    monkeypatch.setattr(workflow, "initial_point", lambda *_a: {"curve": {"step": 0}, "evidence": []})
    paired = workflow.paired_proof(sequence, 20261011, source, prefix, branches, revision, file_digest(freeze), reference)
    paired_path = sequence / "raw/paired_seed_20261011.json"
    guard.immutable_json(paired_path, paired)
    if retirement_mode == "complete":
        state.retire_temporary_weights(branches["repeat"], paired_path, prefix)
    else:
        original = Path.unlink
        def interrupted(path, *args, **kwargs):
            original(path, *args, **kwargs)
            if path.name.startswith("model_world"):
                raise RuntimeError("actual repeat retirement interruption")
        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", interrupted)
            with pytest.raises(RuntimeError):
                state.retire_temporary_weights(branches["repeat"], paired_path, prefix)
        assert not (branches["repeat"] / "raw/temporary_weight_retirement_complete.json").exists()
        monkeypatch.setattr(workflow.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("completed repeat was recomputed"))
        atomic_json(sequence / "raw/stage_repeat_stage.json", {"stage": "repeat_stage", "attempts": [{"run_id": branches["repeat"].name}]})
        assert workflow.execute(sequence, "repeat_stage", [sys.executable, "unused", "--freeze", str(freeze)],
            ["--seed", "20261011", "--phase", "formal", "--branch", "repeat", "--resume-from", str(prefix)],
            [4, 5, 6, 7]) == branches["repeat"]
    assert workflow.paired_proof(sequence, 20261011, source, prefix, branches, revision, file_digest(freeze), reference) == paired
    assert str(clean / "raw/temporary_weight_retirement_complete.json") in {row["path"] for row in paired["evidence"]}
    assert str(branches["repeat"] / "raw/temporary_weight_retirement_complete.json") not in {row["path"] for row in paired["evidence"]}
    assert state.certified_snapshot(branches["corrupt"] / "checkpoints/global_step_32")["step"] == 32


@pytest.mark.parametrize("damage", ["clean_Adam", "source_missing_model", "freeze", "failed_clean", "missing_early", "foreign_proof"])
def test_early_clean_rejects_before_deletion_or_retrospective_acceptance(tmp_path, monkeypatch, damage):
    from checkpoint_store import certify
    workflow, state, validation, sequence, source, prefix, branches, freeze = early_clean_block(tmp_path, monkeypatch)
    clean, point = branches["clean"], branches["clean"] / "checkpoints/global_step_32"
    if damage == "clean_Adam":
        workers = read_json(point / "baseline_worker_state.json")
        workers[0]["optimizer"]["Adam"] = "different"
        atomic_json(point / "baseline_worker_state.json", workers)
        (point / "baseline_complete.json").unlink()
        manifest = read_json(clean / "manifest.json")
        certify(point, learning=manifest["signatures"]["learning"], run_id=clean.name, step=32, world=4, parent=manifest["parent"])
    elif damage == "source_missing_model":
        next((source / "checkpoints/global_step_32/actor").glob("model_*.pt")).unlink()
    elif damage == "freeze":
        atomic_json(freeze, {**read_json(freeze), "environment_signature": "changed"})
    elif damage == "failed_clean":
        atomic_json(clean / "status.json", {"status": "failed", "returncode": -9})
    else:
        path = sequence / "raw/clean_source_seed_20261011.json"
        if damage == "foreign_proof":
            atomic_json(path, {"passed": True})
        atomic_json(clean / "raw/temporary_weight_retirement_complete.json", {"group_proof": str(sequence / "raw/foreign.json"),
            "group_proof_sha256": "0" * 64})
    weights = {path: path.read_bytes() for path in (point / "actor").glob("*.pt")}
    with pytest.raises((ValueError, FileNotFoundError)):
        workflow.clean_source_proof(sequence, 20261011, source, prefix, clean, file_digest(freeze),
            validation.validate_branch(clean, 20261011, "clean", file_digest(freeze)))
    assert all(path.read_bytes() == payload for path, payload in weights.items())
    assert not (clean / "raw/temporary_weight_retirement_plan.json").exists()


@pytest.mark.parametrize("damage", ["none", "source_missing_model", "foreign_proof", "wrong_anchor"])
def test_execute_resumes_partial_clean_retirement_without_relaunching_or_changing_proof(tmp_path, monkeypatch, damage):
    workflow, state, validation, sequence, source, prefix, branches, freeze = early_clean_block(tmp_path, monkeypatch)
    clean = branches["clean"]
    proof_path = sequence / "raw/clean_source_seed_20261011.json"
    proof = workflow.clean_source_proof(sequence, 20261011, source, prefix, clean, file_digest(freeze),
        validation.validate_branch(clean, 20261011, "clean", file_digest(freeze)))
    guard.immutable_json(proof_path, proof)
    original = Path.unlink
    def interrupted(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if path.name.startswith("model_world"):
            raise RuntimeError("actual interruption after one tiny model shard unlink")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupted)
        with pytest.raises(RuntimeError):
            state.retire_temporary_weights(clean, proof_path, prefix)
    assert not (clean / "raw/temporary_weight_retirement_complete.json").exists()
    plan_path = clean / "raw/temporary_weight_retirement_plan.json"
    if damage == "source_missing_model":
        next((source / "checkpoints/global_step_32/actor").glob("model_*.pt")).unlink()
    elif damage in {"foreign_proof", "wrong_anchor"}:
        plan = read_json(plan_path)
        if damage == "foreign_proof":
            other = sequence / "raw/foreign.json"
            atomic_json(other, proof)
            plan["group_proof"] = str(other)
            plan["group_proof_sha256"] = file_digest(other)
        else:
            plan["reconstruction"]["path"] = str(source / "checkpoints/global_step_0")
        atomic_json(plan_path, plan)
    remaining = {path: path.read_bytes() for path in (clean / "checkpoints/global_step_32/actor").glob("*.pt")}
    monkeypatch.setattr(workflow.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("completed clean was recomputed"))
    atomic_json(sequence / "raw/stage_clean_stage.json", {"stage": "clean_stage", "attempts": [{"run_id": clean.name}]})
    args = (sequence, "clean_stage", [sys.executable, "unused", "--freeze", str(freeze)],
            ["--seed", "20261011", "--phase", "formal", "--branch", "clean", "--resume-from", str(prefix)], [4, 5, 6, 7])
    if damage != "none":
        with pytest.raises((ValueError, FileNotFoundError)):
            workflow.execute(*args)
        assert all(path.read_bytes() == payload for path, payload in remaining.items())
        assert not (clean / "raw/temporary_weight_retirement_complete.json").exists()
        return
    result = workflow.execute(*args)
    assert result == clean and read_json(proof_path) == proof
    assert (clean / "raw/temporary_weight_retirement_complete.json").is_file()
    assert state.certified_snapshot(source / "checkpoints/global_step_32")["step"] == 32


@pytest.mark.parametrize("damage", ["complete_only_wrong_anchor", "plan_complete_mismatch"])
def test_execute_rejects_retired_receipt_anchor_or_plan_mismatch(tmp_path, monkeypatch, damage):
    workflow, state, validation, sequence, source, prefix, branches, freeze = early_clean_block(tmp_path, monkeypatch)
    clean = branches["clean"]
    proof_path = sequence / "raw/clean_source_seed_20261011.json"
    guard.immutable_json(proof_path, workflow.clean_source_proof(sequence, 20261011, source, prefix, clean,
        file_digest(freeze), validation.validate_branch(clean, 20261011, "clean", file_digest(freeze))))
    state.retire_temporary_weights(clean, proof_path, prefix)
    path = clean / "raw/temporary_weight_retirement_complete.json"
    receipt = read_json(path)
    receipt["reconstruction"]["path"] = str(source / "checkpoints/global_step_0")
    receipt["reconstruction"]["state_signature"] = state.certified_snapshot(source / "checkpoints/global_step_0")["state_signature"]
    atomic_json(path, receipt)
    if damage == "complete_only_wrong_anchor":
        (clean / "raw/temporary_weight_retirement_plan.json").unlink()
    monkeypatch.setattr(workflow.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("completed clean was recomputed"))
    atomic_json(sequence / "raw/stage_clean_stage.json", {"stage": "clean_stage", "attempts": [{"run_id": clean.name}]})
    with pytest.raises(ValueError):
        workflow.execute(sequence, "clean_stage", [sys.executable, "unused", "--freeze", str(freeze)],
            ["--seed", "20261011", "--phase", "formal", "--branch", "clean", "--resume-from", str(prefix)], [4, 5, 6, 7])
