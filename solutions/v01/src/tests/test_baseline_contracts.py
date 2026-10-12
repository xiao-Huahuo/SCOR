"""Independent counterexamples for data, state, answer and visual identities."""
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_store import certify, commit, plan, retire, validate
from evaluate import aggregate, case_result, solver_answer
from pilot_utils import atomic_json, digest
from prepare_baseline_data import prepare, verify
from records import signatures, stream_seed, value_hash
from run_baseline import check_profile, storage_peak
from native_adapter import solver_cue


def snapshot(root, step, run_id="a"):
    path = root / f"global_step_{step}"
    (path / "actor").mkdir(parents=True)
    for kind in ("model", "optim", "extra_state"):
        (path / f"actor/{kind}_world_size_1_rank_0.pt").write_bytes((kind + str(step)).encode())
    for name in ("data.pt", "baseline_driver_rng.pt"):
        (path / name).write_bytes(b"saved-state")
    atomic_json(path / "baseline_state.json", {"completed_step": step, "next_step": step + 1}, immutable=True)
    atomic_json(path / "baseline_worker_state.json", [{"rank": 0}], immutable=True)
    certify(path, learning="learning", run_id=run_id, step=step, world=1, parent=None)
    return path


def test_scene_splits_and_duplicate_prompt_images(tmp_path):
    path = tmp_path / "data"
    result = prepare(path, 17, {"train": 9, "calibration": 6, "probe": 6, "evaluation": 6})
    assert len({x["scene_id"] for x in result["samples"]}) == 27
    assert len({x["case_id"] for x in result["samples"]}) == 27
    assert verify(path)["data_signature"] == result["data_signature"]
    probes = json.loads((path / "verifier_probes.json").read_text())
    assert len(probes) == 12 and sum(x["expected_correct"] for x in probes) == 6
    with pytest.raises(FileExistsError):
        prepare(path)
    with (path / "train.parquet").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        verify(path)


def test_verifier_and_tool_numbers_do_not_become_solver_answer():
    segments = [{"role": "solver", "text": "I need to calculate."},
                {"role": "verifier", "text": '{"confidence": 5, "answer": "\\boxed{5}"}'},
                {"role": "tool_observation", "text": "\\boxed{5}"}]
    assert solver_answer(segments) is None
    info = {"case_id": "a", "image_sha256": "one", "scene_id": "scene", "family": 0, "answer": "5"}
    case = case_result(info, segments, 1, [])
    assert case["correct"] == 0 and case["native_accuracy"] == 1
    assert case["native_fallback_used"]
    segments.append({"role": "regenerate", "text": "Thus \\boxed{5}."})
    assert solver_answer(segments) == 5
    other = case_result({**info, "case_id": "b", "image_sha256": "two"}, segments, 1, [])
    assert aggregate([case, other])["accuracy"] == 0.5
    with pytest.raises(ValueError, match="unique"):
        aggregate([case, case])


def test_visual_identity_includes_pixels_shape_and_dtype():
    red = Image.new("RGB", (2, 2), "red")
    blue = Image.new("RGB", (2, 2), "blue")
    assert value_hash({"prompt": "same", "image": red}) != value_hash({"prompt": "same", "image": blue})
    assert value_hash(np.zeros((2, 2), dtype=np.int32)) != value_hash(np.zeros((4,), dtype=np.int32))
    assert value_hash(np.zeros((2,), dtype=np.float32)) != value_hash(np.zeros((2,), dtype=np.int32))
    assert stream_seed(1, "train", 2, "task", 0, "solver", 0) == stream_seed(1, "train", 2, "task", 0, "solver", 0)
    assert stream_seed(1, "train", 2, "task", 0, "solver", 0) != stream_seed(1, "eval", 2, "task", 0, "solver", 0)
    assert signatures({"model": 1}, {"branch": "clean"}, None)["learning"] == signatures(
        {"model": 1}, {"branch": "corrupt"}, None)["learning"]


def test_solver_transition_keeps_actual_feedback_and_unexecuted_repair():
    feedback = "[Code Execution Result]\nOutput: 33"
    finished = solver_cue(feedback, True, False)
    assert feedback in finished and "repeat the program" in finished
    assert "\\boxed{integer}" in finished
    assert finished.count("<|im_start|>user") == 1 and finished.count("<|im_start|>assistant") == 1
    assert "changed" not in finished
    for succeeded, repaired in ((False, False), (True, True)):
        pending = solver_cue(feedback, succeeded, repaired)
        assert "fenced Python code" in pending and feedback in pending


def test_three_value_tasks_and_verifier_role_have_no_solver_command(tmp_path):
    import pyarrow.parquet as pq
    path = tmp_path / "data"
    manifest = prepare(path, 17, {"train": 6, "calibration": 6, "probe": 6, "evaluation": 6})
    for sample in manifest["samples"]:
        red, blue, green = sample["values"]
        assert all(100 <= v <= 999 for v in (red, blue, green))
        expected = (red + blue + green, red * blue * green,
                    2 * max(red, blue, green) - red - blue - green)
        assert int(sample["answer"]) == expected[sample["family"]]
    for row in pq.read_table(path / "calibration.parquet").to_pylist():
        question = row["prompt"][-1]["content"]
        if row["extra_info"]["family"] in (0, 1):
            assert "all three bars" in question
        else:
            assert "sum of the other two bar values" in question
    for row in pq.read_table(path / "calibration_verifier.parquet").to_pylist():
        assert "You are the Verifier" in row["prompt"][0]["content"]
        assert "Use a fenced Python" not in row["prompt"][-1]["content"]
        assert "expected_correct" not in row["prompt"][-1]["content"]


def test_success_without_stdout_requests_real_printed_tool_result():
    cue = solver_cue("[Code Execution Result]\nNo output", True, False)
    assert "fenced Python" in cue and "print(result)" in cue
    assert "must be only \\boxed" not in cue


@pytest.mark.parametrize("text", ["FINAL_ANSWER: 5.6", "FINAL_ANSWER: 5e2", "FINAL_ANSWER: 5/2",
                                 "FINAL_ANSWER: 5abc", "\\boxed{5} then \\boxed{5.6}", "\\boxed{}"])
def test_noninteger_final_answer_never_becomes_integer_prefix(text):
    assert solver_answer([{"role": "solver", "text": text}]) is None


def test_snapshot_without_worker_fingerprint_cannot_be_certified(tmp_path):
    path = snapshot(tmp_path, 0)
    (path / "baseline_complete.json").unlink()
    (path / "baseline_worker_state.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        certify(path, learning="learning", run_id="a", step=0, world=1, parent=None)
    assert not (path / "baseline_complete.json").exists()


def test_partial_snapshot_never_commits_and_original_remains(tmp_path):
    old = snapshot(tmp_path, 0)
    partial = tmp_path / "pending/global_step_1"
    partial.mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete"):
        certify(partial, learning="learning", run_id="a", step=1, world=1, parent=None)
    assert not (partial / "baseline_complete.json").exists()
    assert validate(old)["step"] == 0


@pytest.mark.parametrize("damage", ["missing_optim", "same_size_corruption", "different_learning"])
def test_state_validation_rejects_counterexamples(tmp_path, damage):
    path = snapshot(tmp_path, 1)
    if damage == "missing_optim":
        (path / "actor/optim_world_size_1_rank_0.pt").unlink()
    elif damage == "same_size_corruption":
        file = path / "actor/model_world_size_1_rank_0.pt"
        data = file.read_bytes()
        file.write_bytes(b"X" + data[1:])
    with pytest.raises(ValueError):
        validate(path, "other" if damage == "different_learning" else "learning")


def test_atomic_commit_and_owned_retention(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    protected = snapshot(root, 0)
    previous = snapshot(root, 1)
    foreign = snapshot(root, 99, "another-run")
    pending = snapshot(tmp_path / "pending", 2)
    new = root / "global_step_2"
    commit(pending, new)
    assert retire(root, new, "learning", "a", 1, [0]) == [previous.name]
    assert protected.exists() and foreign.exists() and validate(new)["step"] == 2
    with pytest.raises(ValueError, match="another run"):
        retire(root, foreign, "learning", "a", 1, [0])


def test_same_size_change_after_certification_rejects_fast_commit(tmp_path):
    path = snapshot(tmp_path, 1)
    # Reconstruct an in-memory fresh proof for this already verified fixture.
    certificate = validate(path)
    certificate["_verified_metadata"] = {}
    for name in certificate["files"]:
        st = (path / name).stat()
        certificate["_verified_metadata"][name] = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]
    member = path / "actor/model_world_size_1_rank_0.pt"
    data = member.read_bytes()
    member.write_bytes(b"Z" + data[1:])
    with pytest.raises(ValueError, match="changed|corrupted"):
        commit(path, tmp_path / "target/global_step_1", proof=certificate)
    assert path.exists()


def test_full_horizon_stop_and_initial_state_budget():
    schedule = plan(32, 8, stop=1, protected=(0, 7))
    assert schedule["full_horizon"] == 32 and schedule["scheduled_steps"] == [0, 1]
    assert storage_peak(schedule, 100, 30) == 130
    full = plan(32, 8, protected=(0,))
    assert storage_peak(full, 100, 30) == 230  # S0, old rolling and replacement.
    with pytest.raises(ValueError):
        plan(32, 8, protected=(33,))


def test_candidate_profile_topology_and_update_scope():
    profile = json.loads((Path(__file__).resolve().parents[1] / "configs/baseline_2b_candidate.json").read_text())
    check_profile(profile, [4, 5, 6, 7])
    with pytest.raises(ValueError):
        check_profile(profile, [6, 7])
    profile["overrides"]["actor_rollout_ref.actor.ppo_mini_batch_size"] = 2
    with pytest.raises(ValueError):
        check_profile(profile, [4, 5, 6, 7])
