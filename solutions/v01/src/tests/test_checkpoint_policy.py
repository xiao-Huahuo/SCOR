"""Fault and boundary checks for periodic snapshot storage, using simulated shards."""
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_policy import (complete_checkpoint, install, plan_checkpoints,
                               required_disk_bytes, retire_checkpoints)
from pilot_utils import atomic_json, read_json


def snapshot(root, step, signature="same-training"):
    path = root / f"global_step_{step}"
    (path / "actor").mkdir(parents=True)
    shards = [f"actor/{name}.pt" for name in ("model", "optimizer", "extra")]
    for name in shards + ["data.pt", "scor_driver_rng.pt"]:
        (path / name).write_bytes(b"simulated-unit-test-shard")
    atomic_json(path / "scor_complete.json", {"step": step, "world_size": 1,
                "training_signature": signature, "worker_shards": shards}, immutable=True)
    return path


def test_schedule_includes_final_stop_and_protected_steps():
    assert plan_checkpoints(8, 3)["scheduled_steps"] == [3, 6, 8]
    assert plan_checkpoints(8, 3, start=3)["scheduled_steps"] == [6, 8]
    assert plan_checkpoints(8, 3, stop=4)["scheduled_steps"] == [3, 4]
    assert plan_checkpoints(8, 3, protected=[1, 4])["scheduled_steps"] == [1, 3, 4, 6, 8]
    assert plan_checkpoints(2, 2, stop=1)["scheduled_steps"] == [1]
    assert plan_checkpoints(2, 2, enabled=False)["scheduled_steps"] == []


def test_budget_counts_interval_and_transient_replacement_space():
    model_bytes = 7_510_000_000
    single = plan_checkpoints(2, 2)
    assert single["peak_snapshot_slots"] == 1
    assert required_disk_bytes(model_bytes, single) == 55 * 2**30
    assert plan_checkpoints(8, 2, keep=1)["peak_snapshot_slots"] == 2
    assert plan_checkpoints(8, 2, keep=2)["peak_snapshot_slots"] == 3
    assert plan_checkpoints(8, 2, keep=1, protected=[1])["peak_snapshot_slots"] == 3
    assert required_disk_bytes(model_bytes, plan_checkpoints(2, 2, enabled=False)) == 5 * 2**30


@pytest.mark.parametrize("kwargs", [dict(interval=0), dict(keep=0), dict(start=8),
                                   dict(start=2, stop=2), dict(protected=[9]),
                                   dict(enabled=False, protected=[1])])
def test_invalid_or_unrecoverable_policy_is_rejected(kwargs):
    values = dict(total=8, interval=2)
    values.update(kwargs)
    with pytest.raises(ValueError):
        plan_checkpoints(**values)


def test_failed_replacement_preserves_last_usable_snapshot(tmp_path):
    old = snapshot(tmp_path, 1)
    partial = tmp_path / "global_step_2"
    partial.mkdir()
    (partial / "data.pt").write_bytes(b"partial-write")
    with pytest.raises(ValueError):
        retire_checkpoints(tmp_path, partial, "same-training", 1)
    assert complete_checkpoint(old, "same-training")
    assert not complete_checkpoint(partial)


def test_retention_preserves_pins_foreign_signatures_and_partial_writes(tmp_path):
    first, second = snapshot(tmp_path, 1), snapshot(tmp_path, 2)
    third = snapshot(tmp_path, 3)
    foreign = snapshot(tmp_path, 4, "another-training")
    partial = tmp_path / "global_step_5"
    partial.mkdir()
    removed = retire_checkpoints(tmp_path, third, "same-training", 1, protected=[1])
    assert removed == ["global_step_2"]
    assert complete_checkpoint(first) and complete_checkpoint(third)
    assert foreign.is_dir() and partial.is_dir() and not second.exists()


def test_certificate_without_real_shards_is_not_a_complete_checkpoint(tmp_path):
    path = snapshot(tmp_path, 1)
    (path / "actor/model.pt").unlink()
    assert not complete_checkpoint(path)


def test_retention_never_follows_external_directory_links(tmp_path):
    root = tmp_path / "owned"
    root.mkdir()
    external = snapshot(tmp_path / "external", 1)
    (root / "global_step_1").symlink_to(external, target_is_directory=True)
    latest = snapshot(root, 2)
    assert retire_checkpoints(root, latest, "same-training", 1) == []
    assert complete_checkpoint(external)


def test_stop_between_intervals_saves_before_log_can_interrupt(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "raw").mkdir(parents=True)
    atomic_json(run / "manifest.json", {"training_signature": "same-training", "stop_after": 1,
                "checkpoint_policy": {"enabled": True, "keep_latest": None, "protected_steps": []}})
    atomic_json(run / "raw/group_step_000001.json", {"step": 1})
    events = []

    class Attributes(dict):
        __getattr__ = dict.__getitem__

    config = types.SimpleNamespace(trainer=Attributes(
        save_freq=2, default_local_dir=str(run / "checkpoints"),
        max_actor_ckpt_to_keep=None, remove_previous_ckpt_in_save=False))

    class Trainer:
        def __init__(self, config):
            self.config, self.global_steps = config, 1

        def _save_checkpoint(self):
            root = Path(self.config.trainer.default_local_dir)
            snapshot(root, self.global_steps)
            (root / "latest_checkpointed_iteration.txt").write_text("1\n", encoding="utf-8")
            events.append("saved")

    class Stop(Exception):
        pass

    class Tracking:
        def log(self, data, step, backend=None):
            events.append("logged")
            raise Stop()

    trainer_module = types.ModuleType("verl.trainer.ppo.ray_trainer")
    trainer_module.RayPPOTrainer = Trainer
    tracking_module = types.ModuleType("verl.utils.tracking")
    tracking_module.Tracking = Tracking
    monkeypatch.setitem(sys.modules, trainer_module.__name__, trainer_module)
    monkeypatch.setitem(sys.modules, tracking_module.__name__, tracking_module)
    monkeypatch.setenv("SCOR_RUN_DIR", str(run))
    install(config)
    Trainer(config)
    with pytest.raises(Stop):
        Tracking().log({}, 1)
    assert events == ["saved", "logged"]
    assert read_json(run / "raw/durable_checkpoint_step_000001.json")["model_state_committed"]
    assert complete_checkpoint(run / "checkpoints/global_step_1", "same-training")
