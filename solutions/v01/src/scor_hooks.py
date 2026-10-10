"""Instrument native Agent0-VL; its rollout, reward and updater remain in use."""
from __future__ import annotations

import copy
import functools
import json
import os
from pathlib import Path
import random
import tempfile

from pilot_utils import atomic_json, file_digest, read_json


class IntentionalStop(Exception):
    """Stop only this native trainer after its checkpoint and receipts are complete."""


def torch_save_immutable(value, path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="pending-", suffix=".pt", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)


def plain(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def apply_corruption(tensor, attention_mask, row: int, delta: float):
    """Perturb one sequence return without modifying the original reward tensor."""
    if not 0 <= row < len(tensor):
        raise ValueError("Corruption row outside the training batch")
    result = tensor.clone()
    positions = attention_mask[row, -tensor.shape[-1]:].bool().nonzero().flatten()
    if not positions.numel():
        raise ValueError("Cannot inject a reward into an empty trajectory")
    result[row, positions[-1]] += delta
    return result


def sample_parameters(module):
    """Read bounded samples using exact integer indices, including large shards."""
    import torch
    result = []
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.numel():
            flat = parameter.detach().reshape(-1)
            count = min(64, flat.numel())
            indices = torch.arange(count, device=flat.device, dtype=torch.long)
            indices = indices * (flat.numel() - 1) // max(count - 1, 1)
            result.append(flat[indices].float().cpu().clone())
    return torch.cat(result) if result else torch.empty(0)


def _patch_workers() -> None:
    import torch
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.third_party.vllm import LLM
    if getattr(DataParallelPPOActor, "_scor_instrumented", False):
        return
    native_llm_init = LLM.__init__

    @functools.wraps(native_llm_init)
    def llm_init(self, *args, **kwargs):
        # Keep prefix reuse disabled while validating repeated-image GRPO
        # requests; every visual prefix is computed afresh.
        kwargs["enable_prefix_caching"] = False
        kwargs["max_num_seqs"] = 16
        return native_llm_init(self, *args, **kwargs)

    LLM.__init__ = llm_init
    optimizer_step = DataParallelPPOActor._optimizer_step

    @functools.wraps(optimizer_step)
    def observed_optimizer_step(self):
        before = sample_parameters(self.actor_module)
        norm = optimizer_step(self)
        after = sample_parameters(self.actor_module)
        changed = bool(before.numel() and not torch.equal(before, after))
        self._scor_parameter_probe = {"scor/parameter_sample_changed": float(changed),
                                     "scor/parameter_sample_finite": float(torch.isfinite(after).all()),
                                     "scor/parameter_sample_size": float(after.numel())}
        return norm

    native_update = ActorRolloutRefWorker.update_actor

    @functools.wraps(native_update)
    def observed_update(self, data):
        output = native_update(self, data)
        output.meta_info["metrics"].update(self.actor._scor_parameter_probe)
        return output

    DataParallelPPOActor._optimizer_step = observed_optimizer_step
    ActorRolloutRefWorker.update_actor = observed_update
    DataParallelPPOActor._scor_instrumented = True
    native_wake = LLM.wake_up

    @functools.wraps(native_wake)
    def wake(self, *args, **kwargs):
        before = {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                  "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3)}
        # Native state_dict() can leave unused allocator blocks after its initial
        # empty_cache(). Release only inactive blocks before restoring vLLM memory.
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print("SCOR_WAKE_MEMORY " + json.dumps({"rank": os.environ.get("RANK"),
              "before": before, "free_after_empty_cache_gib": round(free / 2**30, 3),
              "total_gib": round(total / 2**30, 3)}), flush=True)
        return native_wake(self, *args, **kwargs)

    LLM.wake_up = wake


def install(config) -> None:
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.tracking import Tracking
    if getattr(RayPPOTrainer, "_scor_instrumented", False):
        return
    run = Path(os.environ["SCOR_RUN_DIR"])
    manifest = read_json(run / "manifest.json")
    signature = manifest["training_signature"]
    random.seed(manifest["seed"])
    np.random.seed(manifest["seed"])
    torch.manual_seed(manifest["seed"])
    atomic_json(run / "raw/resolved_config.json", OmegaConf.to_container(config, resolve=True), immutable=True)
    original_init = RayPPOTrainer.__init__
    original_save = RayPPOTrainer._save_checkpoint
    original_load = RayPPOTrainer._load_checkpoint
    original_init_workers = RayPPOTrainer.init_workers
    original_fit = RayPPOTrainer.fit
    original_log = Tracking.log
    current = {}

    @functools.wraps(original_init)
    def trainer_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        current["trainer"] = self
        native_reward = self.reward_fn
        native_validation = self.val_reward_fn

        def validation(data, return_dict=False):
            result = native_validation(data, return_dict=True)
            accuracies = plain(result["reward_extra_info"].get("acc", []))
            atomic_json(run / f"raw/validation_step_{self.global_steps:06d}.json",
                        {"step": self.global_steps, "examples": len(data),
                         "outcome_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
                         "correctness": accuracies,
                         "final_answers": plain(data.non_tensor_batch.get("final_answers", [])),
                         "composite_returns": result["reward_tensor"].sum(-1).tolist()}, immutable=True)
            return result if return_dict else result["reward_tensor"]

        def reward(data, return_dict=False):
            # Always call the original native reward computation first.
            result = native_reward(data, return_dict=True)
            tensor = result["reward_tensor"].clone()
            before = tensor.sum(-1).tolist()
            step = self.global_steps
            corruption = manifest["corruption"]
            applied = corruption["step"] == step
            if applied:
                tensor = apply_corruption(tensor, data.batch["attention_mask"],
                                          corruption["row"], corruption["delta"])
            result["reward_tensor"] = tensor
            receipt = {"step": step, "original_returns": before,
                       "used_returns": tensor.sum(-1).tolist(), "corruption_applied": applied,
                       "group_ids": plain(data.non_tensor_batch["uid"]),
                       "final_answers": plain(data.non_tensor_batch.get("final_answers", [])),
                       "steps": plain(data.non_tensor_batch.get("step_data", [])),
                       "native_reward_metrics": plain(result.get("metrics", {}))}
            # A repeat call is a failure, not an opportunity to overwrite raw evidence.
            atomic_json(run / f"raw/reward_step_{step:06d}.json", receipt, immutable=True)
            return result if return_dict else tensor

        self.reward_fn = reward
        self.val_reward_fn = validation

    @functools.wraps(original_init_workers)
    def init_workers(self):
        original_init_workers(self)
        native_generate = self.actor_rollout_wg.generate_sequences
        native_update = self.actor_rollout_wg.update_actor

        def generate(prompts):
            if prompts.meta_info.get("validate", False):
                return native_generate(prompts)
            step = self.global_steps
            replay = manifest.get("replay_generation")
            input_ids = prompts.batch["input_ids"].cpu()
            if replay and replay["step"] == step:
                payload = torch.load(Path(replay["run_dir"]) / f"raw/generation_step_{step:06d}.pt",
                                     map_location="cpu", weights_only=False)
                if payload["training_signature"] != signature:
                    raise ValueError("Replayed generation signature mismatch")
                if not torch.equal(payload["prompt_input_ids"], input_ids):
                    raise ValueError("Replay prompt batch differs from the recorded batch")
                generated = payload["output"]
            else:
                generated = native_generate(prompts)
            saved = {"prompt_input_ids": input_ids, "output": copy.deepcopy(generated).to("cpu"),
                     "step": step, "training_signature": signature}
            torch_save_immutable(saved, run / f"raw/generation_step_{step:06d}.pt")
            return generated

        def update(data):
            step = self.global_steps
            advantage = data.batch["advantages"]
            active = data.batch.get("multiturn_mask", data.batch["response_mask"]).bool()
            if not torch.isfinite(advantage).all():
                raise ValueError("Non-finite advantage; refusing the native update")
            active_advantage = advantage[active]
            payload = copy.deepcopy(data).to("cpu")
            torch_save_immutable(payload, run / f"raw/training_group_step_{step:06d}.pt")
            atomic_json(run / f"raw/group_step_{step:06d}.json",
                        {"step": step, "trajectories": len(data),
                         "nonzero_advantage_tokens": int((active_advantage != 0).sum()),
                         "trainable_tokens": int(active.sum()),
                         "old_log_probs_present": "old_log_probs" in data.batch,
                         "sample_indices": plain(data.non_tensor_batch.get("index", [])),
                         "optimizer_update_requested": True}, immutable=True)
            return native_update(data)

        self.actor_rollout_wg.generate_sequences = generate
        self.actor_rollout_wg.update_actor = update

    @functools.wraps(original_save)
    def save(self):
        directory = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite a checkpoint: {directory}")
        # Fail before writing if the shared disk cannot hold the expected checkpoint.
        import shutil
        if shutil.disk_usage(directory.parent).free < 50 * 2**30:
            raise OSError("Need at least 50 GiB free before saving another complete 3B checkpoint")
        original_save(self)
        required = list((directory / "actor").glob("*.pt"))
        world_size = int(self.config.trainer.n_gpus_per_node)
        if len(required) != world_size * 3 or not (directory / "data.pt").exists():
            raise RuntimeError("Native checkpoint missing model/optimizer/extra/dataloader shards")
        for path in required + [directory / "data.pt"]:
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        driver = {"python": random.getstate(), "numpy": np.random.get_state(),
                  "torch": torch.get_rng_state()}
        torch_save_immutable(driver, directory / "scor_driver_rng.pt")
        atomic_json(directory / "scor_complete.json",
                    {"step": self.global_steps, "training_signature": signature,
                     "world_size": world_size, "worker_shards": [str(p.relative_to(directory)) for p in required],
                     "sampling_rng_bitwise_verified": False}, immutable=True)

    @functools.wraps(original_load)
    def load(self):
        mode = self.config.trainer.resume_mode
        if mode != "disable":
            path = Path(self.config.trainer.resume_from_path)
            certificate = read_json(path / "scor_complete.json")
            if certificate["training_signature"] != signature:
                raise ValueError("Checkpoint training configuration or data signature mismatch")
            if certificate["world_size"] != int(self.config.trainer.n_gpus_per_node):
                raise ValueError("Checkpoint GPU world size mismatch")
        result = original_load(self)
        if mode != "disable":
            driver = torch.load(path / "scor_driver_rng.pt", map_location="cpu", weights_only=False)
            random.setstate(driver["python"])
            np.random.set_state(driver["numpy"])
            torch.set_rng_state(driver["torch"])
            atomic_json(run / "raw/resume_loaded.json",
                        {"checkpoint": str(path), "step": self.global_steps,
                         "training_signature": signature, "driver_rng_restored": True,
                         "worker_model_optimizer_scheduler_rng_loaded": True,
                         "dataloader_restored": True, "sampling_rng_bitwise_verified": False}, immutable=True)
        return result

    @functools.wraps(original_log)
    def log(self, data, step, backend=None):
        result = original_log(self, data, step, backend)
        atomic_json(run / f"raw/metrics_step_{step:06d}.json", plain(data), immutable=True)
        group_path = run / f"raw/group_step_{step:06d}.json"
        if group_path.exists():
            atomic_json(run / f"raw/completed_step_{step:06d}.json",
                        {"step": step, "training_signature": signature,
                         "group_sha256": file_digest(group_path)}, immutable=True)
            if manifest.get("stop_after") == step:
                atomic_json(run / "raw/intentional_stop.json", {"step": step}, immutable=True)
                raise IntentionalStop()
        return result

    @functools.wraps(original_fit)
    def fit(self):
        try:
            return original_fit(self)
        except IntentionalStop:
            return None

    RayPPOTrainer.__init__ = trainer_init
    RayPPOTrainer.init_workers = init_workers
    RayPPOTrainer._save_checkpoint = save
    RayPPOTrainer._load_checkpoint = load
    RayPPOTrainer.fit = fit
    Tracking.log = log
    RayPPOTrainer._scor_instrumented = True


# The official model.external_lib hook imports this in GPU workers as well.
if os.environ.get("SCOR_RUN_DIR") or "RANK" in os.environ:
    _patch_workers()
