"""Record and control a native Agent0-VL baseline without replacing its learner."""
from __future__ import annotations

import copy
import functools
import os
from pathlib import Path
import random
import tempfile
import time

from pilot_utils import atomic_json, file_digest, read_json
from records import prompt_hash, stream_seed, tensor_save, value_hash


class IntentionalStop(Exception):
    pass


def solver_cue(feedback: str, succeeded: bool, repaired: bool) -> str:
    if succeeded and "No output" in feedback and not repaired:
        instruction = ("The tool ran without printing a result. Output fenced Python code that prints the calculation "
                       "with print(result). A bare Python expression does not print in this tool. "
                       "Use the actual printed result before giving your final answer.")
    elif succeeded and not repaired:
        instruction = ("The Python tool has already executed the calculation. Your next Solver response must be "
                       "only \\boxed{integer}, using the numeric result in the tool feedback. "
                       "Format example: for a result of 7, the complete reply is \\boxed{7}. "
                       "Replace 7 with this calculation's result and keep the backslash, boxed and braces. "
                       "Use decimal digits and an optional minus sign only, without commas or expressions. "
                       "Do not give a bare number, repeat the program or switch to another role.")
    else:
        instruction = ("Use the tool feedback and any repair. If a repaired calculation still needs execution "
                       "or execution failed, output fenced Python code; otherwise finish with \\boxed{integer}.")
    return ("\n<|im_start|>user\nNow return to the Solver role. " + instruction +
            "\nActual tool feedback for this step:\n" + (feedback or "No new tool feedback.") +
            "<|im_end|>\n<|im_start|>assistant\n")


def processor_options() -> dict:
    return read_json(Path(os.environ["BASELINE_RUN_DIR"]) / "manifest.json")["learning"]["processor_options"]


def record_processor(processor, *, rank=None, role="driver") -> None:
    expected = processor_options()
    if processor is None:
        raise ValueError("Native multimodal processor initialization failed")
    actual = {key: getattr(processor.image_processor, key) for key in expected}
    if actual != expected:
        raise ValueError("Native processor pixel budget differs from the declared candidate")
    if type(processor.image_processor).__name__.endswith("Fast"):
        raise ValueError("Use the native slow image processor shared with vLLM")
    run = Path(os.environ["BASELINE_RUN_DIR"])
    atomic_json(run / f"raw/processor_{role}_{rank}.json",
                {"expected": expected, "actual": actual, "use_fast": False,
                 "processor_type": type(processor.image_processor).__name__}, immutable=True)


def _deterministic_vision_call(native, *args, **kwargs):
    kwargs["deterministic"] = True
    return native(*args, **kwargs)


def vision_determinism(*, enable=False) -> dict:
    from importlib import import_module
    flags = {}
    for name in ("qwen2_vl", "qwen2_5_vl"):
        model = import_module(f"transformers.models.{name}.modeling_{name}")
        native = model.flash_attn_varlen_func
        if enable and not getattr(native, "_baseline_deterministic", False):
            wrapped = functools.wraps(native)(functools.partial(_deterministic_vision_call, native))
            wrapped._baseline_deterministic = True
            model.flash_attn_varlen_func = wrapped
        flags[name] = bool(getattr(model.flash_attn_varlen_func, "_baseline_deterministic", False))
    return flags


def plain(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported receipt value: {type(value).__name__}")


def rng_state():
    import numpy as np
    import torch
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def restore_driver_and_cursor(dataloader, checkpoint: Path) -> dict:
    import torch
    driver = torch.load(checkpoint / "baseline_driver_rng.pt", map_location="cpu", weights_only=False)
    cursor = torch.load(checkpoint / "data.pt", map_location="cpu", weights_only=False)
    # state_dict materializes a restored lazy iterator and draws its base seed.
    # Prepare that iterator before the final restoration of the driver stream.
    loaded_cursor = value_hash(dataloader.state_dict())
    restore_rng(driver)
    driver_hash, cursor_hash = value_hash(driver), value_hash(cursor)
    if value_hash(rng_state()) != driver_hash or loaded_cursor != cursor_hash:
        raise ValueError("Restored driver RNG or dataloader cursor differs")
    return {"driver_rng_sha256": driver_hash, "dataloader_sha256": cursor_hash}


def seed_worker(rank: int, role: str, config) -> None:
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    run = Path(os.environ["BASELINE_RUN_DIR"])
    manifest = read_json(run / "manifest.json")
    seed = stream_seed(manifest["seed"], "worker_init", 0, str(rank), 0, role, 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    atomic_json(run / f"raw/worker_config_rank_{rank}_{role}.json",
                {"rank": rank, "role": role, "seed": seed,
                 "flash_attention_deterministic": os.environ.get("FLASH_ATTENTION_DETERMINISTIC"),
                 "vision_attention_deterministic": vision_determinism(),
                 "normalized_config": OmegaConf.to_container(config, resolve=True)}, immutable=True)


def generate_role(rollout, inputs, params, indices, role, inner, prompts, metadata):
    import torch
    run = Path(os.environ["BASELINE_RUN_DIR"])
    manifest = read_json(run / "manifest.json")
    phase = prompts.meta_info["baseline_phase"]
    step = int(prompts.meta_info["baseline_step"])
    rank = torch.distributed.get_rank()
    request_params, requests = [], []
    for g, entry in zip(indices, inputs):
        case_id, slot = str(metadata["baseline_case_id"][g]), int(metadata["baseline_slot"][g])
        selected = copy.deepcopy(params)
        selected.seed = stream_seed(manifest["seed"], phase, step, case_id, slot, role, inner)
        request_params.append(selected)
        requests.append({"case_id": case_id, "slot": slot, "input_token_ids": entry["prompt_token_ids"],
                         "visual_input_hash": value_hash(entry.get("multi_modal_data")),
                         "sampling": {k: getattr(selected, k) for k in
                                      ("seed", "temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "n")}})
    begin = time.monotonic()
    outputs = rollout.inference_engine.generate(prompts=inputs, sampling_params=request_params, use_tqdm=False)
    for request, output in zip(requests, outputs):
        sample = output.outputs[0]
        request.update({"output_token_ids": list(sample.token_ids), "finish_reason": sample.finish_reason,
                        "returned_token_logp": [float(lp[token].logprob) for token, lp in
                                                zip(sample.token_ids, sample.logprobs)] if sample.logprobs is not None else None})
    counter = getattr(rollout, "_baseline_call", 0)
    rollout._baseline_call = counter + 1
    rollout._baseline_context = {"step": step, "phase": phase, "role": role, "inner_step": inner}
    atomic_json(run / f"raw/roles/{phase}_step_{step:06d}_rank_{rank}_call_{counter:06d}.json",
                {"rank": rank, "step": step, "phase": phase, "role": role, "inner_step": inner,
                 "wall_seconds": time.monotonic() - begin, "requests": requests,
                 "logp_source": "pinned vLLM returned token logprobs; sampler equivalence separately audited"}, immutable=True)
    return outputs


def _worker_rng(self, action):
    if action == "capture":
        self._baseline_rng_backup = rng_state()
    elif action == "restore":
        restore_rng(self._baseline_rng_backup)
    else:
        raise ValueError("Unknown RNG action")
    return value_hash(rng_state())


def _worker_state(self):
    import torch
    parameters = dict(self.actor.actor_module.named_parameters())
    model, optimizer, buffers = {}, {}, {}
    for name, parameter in parameters.items():
        cpu = parameter.detach().cpu()
        if not torch.isfinite(cpu).all():
            raise FloatingPointError("Non-finite model state")
        model[name] = value_hash(cpu)
        state = self.actor.actor_optimizer.state.get(parameter, {})
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError("Non-finite optimizer state")
        optimizer[name] = value_hash(state)
    for prefix, module in self.actor.actor_module.named_modules():
        for name, buffer in module._buffers.items():
            if buffer is None or name in module._non_persistent_buffers_set:
                continue
            if buffer.is_floating_point() and not torch.isfinite(buffer).all():
                raise FloatingPointError("Non-finite persistent model buffer")
            buffers[(prefix + "." if prefix else "") + name] = value_hash(buffer)
    return {"rank": torch.distributed.get_rank(),
            "model": model, "buffers": buffers, "optimizer": optimizer,
            "all_parameters_and_optimizer_finite": True,
            "optimizer_groups": [{k: plain(v) for k, v in group.items() if k != "params"}
                                 for group in self.actor.actor_optimizer.param_groups],
            "scheduler": plain(self.actor_lr_scheduler.state_dict()), "rng_hash": value_hash(rng_state())}


def _patch_workers():
    import torch
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.workers.rollout.vllm_rollout.vllm_agent0_rollout_spmd import vLLMAgent0Rollout
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.third_party.vllm import LLM
    if getattr(DataParallelPPOActor, "_baseline_patched", False):
        return
    if os.environ.get("FLASH_ATTENTION_DETERMINISTIC") == "1":
        vision_determinism(enable=True)
    ActorRolloutRefWorker.baseline_rng = register(dispatch_mode=Dispatch.ONE_TO_ALL)(_worker_rng)
    ActorRolloutRefWorker.baseline_state = register(dispatch_mode=Dispatch.ONE_TO_ALL)(_worker_state)
    original_init, original_wake = LLM.__init__, LLM.wake_up

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        kwargs.update({"enable_prefix_caching": False, "max_num_seqs": 16})
        options = processor_options()
        if options:
            kwargs["mm_processor_kwargs"] = options
        result = original_init(self, *args, **kwargs)
        actual = self.llm_engine.model_config.mm_processor_kwargs or {}
        if options and actual != options:
            raise ValueError("vLLM processor budget differs from Torch preprocessing")
        atomic_json(Path(os.environ["BASELINE_RUN_DIR"]) / f"raw/vllm_processor_rank_{torch.distributed.get_rank()}.json",
                    {"expected": options, "actual": actual}, immutable=True)
        return result

    @functools.wraps(original_wake)
    def wake(self, *args, **kwargs):
        torch.cuda.empty_cache()
        return original_wake(self, *args, **kwargs)

    LLM.__init__, LLM.wake_up = initialize, wake
    original_step, original_update = DataParallelPPOActor._optimizer_step, ActorRolloutRefWorker.update_actor

    def probe(module):
        values = []
        for p in module.parameters():
            if p.requires_grad and p.numel():
                flat = p.detach().reshape(-1)
                count = min(64, flat.numel())
                indices = torch.arange(count, device=flat.device) * (flat.numel() - 1) // max(count - 1, 1)
                values.append(flat[indices].float().cpu().clone())
        return torch.cat(values) if values else torch.empty(0)

    @functools.wraps(original_step)
    def optimizer_step(self):
        before = probe(self.actor_module)
        used_lr = float(self.actor_optimizer.param_groups[0]["lr"])
        norm = original_step(self)
        if not torch.isfinite(torch.as_tensor(norm)).all():
            raise FloatingPointError("Non-finite native gradient norm")
        after = probe(self.actor_module)
        steps = [float(s["step"]) for s in self.actor_optimizer.state.values() if "step" in s]
        self._baseline_probe = {"baseline/parameter_probe_changed": float(bool(before.numel() and not torch.equal(before, after))),
                                "baseline/parameter_probe_finite": float(torch.isfinite(after).all()),
                                "baseline/optimizer_step": max(steps) if steps else 0.0,
                                "baseline/lr_used": used_lr,
                                "baseline/cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                "baseline/cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        return norm

    @functools.wraps(original_update)
    def update(self, data):
        result = original_update(self, data)
        result.meta_info["metrics"].update(self.actor._baseline_probe)
        return result

    DataParallelPPOActor._optimizer_step, ActorRolloutRefWorker.update_actor = optimizer_step, update
    native_tool = vLLMAgent0Rollout._execute_code_in_sandbox

    @functools.wraps(native_tool)
    def tool(self, codes):
        begin = time.monotonic()
        result = native_tool(self, codes)
        run = Path(os.environ["BASELINE_RUN_DIR"])
        context = dict(self._baseline_context)
        counter = getattr(self, "_baseline_tool_call", 0)
        self._baseline_tool_call = counter + 1
        rank = torch.distributed.get_rank()
        atomic_json(run / f"raw/tools/{context['phase']}_step_{context['step']:06d}_rank_{rank}_call_{counter:06d}.json",
                    {**context, "rank": rank, "code": codes, "results": plain(result),
                     "wall_seconds": time.monotonic() - begin}, immutable=True)
        return result

    vLLMAgent0Rollout._execute_code_in_sandbox = tool
    DataParallelPPOActor._baseline_patched = True


def install(config):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.tracking import Tracking
    from checkpoint_store import certify, commit, retire, validate
    from evaluate import aggregate, case_result
    if getattr(RayPPOTrainer, "_baseline_patched", False):
        return
    _patch_workers()
    run = Path(os.environ["BASELINE_RUN_DIR"])
    manifest = read_json(run / "manifest.json")
    learning = manifest["signatures"]["learning"]
    random.seed(manifest["seed"])
    np.random.seed(manifest["seed"])
    torch.manual_seed(manifest["seed"])
    atomic_json(run / "raw/resolved_config.json", OmegaConf.to_container(config, resolve=True), immutable=True)
    originals = {name: getattr(RayPPOTrainer, name) for name in
                 ("__init__", "init_workers", "_save_checkpoint", "_load_checkpoint", "fit")}
    original_log = Tracking.log
    current = {}

    def initialize(self, *args, **kwargs):
        originals["__init__"](self, *args, **kwargs)
        current["trainer"] = self
        self._baseline_zero_signal = 0
        self._baseline_snapshots = set()
        self._baseline_evaluations = {}
        native_reward = self.reward_fn

        def reward(data, return_dict=False):
            result = native_reward(data, return_dict=True)
            original = result["reward_tensor"].clone()
            revision = {"status": "none", "step": self.global_steps}
            if manifest.get("injection_step") == self.global_steps:
                from interventions import false_positive
                result["reward_tensor"], revision = false_positive(
                    data, original, result["reward_extra_info"]["acc"], step=self.global_steps,
                    alpha_out=float(self.config.reward_model.alpha_out), delta=manifest.get("reward_delta", 1))
            atomic_json(run / f"raw/reward_step_{self.global_steps:06d}.json",
                        {"step": self.global_steps, "native_returns": original.sum(-1).tolist(),
                         "used_returns": result["reward_tensor"].sum(-1).tolist(),
                         "native_info": plain(result["reward_extra_info"]), "revision": revision,
                         "group_ids": plain(data.non_tensor_batch["uid"])}, immutable=True)
            return result if return_dict else result["reward_tensor"]

        self.reward_fn = reward

    def init_workers(self):
        originals["init_workers"](self)
        native_generate, native_update = self.actor_rollout_wg.generate_sequences, self.actor_rollout_wg.update_actor

        def generate(prompts):
            if prompts.meta_info.get("validate"):
                return native_generate(prompts)
            step = self.global_steps
            identity = prompt_hash(prompts)
            replay = manifest.get("replay")
            if replay and replay["start"] <= step <= replay["end"]:
                from replay import load_generation
                output = load_generation(Path(replay["source"]), step=step, learning=learning, prompts=prompts)
            else:
                output = native_generate(prompts)
            tensor_save({"step": step, "learning_signature": learning, "prompt_hash": identity,
                         "output": copy.deepcopy(output).to("cpu")}, run / f"raw/generation_step_{step:06d}.pt")
            return output

        def update(data):
            step = self.global_steps
            mask = data.batch.get("multiturn_mask", data.batch["response_mask"]).bool()
            advantage = data.batch["advantages"]
            if not torch.isfinite(advantage).all() or not mask.any():
                raise ValueError("Non-finite advantage or empty update mask")
            ids = plain(data.non_tensor_batch["uid"])
            groups = set(ids)
            n = int(self.config.actor_rollout_ref.rollout.n)
            if any(ids.count(group) != n for group in groups):
                raise ValueError("Native update has incomplete GRPO groups")
            nonzero = int((advantage[mask] != 0).sum())
            self._baseline_zero_signal = self._baseline_zero_signal + 1 if nonzero == 0 else 0
            tensor_save(copy.deepcopy(data).to("cpu"), run / f"raw/training_group_step_{step:06d}.pt")
            atomic_json(run / f"raw/group_step_{step:06d}.json",
                        {"step": step, "trajectories": len(data), "groups": len(groups),
                         "group_ids": ids, "trainable_tokens": int(mask.sum()),
                         "nonzero_advantage_tokens": nonzero, "zero_signal_streak": self._baseline_zero_signal,
                         "payload_hash": prompt_hash(data)}, immutable=True)
            if self._baseline_zero_signal >= 3:
                raise ValueError("Three consecutive zero-advantage units; pause for calibration")
            return native_update(data)

        self.actor_rollout_wg.generate_sequences, self.actor_rollout_wg.update_actor = generate, update

    def evaluate_once(self):
        from verl import DataProto
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        driver = rng_state()
        before_cursor = value_hash(self.train_dataloader.state_dict())
        worker_before = self.actor_rollout_wg.baseline_rng("capture")
        cases = []
        begin = time.monotonic()
        try:
            for batch in self.val_dataloader:
                full = DataProto.from_single_dict(batch)
                for start in range(0, len(full), 16):
                    chunk = full[start:start + 16]
                    names = ["raw_prompt_ids"]
                    if "multi_modal_inputs" in chunk.non_tensor_batch:
                        names += ["multi_modal_data", "multi_modal_inputs"]
                    prompts = chunk.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=names)
                    prompts.non_tensor_batch["baseline_case_id"] = np.array(
                        [info["case_id"] for info in chunk.non_tensor_batch["extra_info"]], dtype=object)
                    prompts.meta_info.update({"validate": True, "do_sample": False,
                        "baseline_step": self.global_steps, "baseline_phase": "eval_" + manifest["evaluation_split"]})
                    padded, size = pad_dataproto_to_divisor(prompts, self.actor_rollout_wg.world_size)
                    generated = unpad_dataproto(self.actor_rollout_wg.generate_sequences(padded), pad_size=size)
                    chunk = chunk.union(generated)
                    native = self.val_reward_fn(chunk, return_dict=True)
                    for i in range(len(chunk)):
                        cases.append(case_result(chunk.non_tensor_batch["extra_info"][i],
                            chunk.non_tensor_batch["baseline_segments"][i], native["reward_extra_info"]["acc"][i],
                            chunk.non_tensor_batch["step_data"][i]))
            from probe import evaluate_verifier
            verifier_probes = evaluate_verifier(self, manifest)
        finally:
            worker_after = self.actor_rollout_wg.baseline_rng("restore")
            restore_rng(driver)
        after_cursor = value_hash(self.train_dataloader.state_dict())
        if worker_before != worker_after or value_hash(rng_state()) != value_hash(driver) or after_cursor != before_cursor:
            raise ValueError("Evaluation changed training RNG or data cursor")
        summary = aggregate(cases)
        count = self._baseline_evaluations.get(self.global_steps, 0)
        suffix = "" if count == 0 else f"_repeat_{count:02d}"
        atomic_json(run / f"raw/evaluation_step_{self.global_steps:06d}{suffix}.json",
                    {"step": self.global_steps, "split": manifest["evaluation_split"], "cases": cases,
                     "verifier_probes": verifier_probes,
                     "summary": summary, "worker_rng_restored": True, "driver_rng_restored": True,
                     "worker_rng_before": worker_before, "worker_rng_after": worker_after,
                     "driver_rng_before": value_hash(driver), "driver_rng_after": value_hash(rng_state()),
                     "data_cursor_before": before_cursor, "data_cursor_after": after_cursor,
                     "data_cursor_unchanged": True, "wall_seconds": time.monotonic() - begin}, immutable=True)
        self._baseline_evaluations[self.global_steps] = count + 1
        return {"baseline/evaluation_accuracy": summary["accuracy"],
                "baseline/explicit_answer_rate": summary["explicit_answer_rate"],
                "baseline/verifier_parse_rate": verifier_probes["summary"]["parse_rate"]}

    def evaluate(self):
        initial = manifest["checkpoint_plan"]["start"]
        first = self._baseline_evaluations.get(self.global_steps, 0) == 0
        metrics = evaluate_once(self)
        if first and self.global_steps == initial and manifest["protocol"].get("repeat_initial_evaluation"):
            metrics = evaluate_once(self)
        return metrics

    def save(self):
        root = run / "checkpoints"
        destination = root / f"global_step_{self.global_steps}"
        if destination.exists():
            validate(destination, learning)
            return
        import shutil
        if shutil.disk_usage(root).free < manifest["checkpoint_estimate_bytes"] + 2 * 2**30:
            raise OSError("Free space insufficient for full state plus save margin")
        container = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
        old = self.config.trainer.default_local_dir
        self.config.trainer.default_local_dir = str(container)
        begin = time.monotonic()
        worker_state = self.actor_rollout_wg.baseline_state()
        try:
            originals["_save_checkpoint"](self)
        finally:
            self.config.trainer.default_local_dir = old
        pending = container / destination.name
        tensor_save(rng_state(), pending / "baseline_driver_rng.pt")
        atomic_json(pending / "baseline_state.json",
                    {"completed_step": self.global_steps, "next_step": self.global_steps + 1,
                     "full_horizon": self.total_training_steps, "zero_signal_streak": self._baseline_zero_signal,
                     "sampling_stream": "seed,phase,step,case,slot,role,inner; no branch name",
                     "parent": manifest.get("parent")}, immutable=True)
        atomic_json(pending / "baseline_worker_state.json", plain(worker_state), immutable=True)
        certificate = certify(pending, learning=learning, run_id=manifest["run_id"],
            step=self.global_steps, world=int(self.config.trainer.n_gpus_per_node), parent=manifest.get("parent"))
        commit(pending, destination, proof=certificate)
        self._baseline_snapshots.add(self.global_steps)
        # Only this run's validated unprotected snapshots are eligible for retirement.
        deleted = retire(root, destination, learning, manifest["run_id"],
                         manifest["checkpoint_plan"]["keep"], manifest["checkpoint_plan"]["protected"], proof=certificate)
        atomic_json(run / f"raw/checkpoint_step_{self.global_steps:06d}.json",
                    {"step": self.global_steps, "state_signature": certificate["state_signature"],
                     "snapshot_bytes": sum(x["bytes"] for x in certificate["files"].values()),
                     "retired": deleted, "wall_seconds": time.monotonic() - begin}, immutable=True)

    def load(self):
        if self.config.trainer.resume_mode == "disable":
            originals["_load_checkpoint"](self)
            self._save_checkpoint()  # Actual complete S[0], before evaluation or data consumption.
        else:
            path = Path(self.config.trainer.resume_from_path)
            certificate = validate(path, learning)
            originals["_load_checkpoint"](self)
            driver_state = restore_driver_and_cursor(self.train_dataloader, path)
            state = read_json(path / "baseline_state.json")
            worker_loaded = plain(self.actor_rollout_wg.baseline_state())
            if worker_loaded != read_json(path / "baseline_worker_state.json"):
                raise ValueError("Restored model/optimizer/scheduler/worker RNG fingerprints differ")
            self._baseline_zero_signal = state["zero_signal_streak"]
            if self.global_steps != certificate["step"] or self.total_training_steps != state["full_horizon"]:
                raise ValueError("Restored position/horizon differs")
            atomic_json(run / "raw/resume_loaded.json",
                        {"parent_state": certificate["state_signature"], "step": self.global_steps,
                         "model_optimizer_scheduler_worker_rng_loaded": True,
                         "driver_rng_restored": True, "dataloader_loaded": True,
                         "full_worker_state_fingerprints_match": True,
                         **driver_state,
                         "sampling_rng_bitwise_verified": False}, immutable=True)

    def log(self, data, step, backend=None):
        trainer = current["trainer"]
        group = run / f"raw/group_step_{step:06d}.json"
        if group.exists():
            if step in manifest["checkpoint_plan"]["scheduled_steps"] and step not in trainer._baseline_snapshots:
                trainer._save_checkpoint()
            if manifest.get("stop_after") == step and not (run / f"raw/evaluation_step_{step:06d}.json").exists():
                data = {**data, **trainer._validate()}
        original_log(self, data, step, backend)
        atomic_json(run / f"raw/metrics_step_{step:06d}.json", plain(data), immutable=True)
        if group.exists():
            atomic_json(run / f"raw/completed_step_{step:06d}.json",
                        {"step": step, "learning_signature": learning, "group_sha256": file_digest(group),
                         "model_snapshot_committed": step in trainer._baseline_snapshots}, immutable=True)
            if manifest.get("stop_after") == step:
                atomic_json(run / "raw/intentional_stop.json", {"step": step}, immutable=True)
                raise IntentionalStop()

    def fit(self):
        try:
            return originals["fit"](self)
        except IntentionalStop:
            return None

    RayPPOTrainer.__init__, RayPPOTrainer.init_workers = initialize, init_workers
    RayPPOTrainer._validate, RayPPOTrainer._save_checkpoint, RayPPOTrainer._load_checkpoint = evaluate, save, load
    RayPPOTrainer.fit, Tracking.log = fit, log
    RayPPOTrainer._baseline_patched = True


if os.environ.get("BASELINE_RUN_DIR"):
    _patch_workers()
