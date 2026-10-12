"""Apply exact recording/identity patches to a separate pinned native checkout."""
from __future__ import annotations

from pathlib import Path
import subprocess

from apply_instrumentation import AFTER, apply
from pilot_utils import SRC, UPSTREAM_COMMIT, atomic_json, file_digest


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError("Pinned native patch location differs; inspect before changing it")
    return text.replace(before, after)


def prepare(checkout: Path, original: Path) -> dict:
    if not checkout.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(checkout), UPSTREAM_COMMIT], cwd=original, check=True)
    marker = checkout / ".baseline_patch.json"
    engine = {name: file_digest(SRC / name) for name in
              ("apply_baseline_instrumentation.py", "apply_instrumentation.py")}
    if marker.exists():
        from pilot_utils import read_json
        manifest = read_json(marker)
        if manifest["patch_sources"] != engine or any(
                file_digest(checkout / name) != sha for name, sha in manifest["patched_files"].items()):
            raise ValueError("Existing baseline checkout identity differs; use a new checkout")
        return manifest
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True).strip():
        raise ValueError("Baseline patch requires a clean separate checkout")
    apply(checkout)  # Existing verified padding/GRPO metric adaptations, on the new copy only.
    prefix = "Agent0-VL/verl/"
    paths = {name: checkout / (prefix + name) for name in
             ("trainer/main_ppo.py", "trainer/ppo/ray_trainer.py", "workers/fsdp_workers.py",
              "workers/rollout/vllm_rollout/vllm_agent0_rollout_spmd.py")}
    main = paths["trainer/main_ppo.py"]
    main.write_text(replace_once(main.read_text(encoding="utf-8"), AFTER,
        "    def run(self, config):\n        from native_adapter import install\n"
        "        install(config)\n        from verl.utils.fs import copy_to_local\n"), encoding="utf-8")
    main.write_text(replace_once(main.read_text(encoding="utf-8"),
        "        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none",
        "        from native_adapter import processor_options, record_processor\n"
        "        processor = hf_processor(local_path, use_fast=False, **processor_options())\n"
        "        record_processor(processor)"), encoding="utf-8")
    trainer = paths["trainer/ppo/ray_trainer.py"]
    text = trainer.read_text(encoding="utf-8")
    text = replace_once(text, "                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)",
        "                        gen_batch.non_tensor_batch['baseline_case_id'] = np.array(\n"
        "                            [info['case_id'] for info in batch.non_tensor_batch['extra_info']], dtype=object)\n"
        "                        gen_batch.meta_info.update({'baseline_step': self.global_steps, 'baseline_phase': 'train'})\n"
        "                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)")
    text = replace_once(text,
        "                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],\n"
        "                                                             dtype=object)",
        "                    batch.non_tensor_batch['uid'] = np.array(\n"
        "                        [str(self.global_steps) + ':' + info['case_id']\n"
        "                         for info in batch.non_tensor_batch['extra_info']], dtype=object)")
    trainer.write_text(text, encoding="utf-8")
    worker = paths["workers/fsdp_workers.py"]
    text = worker.read_text(encoding="utf-8")
    text = replace_once(text, "        assert role in ['actor', 'ref']\n",
        "        assert role in ['actor', 'ref']\n"
        "        from native_adapter import seed_worker\n"
        "        seed_worker(self.rank, role, self.config)\n")
    text = replace_once(text, "        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)",
        "        from native_adapter import processor_options, record_processor\n"
        "        self.processor = hf_processor(local_path, use_fast=False, trust_remote_code=trust_remote_code, **processor_options())\n"
        "        record_processor(self.processor, rank=self.rank, role=role)")
    worker.write_text(text, encoding="utf-8")
    rollout = paths["workers/rollout/vllm_rollout/vllm_agent0_rollout_spmd.py"]
    text = rollout.read_text(encoding="utf-8")
    text = replace_once(text, "        verify_kwargs = dict(sample_kwargs)\n",
        "        non_tensor_batch['baseline_slot'] = np.arange(batch_size) % n_repeat\n"
        "        role_segments = [[] for _ in range(batch_size)]\n"
        "        active_role, call_truncated = 'solver', {}\n"
        "        verify_kwargs = dict(sample_kwargs)\n")
    text = replace_once(text,
        "            # --- Step 2: Execute tools for still-active samples ---",
        "            if prompts.meta_info.get('baseline_probe_role'):\n                break\n\n"
        "            # --- Step 2: Execute tools for still-active samples ---")
    text = replace_once(text,
        "        def _append_tokens(global_idx: int, tokens: List[int], trainable: bool) -> int:",
        "        def _append_tokens(global_idx: int, tokens: List[int], trainable: bool, role=None) -> int:")
    text = replace_once(text, "            tokens = tokens[:room]\n",
        "            original_length = len(tokens)\n            tokens = tokens[:room]\n")
    text = replace_once(text, "            current_positions[global_idx] = pos + len(tokens)\n",
        "            current_positions[global_idx] = pos + len(tokens)\n"
        "            role_segments[global_idx].append({'role': role or active_role,\n"
        "                'start': pos, 'end': pos + len(tokens), 'trainable': trainable,\n"
        "                'token_ids': list(tokens), 'text': self.tokenizer.decode(tokens, skip_special_tokens=True),\n"
        "                'truncated': len(tokens) < original_length or (trainable and call_truncated.get(global_idx, False))})\n")
    text = replace_once(text,
        "        def _batched_generate(vllm_inputs: List[dict], gen_kwargs: dict) -> List:",
        "        def _batched_generate(vllm_inputs: List[dict], gen_kwargs: dict, sample_indices, role) -> List:\n"
        "            nonlocal active_role, call_truncated\n            active_role = role")
    text = replace_once(text,
        "                return self.inference_engine.generate(\n"
        "                    prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False)",
        "                from native_adapter import generate_role\n"
        "                outputs = generate_role(self, vllm_inputs, self.sampling_params, sample_indices,\n"
        "                                        role, step_idx, prompts, non_tensor_batch)\n"
        "                call_truncated = {g: out.outputs[0].finish_reason == 'length'\n"
        "                                  for g, out in zip(sample_indices, outputs)}\n"
        "                return outputs")
    for indices, kwargs, role in [("active_samples", "sample_kwargs", "solver"),
                                  ("verify_candidates", "verify_kwargs", "verifier"),
                                  ("repair_candidates", "sample_kwargs", "repair"),
                                  ("regen_indices", "sample_kwargs", "regenerate")]:
        before = f"_batched_generate(_build_vllm_inputs({indices}), {kwargs})"
        role_argument = "prompts.meta_info.get('baseline_probe_role', 'solver')" if role == "solver" else repr(role)
        text = replace_once(text, before, before[:-1] + f", {indices}, {role_argument})")
    text = replace_once(text,
        "            active_samples = [g for g in active_samples if final_answers[g] is None]\n",
        "            active_samples = [g for g in active_samples if final_answers[g] is None]\n"
        "            if step_idx + 1 < self.max_reasoning_steps:\n"
        "                from native_adapter import solver_cue\n"
        "                for g in active_samples:\n"
        "                    cue = self.tokenizer.encode(solver_cue(\n"
        "                        tool_output_by_g.get(g, ''), tool_success_by_g.get(g, False),\n"
        "                        repaired_by_g.get(g, False)), add_special_tokens=False)\n"
        "                    written = _append_tokens(g, cue, trainable=False, role='solver_prompt')\n"
        "                    current_inputs[g].extend(cue[:written])\n")
    for variable, role in [("tool_tokens", "tool_observation"), ("verify_prompt_tokens", "verifier_prompt"),
                           ("repair_prompt_tokens", "repair_prompt"), ("regen_prompt_tokens", "regenerate_prompt")]:
        before = f"_append_tokens(g, {variable}, trainable=False)"
        text = replace_once(text, before, before[:-1] + f", role='{role}')")
    text = replace_once(text, "        output_non_tensor = {\n",
        "        baseline_segments = np.empty(batch_size, dtype=object)\n"
        "        for i, segments in enumerate(role_segments):\n            baseline_segments[i] = segments\n"
        "        output_non_tensor = {\n            'baseline_segments': baseline_segments,\n")
    rollout.write_text(text, encoding="utf-8")
    modified = subprocess.check_output(["git", "diff", "--name-only"], cwd=checkout, text=True).splitlines()
    for name in modified:
        if name.endswith(".py"):
            compile((checkout / name).read_text(encoding="utf-8"), name, "exec")
    manifest = {"upstream": UPSTREAM_COMMIT, "patch_sources": engine,
                "patched_files": {name: file_digest(checkout / name) for name in modified},
                "learning_algorithm_changed": False,
                "changes": "baseline hooks, stable group/input IDs, explicit worker/request seeds, role records; existing padding and no-critic metrics"}
    atomic_json(marker, manifest, immutable=True)
    return manifest
