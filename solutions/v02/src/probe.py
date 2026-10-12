"""Fixed Verifier decisions and teacher-forced scores on separate diagnostics."""
from __future__ import annotations

from pathlib import Path


def fixed_response(prompts, tokenizer):
    import torch
    from tensordict import TensorDict
    from verl import DataProto
    text = '{"step_index":1,"score":1,"confidence":0.9,"critique":"Correct.","tool_check":false}'
    ids = tokenizer.encode(text, add_special_tokens=False)
    inputs, positions, attention = (prompts.batch[k] for k in ("input_ids", "position_ids", "attention_mask"))
    response = torch.tensor(ids, dtype=inputs.dtype).unsqueeze(0).repeat(len(prompts), 1)
    delta = torch.arange(1, len(ids) + 1).unsqueeze(0).expand(len(prompts), -1)
    if positions.dim() == 3:
        delta = delta.view(len(prompts), 1, -1).expand(len(prompts), 3, -1)
    tensors = {"prompts": inputs, "responses": response, "input_ids": torch.cat([inputs, response], -1),
               "position_ids": torch.cat([positions, positions[..., -1:] + delta], -1),
               "attention_mask": torch.cat([attention, torch.ones_like(response)], -1),
               "multiturn_mask": torch.ones_like(response, dtype=torch.bool)}
    return DataProto(batch=TensorDict(tensors, batch_size=[len(prompts)]),
                     non_tensor_batch=dict(prompts.non_tensor_batch)), text


def evaluate_verifier(trainer, manifest):
    import numpy as np
    from torch.utils.data import DataLoader
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from verl.workers.rollout.vllm_rollout.vllm_agent0_rollout_spmd import vLLMAgent0Rollout
    split = "probe" if manifest["evaluation_split"] == "evaluation" else "calibration"
    path = Path(manifest["dataset_directory"]) / f"{split}_verifier.parquet"
    dataset = RLHFDataset(parquet_files=str(path), tokenizer=trainer.tokenizer, processor=trainer.processor,
        prompt_key=trainer.config.data.prompt_key, image_key="images",
        max_prompt_length=trainer.config.data.max_prompt_length, filter_overlong_prompts=False,
        truncation="error", num_workers=1)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)
    cases = []
    for batch in loader:
        data = DataProto.from_single_dict(batch)
        names = ["raw_prompt_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch:
            names += ["multi_modal_data", "multi_modal_inputs"]
        prompts = data.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=names)
        prompts.non_tensor_batch["baseline_case_id"] = np.array(
            [info["case_id"] for info in data.non_tensor_batch["extra_info"]], dtype=object)
        prompts.meta_info.update({"validate": True, "do_sample": False, "baseline_step": trainer.global_steps,
                                 "baseline_phase": "verifier_probe_" + split, "baseline_probe_role": "verifier_probe"})
        scoring, fixed_text = fixed_response(prompts, trainer.tokenizer)
        scores = trainer.actor_rollout_wg.compute_log_prob(scoring).batch["old_log_probs"].tolist()
        generated = trainer.actor_rollout_wg.generate_sequences(prompts)
        for i, info in enumerate(data.non_tensor_batch["extra_info"]):
            segments = generated.non_tensor_batch["baseline_segments"][i]
            response = next(segment["text"] for segment in segments if segment["role"] == "verifier_probe")
            parsed = vLLMAgent0Rollout._parse_verification_output(None, response)
            accepted = bool(parsed and parsed["score"] > 0 and parsed["confidence"] >= trainer.config.reward_model.tau_c)
            cases.append({"probe_id": info["case_id"], "expected_correct": bool(info["expected_correct"]),
                          "parsed": parsed is not None, "accepted": accepted, "response": response,
                          "parsed_verifier": parsed, "fixed_response": fixed_text,
                          "fixed_response_actor_logp": scores[i],
                          "score_source": "native actor teacher-forced model score; not sampler probability"})
    wrong, correct = [x for x in cases if not x["expected_correct"]], [x for x in cases if x["expected_correct"]]
    summary = {"cases": len(cases), "parse_rate": sum(x["parsed"] for x in cases) / len(cases),
               "incorrect_acceptance_rate": sum(x["accepted"] for x in wrong) / len(wrong),
               "correct_acceptance_rate": sum(x["accepted"] for x in correct) / len(correct),
               "threshold": {"score": ">0", "confidence": float(trainer.config.reward_model.tau_c)}}
    return {"split": split, "cases": cases, "summary": summary}
