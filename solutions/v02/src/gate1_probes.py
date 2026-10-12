"""Immutable dual-role fixed candidate inputs and native actor scores."""
from __future__ import annotations

import argparse
from pathlib import Path

from pilot_utils import SRC, VERSION, atomic_json, digest, file_digest, read_json
from records import prompt_hash, tensor_save

SOLVER_CANDIDATES = ["boxed_ground_truth", "boxed_ground_truth_plus_one"]
VERIFIER_CANDIDATES = [
    '{"step_index":1,"score":1,"confidence":0.9,"critique":"Correct.","tool_check":false}',
    '{"step_index":1,"score":0,"confidence":0.9,"critique":"Incorrect.","tool_check":false}']


def candidate_response(prompts, tokenizer, text):
    import torch
    from tensordict import TensorDict
    from verl import DataProto
    texts = [text]*len(prompts) if isinstance(text, str) else text
    if len(texts) != len(prompts):
        raise ValueError("Candidate count differs from fixed cases")
    ids = [tokenizer.encode(item, add_special_tokens=False) for item in texts]
    if not all(ids):
        raise ValueError("Empty probe candidate")
    inputs, positions, attention = (prompts.batch[k] for k in ("input_ids", "position_ids", "attention_mask"))
    width = max(map(len, ids))
    response = torch.full((len(prompts), width), tokenizer.pad_token_id, dtype=inputs.dtype)
    response_mask = torch.zeros_like(response, dtype=torch.bool)
    for i, tokens in enumerate(ids):
        response[i, :len(tokens)] = torch.tensor(tokens, dtype=inputs.dtype)
        response_mask[i, :len(tokens)] = True
    delta = torch.arange(1, width+1).unsqueeze(0).expand(len(prompts), -1)
    if positions.dim() == 3:
        delta = delta.view(len(prompts), 1, -1).expand(len(prompts), positions.shape[1], -1)
    tensors = {"prompts": inputs, "responses": response, "input_ids": torch.cat([inputs, response], -1),
               "position_ids": torch.cat([positions, positions[..., -1:]+delta], -1),
               "attention_mask": torch.cat([attention, response_mask.to(attention.dtype)], -1),
               "multiturn_mask": response_mask}
    return DataProto(batch=TensorDict(tensors, batch_size=[len(prompts)]),
                     non_tensor_batch=dict(prompts.non_tensor_batch))


def prepare(model: Path, data: Path, destination: Path, split: str) -> dict:
    from torch.utils.data import DataLoader
    from verl import DataProto
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from prepare_baseline_data import verify
    profile = read_json(SRC / "configs/baseline_3b_candidate.json")
    dataset_manifest = verify(data)
    tokenizer = hf_tokenizer(str(model))
    processor = hf_processor(str(model), use_fast=False, **profile["processor_options"])
    batches = []
    for role, suffix, candidates in [("solver", "", SOLVER_CANDIDATES),
                                     ("verifier", "_verifier", VERIFIER_CANDIDATES)]:
        dataset = RLHFDataset(parquet_files=str(data/f"{split}{suffix}.parquet"), tokenizer=tokenizer,
            processor=processor, prompt_key=profile["overrides"].get("data.prompt_key", "prompt"), image_key="images",
            max_prompt_length=profile["overrides"]["data.max_prompt_length"], filter_overlong_prompts=False,
            truncation="error", num_workers=1)
        for batch in DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0):
            full = DataProto.from_single_dict(batch)
            info = full.non_tensor_batch["extra_info"]
            names = ["raw_prompt_ids"]
            if "multi_modal_inputs" in full.non_tensor_batch:
                names += ["multi_modal_data", "multi_modal_inputs"]
            prompts = full.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=names)
            prefix = prompt_hash(prompts)
            for index, candidate in enumerate(candidates):
                texts = [f"\\boxed{{{int(item['answer']) + index}}}" for item in info] if role == "solver" else [
                    candidate]*len(prompts)
                scoring = candidate_response(prompts, tokenizer, texts)
                identity = prompt_hash(scoring)
                cases = [{"case_id": str(item["case_id"]), "base_case_id": str(item.get("base_case_id", item["case_id"])),
                          "role": role, "candidate_id": index, "candidate": texts[i],
                          "candidate_mask": scoring.batch["multiturn_mask"][i].tolist(),
                          "image_sha256": item.get("image_sha256"), "batch_prefix_hash": prefix,
                          "batch_scoring_hash": identity} for i, item in enumerate(info)]
                batches.append({"data": scoring, "cases": cases, "input_hash": identity})
    identities = [(x["input_hash"], x["cases"]) for x in batches]
    receipt = {"split": split, "dataset_signature": dataset_manifest["data_signature"],
               "source_sha256": file_digest(Path(__file__)), "profile_sha256": file_digest(SRC/"configs/baseline_3b_candidate.json"),
               "model_tokenizer_sha256": file_digest(model/"tokenizer.json"),
               "solver_candidates": SOLVER_CANDIDATES, "verifier_candidates": VERIFIER_CANDIDATES,
               "batch_identities": identities}
    receipt["probe_signature"] = digest(receipt)
    destination.mkdir(parents=True, exist_ok=False)
    tensor_save(batches, destination/"fixed_inputs.pt")
    receipt["payload_sha256"] = file_digest(destination/"fixed_inputs.pt")
    atomic_json(destination/"manifest.json", receipt, immutable=True)
    return receipt


def evaluate_fixed(trainer, directory: Path) -> dict:
    import torch
    manifest = read_json(directory/"manifest.json")
    unsigned = {k: v for k, v in manifest.items() if k not in {"probe_signature", "payload_sha256"}}
    if digest(unsigned) != manifest["probe_signature"] or file_digest(directory/"fixed_inputs.pt") != manifest["payload_sha256"]:
        raise ValueError("Fixed probe manifest/payload changed")
    batches = torch.load(directory/"fixed_inputs.pt", map_location="cpu", weights_only=False)
    cases = []
    if len(batches) != len(manifest["batch_identities"]):
        raise ValueError("Fixed probe batch count differs")
    for batch, expected in zip(batches, manifest["batch_identities"]):
        scoring = batch["data"]
        if prompt_hash(scoring) != expected[0] or digest(batch["cases"]) != digest(expected[1]):
            raise ValueError("Fixed probe visual/token/mask identity differs")
        scores = trainer.actor_rollout_wg.compute_log_prob(scoring).batch["old_log_probs"]
        mask = scoring.batch["multiturn_mask"].bool()
        if scores.shape != scoring.batch["responses"].shape or not torch.isfinite(scores[mask]).all():
            raise ValueError("Fixed probe native score shape/values invalid")
        cases.extend({**case, "token_logp": score[mask[i]].tolist()} for i, (case, score) in
                     enumerate(zip(batch["cases"], scores)))
    return {"probe_signature": manifest["probe_signature"], "split": manifest["split"], "cases": cases,
            "score_source": "native actor teacher-forced fixed-candidate token log-score"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=VERSION/"inputs/visual_baseline_08")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["probe", "calibration"], required=True)
    args = parser.parse_args()
    print(prepare(args.model_path, args.data, args.output, args.split)["probe_signature"])


if __name__ == "__main__":
    main()
