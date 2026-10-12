"""Native tensor/GRPO counterexamples; CPU only, no model allocation."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
torch=pytest.importorskip("torch")
from tensordict import TensorDict
from verl import DataProto
from gate1_contract import fixed_unit_audit, historical_output_matches
from gate1_probes import candidate_response
from pilot_utils import atomic_json, file_digest
from records import prompt_hash, tensor_save
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def group_fixture():
    ids=np.array(["a"]*4+["b"]*4,dtype=object)
    rewards=torch.tensor([[0.,0.,v] for v in [0,1,2,3,0,1,2,3]])
    mask=torch.ones_like(rewards)
    adv,ret=compute_grpo_outcome_advantage(rewards,mask,ids)
    fields={"prompts":torch.ones(8,2,dtype=torch.long),"responses":torch.ones(8,3,dtype=torch.long),
            "input_ids":torch.ones(8,5,dtype=torch.long),"attention_mask":torch.ones(8,5,dtype=torch.long),
            "position_ids":torch.arange(5).expand(8,3,5),"multiturn_mask":torch.ones(8,3,dtype=torch.bool),
            "token_level_scores":rewards,"token_level_rewards":rewards.clone(),"advantages":adv,"returns":ret,
            "response_mask":mask,"ref_log_prob":-torch.ones_like(rewards),"old_log_probs":-torch.ones_like(rewards)}
    return DataProto(batch=TensorDict(fields,batch_size=[8]),non_tensor_batch={"uid":ids,
        "visual":np.array(["same-image"]*8,dtype=object)})


def record(source, group):
    tensor_save(group,source/"raw/training_group_step_000008.pt")
    atomic_json(source/"raw/group_step_000008.json",{"payload_hash":prompt_hash(group)})
    atomic_json(source/"raw/completed_step_000008.json",{"learning_signature":"original",
               "group_sha256":file_digest(source/"raw/group_step_000008.json")})


@pytest.mark.parametrize("damage",["none","anchor","image","mask","reward","group","single_advantage","injected"])
def test_fixed_unit_rejects_mismatched_inputs_and_uses_native_whole_group_oracle(tmp_path,damage):
    original=group_fixture();record(tmp_path,original);actual=deepcopy(original)
    if damage=="anchor": actual.batch["old_log_probs"]-=.1
    elif damage=="image": actual.non_tensor_batch["visual"][0]="changed-image"
    elif damage=="mask": actual.batch["multiturn_mask"][0,0]=False
    elif damage=="reward": actual.batch["token_level_rewards"][4,-1]+=1
    elif damage=="group": actual.non_tensor_batch["uid"][0]="different-group"
    elif damage in {"single_advantage","injected"}:
        actual.batch["token_level_scores"][0,-1]+=1
        actual.batch["token_level_rewards"]=actual.batch["token_level_scores"].clone()
        adv,ret=compute_grpo_outcome_advantage(actual.batch["token_level_rewards"],actual.batch["response_mask"],
                                              actual.non_tensor_batch["uid"])
        if damage=="injected": actual.batch["advantages"],actual.batch["returns"]=adv,ret
        else: actual.batch["advantages"][0]=adv[0]
    if damage in {"none","anchor","injected"}:
        assert fixed_unit_audit(actual,tmp_path,8,"original",injected=damage=="injected")["passed"]
    else:
        with pytest.raises(ValueError): fixed_unit_audit(actual,tmp_path,8,"original",injected=damage=="single_advantage")


def test_fixed_candidate_preserves_visual_and_three_mrope_axes():
    class Tokenizer:
        pad_token_id=0
        def encode(self,text,**kwargs): return [9,10]
    prompts=DataProto(batch=TensorDict({"input_ids":torch.tensor([[1,2,3]]),
        "attention_mask":torch.tensor([[0,1,1]]),"position_ids":torch.tensor([[[0,1,2]]*3])},batch_size=[1]),
        non_tensor_batch={"visual":np.array(["same-image"],dtype=object)})
    result=candidate_response(prompts,Tokenizer(),"fixed")
    assert result.batch["input_ids"].tolist()==[[1,2,3,9,10]]
    assert result.batch["position_ids"].tolist()==[[[0,1,2,3,4]]*3]
    assert result.batch["attention_mask"].tolist()==[[0,1,1,1,1]]
    assert result.non_tensor_batch["visual"].tolist()==["same-image"]
    assert prompts.batch["input_ids"].tolist()==[[1,2,3]]


def test_variable_length_boxed_candidates_mask_padding_out_of_attention_and_score():
    class Tokenizer:
        pad_token_id=0
        def encode(self,text,**kwargs): return [9] if text=="short" else [7,8,9]
    prompts=DataProto(batch=TensorDict({"input_ids":torch.tensor([[1,2,3],[1,2,3]]),
        "attention_mask":torch.tensor([[0,1,1],[0,1,1]]),
        "position_ids":torch.tensor([[[0,1,2]]*3]*2)},batch_size=[2]),
        non_tensor_batch={"visual":np.array(["a","b"],dtype=object)})
    result=candidate_response(prompts,Tokenizer(),["short","long"])
    assert result.batch["responses"].tolist()==[[9,0,0],[7,8,9]]
    assert result.batch["multiturn_mask"].tolist()==[[True,False,False],[True,True,True]]
    assert result.batch["attention_mask"].tolist()==[[0,1,1,1,0,0],[0,1,1,1,1,1]]
    assert result.batch["position_ids"].tolist()==[[[0,1,2,3,4,5]]*3]*2
