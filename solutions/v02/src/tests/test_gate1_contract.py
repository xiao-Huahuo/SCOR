"""Independent acceptance counterexamples for Gate1 statistics and identities."""
from copy import deepcopy
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate1_contract import compatible_learning, paired_interval, score_distance
from pilot_utils import ROOT, SRC, file_digest


def scores():
    cases=[]
    for role in ["solver", "verifier"]:
        for case in ["a", "b"]:
            for candidate in range(2):
                cases.append({"case_id":case,"base_case_id":case,"role":role,"candidate_id":candidate,
                              "input_hash":"same-visual-prefix-mask", "token_logp":[-2.]*(1 if candidate==0 else 9)})
    return {"probe_signature":"fixed-before-outcomes", "cases":cases}


def test_case_candidate_role_weighting_does_not_weight_long_candidates_more():
    a=scores();b=deepcopy(a)
    for case in b["cases"]:
        shift=3 if case["role"]=="solver" and case["candidate_id"]==0 else 0
        case["token_logp"]=[x+shift for x in case["token_logp"]]
    result=score_distance(a,b)
    assert result["roles"]["solver"]==pytest.approx(3/math.sqrt(2))
    assert result["roles"]["verifier"]==0
    assert result["D_score"]==pytest.approx(3/(2*math.sqrt(2)))


@pytest.mark.parametrize("damage",["duplicate","missing_role","identity","nan","length","base_case"])
def test_probe_damage_rejected(damage):
    a=scores();b=deepcopy(a)
    if damage=="duplicate": b["cases"].append(deepcopy(b["cases"][0]))
    elif damage=="missing_role":
        a["cases"]=[x for x in a["cases"] if x["role"]=="solver"]
        b=deepcopy(a)
    elif damage=="identity": b["cases"][0]["input_hash"]="different-image"
    elif damage=="nan": b["cases"][0]["token_logp"][0]=float("nan")
    elif damage=="length": b["cases"][0]["token_logp"].append(-2.)
    elif damage=="base_case":
        for payload in [a,b]:
            for item in payload["cases"]:
                if item["role"]=="verifier": item["base_case_id"]+="-different"
    with pytest.raises(ValueError): score_distance(a,b)


def test_t_interval_matches_published_df4_critical_and_preserves_negative_floors():
    effect=[-.01,.00,.01,.02,.03]
    result=paired_interval(effect,.01)
    # Mean=.01, sample variance=.00025, SE=sqrt(.00005).
    radius=2.7764451051977987*math.sqrt(.00005)
    assert result["mean"]==pytest.approx(.01)
    assert result["lower"]==pytest.approx(.01-radius)
    assert result["upper"]==pytest.approx(.01+radius)
    assert result["decision"]=="uncertain"


@pytest.mark.parametrize("values,expected",[([.02]*5,"supported"),([0.]*5,"within_declared_tolerance"),
                                         ([.02]*2,"uncertain"),([.02],"uncertain")])
def test_three_decisions_and_minimum_training_seed_count(values,expected):
    assert paired_interval(values,.01)["decision"]==expected


@pytest.mark.parametrize("values,epsilon",[([float("nan")]*5,.01),([1.]*6,.01),([1.]*5,0)])
def test_invalid_statistics_rejected(values,epsilon):
    with pytest.raises(ValueError): paired_interval(values,epsilon)


@pytest.mark.parametrize("damage",["none","lr","seed","model","data","environment","patch","missing_engine"])
def test_source_bridge_accepts_only_equal_learning_conditions(damage):
    names=["run_baseline.py","native_adapter.py","records.py","checkpoint_store.py","replay.py",
           "interventions.py","evaluate.py","probe.py","baseline_acceptance.py",
           "apply_baseline_instrumentation.py","apply_instrumentation.py","pilot_utils.py"]
    canonical={name:file_digest(ROOT/"solutions/v01/src"/name) for name in names}
    source={"native_parameters":{"lr":5e-7},"seed":20261011,"model_signature":"original-model",
            "dataset_signature":"original-data","environment_signature":"frozen-packages", "patch":"pinned",
            "source_sha256":canonical}
    current=deepcopy(source)
    current["source_sha256"]={name:file_digest(SRC/name) for name in names+[
        "run_gate1.py","gate1_contract.py","gate1_probes.py"]}
    if damage=="lr": current["native_parameters"]["lr"]=1e-6
    elif damage=="seed": current["seed"]+=1
    elif damage=="model": current["model_signature"]="another-model"
    elif damage=="data": current["dataset_signature"]="another-data"
    elif damage=="environment": current["environment_signature"]="another-environment"
    elif damage=="patch": current["patch"]="different-upstream"
    elif damage=="missing_engine": del current["source_sha256"]["native_adapter.py"]
    if damage=="none":
        assert compatible_learning(current,source,canonical)["source_learning_signature"]
    else:
        with pytest.raises(ValueError): compatible_learning(current,source,canonical)
