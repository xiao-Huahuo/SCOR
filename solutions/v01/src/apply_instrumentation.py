"""Add one explicit project hook to the pinned upstream driver."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from pilot_utils import UPSTREAM_COMMIT, file_digest

BEFORE = "    def run(self, config):\n        from verl.utils.fs import copy_to_local\n"
AFTER = ("    def run(self, config):\n"
         "        from scor_hooks import install\n"
         "        install(config)\n"
         "        from verl.utils.fs import copy_to_local\n")


def apply(checkout: Path) -> str:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise ValueError(f"Unexpected upstream commit: {actual}")
    path = checkout / "Agent0-VL/verl/trainer/main_ppo.py"
    text = path.read_text(encoding="utf-8")
    if AFTER not in text:
        if text.count(BEFORE) != 1:
            raise ValueError("Upstream hook location changed; do not patch by guesswork")
        path.write_text(text.replace(BEFORE, AFTER), encoding="utf-8")
    score_path = checkout / "Agent0-VL/verl/utils/reward_score/math_verify.py"
    score_text = score_path.read_text(encoding="utf-8")
    native_import = "from math_verify.errors import TimeoutException"
    temporary_import = "from builtins import TimeoutError as TimeoutException"
    # Undo the unsuccessful 0.3.0 compatibility experiment: 0.7.0 supports the
    # author's original import and boxed_match_priority API without a score patch.
    if temporary_import in score_text:
        score_path.write_text(score_text.replace(temporary_import, native_import), encoding="utf-8")
    elif native_import not in score_text:
        raise ValueError("Native scoring source changed; inspect before running")
    rollout_path = checkout / "Agent0-VL/verl/workers/rollout/vllm_rollout/vllm_agent0_rollout_spmd.py"
    rollout_text = rollout_path.read_text(encoding="utf-8")
    old_length = "max_response_len = max(max(current_positions), 1) if current_positions else 1"
    new_length = "max_response_len = max_total_length  # SCOR: consistent padded shape across DP ranks"
    if old_length in rollout_text:
        rollout_text = rollout_text.replace(old_length, new_length)
        rollout_text = rollout_text.replace(
            "max_verify_len = max((len(vp) for vp in verify_probs), default=0)",
            "max_verify_len = self.max_reasoning_steps  # SCOR: consistent metadata shape across DP ranks")
        rollout_path.write_text(rollout_text, encoding="utf-8")
    elif new_length not in rollout_text:
        raise ValueError("Native rollout finalization changed; inspect before adapting")
    metric_path = checkout / "Agent0-VL/verl/trainer/ppo/metric_utils.py"
    metric_text = metric_path.read_text(encoding="utf-8")
    old_values = (
        "        eos_value_indices = torch.clamp(last_pos + 1, 0, values.size(1) - 1)\n"
        "        turn_values = torch.masked_select(values, turn_mask)\n"
        "        turn_eos_value = torch.gather(values, dim=1, index=eos_value_indices.unsqueeze(1))\n")
    if old_values in metric_text:
        metric_text = metric_text.replace(old_values,
            "        if use_critic:  # SCOR: GRPO has no critic values\n" +
            "\n".join("    " + line for line in old_values.rstrip().splitlines()) + "\n")
        metric_text = metric_text.replace("        turn_eos_value = turn_eos_value[valid_mask]",
            "        if use_critic:\n            turn_eos_value = turn_eos_value[valid_mask]")
        start = metric_text.index("            values_metrics = {", metric_text.index("for turn in range"))
        end = metric_text.index("            metrics.update(values_metrics)", start) + len("            metrics.update(values_metrics)")
        block = metric_text[start:end].replace("            metrics.update(turn_metrics)\n", "")
        replacement = ("            metrics.update(turn_metrics)\n"
                       "            if use_critic:\n" +
                       "\n".join("    " + line if line.strip() else line for line in block.splitlines()))
        metric_text = metric_text[:start] + replacement + metric_text[end:]
        metric_path.write_text(metric_text, encoding="utf-8")
    elif "if use_critic:  # SCOR: GRPO has no critic values" not in metric_text:
        raise ValueError("Native GRPO metrics changed; inspect before adapting")
    return file_digest(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    print(apply(parser.parse_args().checkout.resolve()))
