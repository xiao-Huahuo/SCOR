# v01：Agent0-VL 小规模实验

本版采用官方 Agent0-VL 原生训练链路。四卡 3B 试验已执行两次 GRPO 更新，完整 step1 checkpoint 的恢复续训通过；具体证据和限制见 [pilot_report.md](../../docs/v01/pilot_report.md)。配对 reward 错误 GPU 对照尚因资源预检未通过而未启动。

方案与验收规则见 [pilot_plan.md](../../docs/v01/pilot_plan.md)。`inputs/` 与 `outputs/` 不进入 Git；凭据不写入本版代码或收据。

上游固定为 `aiming-lab/Agent0@f775b5101e62fe92976831adf4a21a38fcc0a767`。源码修改为三处：`TaskRunner.run` 的两行项目 hook；多卡 padding 与验证元数据宽度一致性修复；GRPO 无 critic 时跳过 critic 数值指标，保留长度指标。真实 token、mask、原生奖励、GRPO loss 和 optimizer 规则不改。工具、Verifier、Repair 和 FSDP 仍使用官方实现。`math-verify` 使用符合原生评分接口的 0.7.0，评分源码保持原样。24 GB 卡在 vLLM 唤醒前释放未使用的 PyTorch 缓存；使用 V0、0.55 显存预算，并关闭 prefix cache 和 chunked prefill 以避免重复图像的占位符错误。

## 入口

以下命令在项目根目录执行，服务器 Python 为本项目 `.runtime/gpu-venv/bin/python`。本机数据与结构测试使用 `.runtime/venvs/scor-local/bin/python`。

Git 不包含 `.runtime/` 上游源码与依赖环境。新克隆的工作目录先准备固定的官方源码，然后按 `src/bootstrap_server.sh` 建立独立服务器环境：

```bash
mkdir -p .runtime/upstream
git clone https://github.com/aiming-lab/Agent0.git .runtime/upstream/Agent0
git -C .runtime/upstream/Agent0 checkout --detach f775b5101e62fe92976831adf4a21a38fcc0a767
```

```bash
python solutions/v01/src/prepare_data.py
python solutions/v01/src/preflight.py --gpus 4,5,6,7
python solutions/v01/src/run_pilot.py --run-id native_stop1 --model-path <本机已有的完整模型目录> --gpus 4,5,6,7 --stop-after 1
python solutions/v01/src/run_pilot.py --run-id native_resume2 --model-path <同一模型目录> --gpus 4,5,6,7 --resume-from solutions/v01/outputs/native_stop1/checkpoints/global_step_1 --no-new-checkpoint
python solutions/v01/src/verify_run.py solutions/v01/outputs/native_resume2_01
python solutions/v01/src/compare_rewards.py solutions/v01/outputs/native_stop1_10 --step 1 --row 0 --delta -2
```

实际可恢复 checkpoint 为服务器 `solutions/v01/outputs/native_stop1_10/checkpoints/global_step_1`，完整文件约 43.1 GiB。第一步运行的失败状态不改写；恢复成功运行是 `native_resume2_01`。磁盘现剩约 32.8 GiB，不能再保存第二份完整 checkpoint。恢复试验使用 `--no-new-checkpoint`：原 checkpoint 保留，恢复后更新及指标记录，但最终模型不保存。这已验收恢复后训练执行，未验收逐位采样恢复，也未保存 step2 模型。

受控错误入口为 `--corrupt-step 1 --corrupt-row 0 --corrupt-delta -2`。要比较同一批轨迹的奖励变化，必须再加 `--replay-generation-from <已保存原生生成结果的运行目录> --replay-step 1`，并使用同一模型初始状态和数据配置。这个试验不等同于真实 Verifier mistakes。

已保存的 `native_corrupt1_01` 因 GPU4/5 被其他任务占用，在训练前停止。四卡空闲时应使用新编号，例如 `native_corrupt1_02`，从原始模型启动，配合 `--replay-generation-from solutions/v01/outputs/native_stop1_10 --replay-step 1 --no-new-checkpoint`；不能覆盖旧目录。离线 `compare_rewards.py` 已在真正进入 updater 的 batch 上验证：只改一条 return，同组四条 advantage 都变，另一条轨迹由负变正。离线脚本不会做新的 GPU 参数更新。

## 证据与边界

- `raw/generation_step_*.pt`：官方 rollout 输出及用于检查配对一致性的 prompt。
- `raw/training_group_step_*.pt`：真正进入原生 updater 的完整 batch，包含原行为 logprob、reward 和 advantage。
- `raw/reward_step_*.json`：原始 reward、修改后 reward 与工具/验证/repair 信息。
- `raw/metrics_step_*.json`：原生训练指标与实际参数采样变化。
- `checkpoints/global_step_*/scor_complete.json`：模型、optimizer、scheduler/RNG 和 dataloader 写完后的完成标记。
- `processed/verification.json`：只根据实际收据评定；缺失、零 advantage、未观察到 Repair 都不会被包装为成功。

vLLM 采样器的逐位随机状态恢复尚未验证，不能据此宣称两次重新 rollout 完全相同。`.pt` 文件只加载本项目自身生成的可信收据。

本机仅同步 JSON/CSV/log 和 checkpoint 完成证明；大型 `.pt` 和完整模型仍在服务器。需要加载 tensor、验证完整快照或恢复训练的命令应在服务器工作目录执行，本机的完成证明不能代替模型分片。
