# v01：自有机制 baseline

本版目标是基于 Agent0-VL 建立自己的机制 baseline，后续最多四张4090、各实验分支串行。设计见 [baseline_design.md](../../docs/v01/baseline_design.md)，六Gate协议见 [gate_protocol.md](../../docs/v01/gate_protocol.md)。优先验证2B候选；配置未冻结、新入口尚未实现。v02–v07规划为Gate1–6，正式baseline不另占v02。

本版采用官方 Agent0-VL 原生训练链路。四卡 3B 试验已执行两次 GRPO 更新，完整 step1 checkpoint 的恢复续训通过；具体证据和限制见 [pilot_report.md](../../docs/v01/pilot_report.md)。配对 reward 错误 GPU 对照尚因资源预检未通过而未启动。

方案与验收规则见 [pilot_plan.md](../../docs/v01/pilot_plan.md)。`inputs/` 与 `outputs/` 不进入 Git；凭据不写入本版代码或收据。

上游固定为 `aiming-lab/Agent0@f775b5101e62fe92976831adf4a21a38fcc0a767`。源码修改为三处：`TaskRunner.run` 的记录与检查点策略 hook；多卡 padding 与验证元数据宽度一致性修复；GRPO 无 critic 时跳过 critic 数值指标，保留长度指标。真实 token、mask、原生奖励、GRPO loss 和 optimizer 规则不改。工具、Verifier、Repair 和 FSDP 仍使用官方实现。`math-verify` 使用符合原生评分接口的 0.7.0，评分源码保持原样。24 GB 卡在 vLLM 唤醒前释放未使用的 PyTorch 缓存；使用 V0、0.55 显存预算，并关闭 prefix cache 和 chunked prefill 以避免重复图像的占位符错误。

## 现有工程试跑入口

以下入口和配置仍针对已验证的3B试跑，最多8步、仅支持初始step1 generation重放，保存空间门槛也是3B条件。新设计中的run_baseline/state分叉/replay接口尚未实现，不能直接以这些命令执行正式baseline或后续Gate。

以下命令在项目根目录执行；服务器根目录统一为 `/data1/yangbin/SCOR/`，Python 为其 `.runtime/gpu-venv/bin/python`。本机数据与结构测试使用 `.runtime/venvs/scor-local/bin/python`。旧 `/data1/yangbin/SCOR_v01` 仅为兼容链接，历史记录中的原路径仍保留。

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

历史恢复试验使用服务器 `solutions/v01/outputs/native_stop1_10/checkpoints/global_step_1`，当时完整文件约43.1GiB；第一步运行的失败状态不改写，恢复成功运行为 `native_resume2_01`。2026-10-11用户授权清理该快照的模型/优化器分片，完整性检查现为false，不能再用它续训。清理后可用75.87GiB，原始轨迹、日志、配置和历史完成证书保留；完成证书只证明当时保存成功。恢复试验采用 `--no-new-checkpoint`，已验收当时的恢复后训练执行，未验收逐位采样恢复，也未保存step2模型。[清理记录](../../docs/v01/checkpoint_retirement.md)

受控错误入口为 `--corrupt-step 1 --corrupt-row 0 --corrupt-delta -2`。要比较同一批轨迹的奖励变化，必须再加 `--replay-generation-from <已保存原生生成结果的运行目录> --replay-step 1`，并使用同一模型初始状态和数据配置。这个试验不等同于真实 Verifier mistakes。

已保存的 `native_corrupt1_01` 因 GPU4/5 被其他任务占用，在训练前停止。四卡空闲时应使用新编号，例如 `native_corrupt1_02`，从原始模型启动，配合 `--replay-generation-from solutions/v01/outputs/native_stop1_10 --replay-step 1 --no-new-checkpoint`；不能覆盖旧目录。离线 `compare_rewards.py` 已在真正进入 updater 的 batch 上验证：只改一条 return，同组四条 advantage 都变，另一条轨迹由负变正。离线脚本不会做新的 GPU 参数更新。

## 间隔式 checkpoint

默认每2个训练步保存完整快照，末步、主动停止点和显式关键点额外保存。可用 `--checkpoint-interval N` 调整间隔，`--checkpoint-keep N` 在新快照完整写入后只清理本轮旧快照，`--checkpoint-pin-step S` 强制保存并保护关键训练步。默认保留所有已保存快照，不自动清理旧运行。

磁盘预算按实际保存步和替换时的新旧共存峰值计算。每步仍保存轨迹和指标；模型状态仅在完整快照边界提交，中断后可能需要重算其后的未提交区间。策略、配置、验收和恢复边界见 [checkpoint_policy.md](../../docs/v01/checkpoint_policy.md)。新策略CPU测试通过，真实GPU间隔保存/恢复尚待资源与存储到位。

## 证据与边界

- `raw/generation_step_*.pt`：官方 rollout 输出及用于检查配对一致性的 prompt。
- `raw/training_group_step_*.pt`：真正进入原生 updater 的完整 batch，包含训练器重算的 `old_log_probs`、reward 和 advantage；该字段尚未验收为实际采样概率。
- `raw/reward_step_*.json`：原始 reward、修改后 reward 与工具/验证/repair 信息。
- `raw/metrics_step_*.json`：原生训练指标与实际参数采样变化。
- `checkpoints/global_step_*/scor_complete.json`：模型、optimizer、scheduler/RNG 和 dataloader 写完后的完成标记。
- `processed/verification.json`：只根据实际收据评定；缺失、零 advantage、未观察到 Repair 都不会被包装为成功。

vLLM 采样器的逐位随机状态恢复尚未验证，不能据此宣称两次重新 rollout 完全相同。`.pt` 文件只加载本项目自身生成的可信收据。

本机仅同步JSON/CSV/log和历史checkpoint完成证明；大型轨迹 `.pt` 仍在服务器，旧训练后的模型/优化器分片已清理。需要加载tensor的命令应在服务器工作目录执行；历史完成证明不能代替当前完整模型分片，旧快照不能再作为恢复输入。
