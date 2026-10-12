# v01：自有机制 baseline

本版目标是基于Agent0-VL建立自己的机制baseline，最多四张4090、分支串行。固定3B/data08、cal11和三个实际GPU对照已通过六项真实冻结证据；[freeze](../../docs/v01/baseline_freeze.md)记录完整条件与累计预算。正式D三个预列seed各32步/五次评测/完整S32已正常完成，E同S7/U8实际更新、唯一terminal+1、全组adv和完整S8审计通过。实际状态见 [baseline_report.md](../../docs/v01/baseline_report.md)，审查见 [baseline_code_review.md](../../docs/v01/baseline_code_review.md)。v02–v07为Gate1–6，均未执行。

以下历史pilot采用官方 Agent0-VL 原生训练链路。四卡 3B 试验已执行两次 GRPO 更新，完整 step1 checkpoint 的恢复续训通过；具体证据和限制见 [pilot_report.md](../../docs/v01/pilot_report.md)。配对 reward 错误 GPU 对照尚因资源预检未通过而未启动。

方案与验收规则见 [pilot_plan.md](../../docs/v01/pilot_plan.md)。`inputs/` 与 `outputs/` 不进入 Git；凭据不写入本版代码或收据。

上游固定为 `aiming-lab/Agent0@f775b5101e62fe92976831adf4a21a38fcc0a767`。源码修改为三处：`TaskRunner.run` 的记录与检查点策略 hook；多卡 padding 与验证元数据宽度一致性修复；GRPO 无 critic 时跳过 critic 数值指标，保留长度指标。真实 token、mask、原生奖励、GRPO loss 和 optimizer 规则不改。工具、Verifier、Repair 和 FSDP 仍使用官方实现。`math-verify` 使用符合原生评分接口的 0.7.0，评分源码保持原样。24 GB 卡在 vLLM 唤醒前释放未使用的 PyTorch 缓存；使用 V0、0.55 显存预算，并关闭 prefix cache 和 chunked prefill 以避免重复图像的占位符错误。

## 现有工程试跑入口

以下入口和配置仍针对已验证的3B试跑，最多8步、仅支持初始step1 generation重放，保存空间门槛也是3B条件。这些历史命令保持原意；新的baseline使用下列独立入口，正式运行要求六项GPU/代码证据及冻结文件。

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

已保存的 `native_corrupt1_01` 因 GPU4/5 被其他任务占用，在训练前停止。历史pilot补试计划若另行执行，须使用新编号，例如 `native_corrupt1_02`，从原始模型启动，配合 `--replay-generation-from solutions/v01/outputs/native_stop1_10 --replay-step 1 --no-new-checkpoint`；不能覆盖旧目录。离线 `compare_rewards.py` 已在真正进入 updater 的 batch 上验证：只改一条 return，同组四条 advantage 都变，另一条轨迹由负变正。离线脚本不会做新的 GPU 参数更新。

## 历史pilot的间隔式 checkpoint

默认每2个训练步保存完整快照，末步、主动停止点和显式关键点额外保存。可用 `--checkpoint-interval N` 调整间隔，`--checkpoint-keep N` 在新快照完整写入后只清理本轮旧快照，`--checkpoint-pin-step S` 强制保存并保护关键训练步。默认保留所有已保存快照，不自动清理旧运行。

磁盘预算按实际保存步和替换时的新旧共存峰值计算。每步仍保存轨迹和指标；模型状态仅在完整快照边界提交，中断后可能需要重算其后的未提交区间。策略、配置、验收和恢复边界见 [checkpoint_policy.md](../../docs/v01/checkpoint_policy.md)。此段保留历史pilot策略的验收范围；独立baseline的间隔8/保护点及实际S0/S7恢复、滚动提交证据见下文和正式报告。

## 证据与边界

- `raw/generation_step_*.pt`：官方 rollout 输出及用于检查配对一致性的 prompt。
- `raw/training_group_step_*.pt`：真正进入原生 updater 的完整 batch，包含训练器重算的 `old_log_probs`、reward 和 advantage；该字段尚未验收为实际采样概率。
- `raw/reward_step_*.json`：原始 reward、修改后 reward 与工具/验证/repair 信息。
- `raw/metrics_step_*.json`：原生训练指标与实际参数采样变化。
- `checkpoints/global_step_*/scor_complete.json`：模型、optimizer、scheduler/RNG 和 dataloader 写完后的完成标记。
- `processed/verification.json`：只根据实际收据评定；缺失、零 advantage、未观察到 Repair 都不会被包装为成功。

vLLM 采样器的逐位随机状态恢复尚未验证，不能据此宣称两次重新 rollout 完全相同。`.pt` 文件只加载本项目自身生成的可信收据。

本机有经完整SHA验证的2B cal01 S0压缩归档；cal01 S1/S2的传输中断，完整服务器副本保留。本机另同步JSON/CSV/log和历史checkpoint完成证明；大型轨迹 `.pt` 仍在服务器，旧训练后的模型/优化器分片已清理。需要加载tensor的命令应在服务器工作目录执行；历史完成证明不能代替当前完整模型分片，旧快照不能再作为恢复输入。

## 独立baseline入口

最终校准profile为`src/configs/baseline_3b_candidate.json`，正式冻结以六项真实证据和该文件SHA约束。模型revision `66285546d2b821cf421d4f5eb2576359d3770cd3`；data08 `visual_baseline_08`；native slow pixels min3136/max32768；四轮×单次256、总response1536/context6144；full32原生cosine/warmup0/初始usedLR5e-7；batch4×n4、micro1、FSDP4/rollout TP2。模型、原生reward/GRPO/AdamW、源码/环境/数据在正式阶段锁定。

```bash
.runtime/gpu-venv/bin/python solutions/v01/src/baseline_acceptance.py health solutions/v01/outputs/baseline_calibration_11 --output <新的验收路径>
.runtime/gpu-venv/bin/python solutions/v01/src/run_baseline.py --run-id <新的正常run_id> --phase baseline --freeze <六项证据齐全的冻结JSON> --seed 20261011 --data solutions/v01/inputs/visual_baseline_08 --model-path .runtime/models/Qwen2.5-VL-3B-Instruct/66285546d2b821cf421d4f5eb2576359d3770cd3 --gpus 4,5,6,7 --pin-step 7 --pin-step 8
.runtime/gpu-venv/bin/python solutions/v01/src/run_baseline_sequence.py --sequence-id <新的序列id> --freeze <六项证据齐全的冻结JSON> --model-path .runtime/models/Qwen2.5-VL-3B-Instruct/66285546d2b821cf421d4f5eb2576359d3770cd3 --data solutions/v01/inputs/visual_baseline_08 --gpus 4,5,6,7
.runtime/gpu-venv/bin/python solutions/v01/src/analyze_baseline.py summarize <已完成32步的正常run目录> --output <新的报告JSON>
.runtime/gpu-venv/bin/python solutions/v01/src/analyze_baseline.py pair <首个正常run目录> --changed <完成的E分支目录> --output <新的配对核验JSON>
.runtime/gpu-venv/bin/python solutions/v01/src/export_baseline_summary.py solutions/v01/outputs/baseline_formal_20261011
```

序列入口只在六项真实witness通过后执行：三个预列seeds20261011/12/13各32步，先完成全部D，再从第一个seed的同S7、固定U8执行E单terminal outcome+1。E正对照复用已完成的native U8/S8；不重算已经完成的控制。各run只有自己新目录，任一实际异常保留失败现场并停止序列。每次启动重新检查资源；GPU sampler在raw目录创建后启动，覆盖恢复前SHA耗时不分配GPU的阶段之后的真实任务。

baseline raw额外保存实际角色/工具、Solver逐case显式答案、固定Verifier诊断、奖励及完整updater batch、完成标记和source/profile。`baseline_complete.json`包含所有model/optimizer/extra/driver RNG/data/worker指纹文件SHA；部分写入不作为完成状态。每8步保存滚动完整状态，额外保护S0/S7，首seed另保护S8。当前引擎S0/S7恢复、无修订U1/U8原生/重放和两次评测+delta0之后下一native U1/fullS1精确一致已经实测。

`analyze_baseline.py`只重建结果，不改变训练；正常报告要求全部32个完成标记、实际optimizer位置/LR、5次独立held-out评测及完整终态。配对审计独立核对固定tensor/non-tensor、第一稳定语义错误目标、唯一terminal+1、Python sample-std/epsilon组advantage和其它group不变；不能用noeligible证书跳过实际eligible目标。9个配对及3个成本原生CPU反例通过，最新CPU04实际12passed/3.17秒；真实E已通过sameS7/U8单reward、完整S8和独立配对审计。

最终`export_baseline_summary.py`要求D三seed/full32/全部五次256case及实际finite E配对已完成，按seed为统计单位生成results.csv、costs.csv、role_costs.csv、paired_advantages.csv和result_summary.json；计时在服务器按原status mtime重建，device显存来自physical_gpu.csv。最新配对/成本/导出反例合计17passed/5.20秒。

teacher-forced actor分数、vLLM返回分数和sampler分布不同来源分列；通用采样逐位恢复尚未证明。cal11固定Verifier正确候选接受率0是校准能力阴性，高JSON解析率不等于判断准确。v01没有实现HED/CSS/TRA/RR或执行六Gate。

## 当前归档

cal05 S1本机完整lossless归档通过整包与17个member bytes/SHA核验，服务器重核源与压缩副本后才回收展开S1和staged archive。cal06 S1及cal05/06/08/09/10 S0另有六份完整验证的同服务器压缩恢复副本；原始raw/source/失败事实和cert保留，压缩副本在`.runtime/checkpoint_archives`，恢复前先核对archive与原certificate的全部成员。cal06本机下载只有96/145已验证块，完整异地备份未完成。当前cal11与三个GPUcontrols、正式保护与终态仍在服务器；本机small receipts及cert copy不等于本机拥有完整训练状态。
