# v01：Agent0-VL 小规模运行结果

记录日期：2026-10-10（Asia/Shanghai）。工程 baseline 的核心训练链路与 checkpoint 恢复已跑通。论文数值复现、六个 Gate 和 SCOR 方法优越性均未验收。

资产更新（2026-10-11）：用户已授权清理下述step1快照的权重与优化器分片；该快照当前不可恢复，原完成证书与当时的恢复试验仍是历史证据。后续清理范围和验收见 [checkpoint_retirement.md](checkpoint_retirement.md)。

学习率复核（2026-10-11）：固定上游的get_constant_schedule_with_warmup在warmup_steps=0时仍以step/max(1,0)初始化LambdaLR，因此第一轮实际LR=0，日志actor/lr是scheduler.step后的下一轮值。表中step1证明原生update/保存执行，不能证明step1权重改变；step2参数抽样变化证据保留。缺失的step1 GPU指标没有补造。新2B候选改用原生cosine并记录baseline/lr_used。

## 条件与代码范围

固定官方 `aiming-lab/Agent0` commit `f775b5101e62fe92976831adf4a21a38fcc0a767`，使用已有 `Qwen2.5-VL-3B-Instruct` snapshot `66285546d2b821cf421d4f5eb2576359d3770cd3`。服务器实际有八张 RTX 4090，每卡 24 GB；本任务只使用物理 GPU4–7，独立环境和工作目录为 `/data1/yangbin/SCOR_v01`。

真实视觉输入、共享 Solver/Verifier、Python 工具、Verifier 判断、内循环 Repair、原生奖励合成、GRPO/FSDP 更新与更新后 rollout 均保留。训练奖励仍包含 ground-truth outcome 项 `alpha_out=1.0`，不能表述为无外部监督。

本版配置每次 4 道训练题、每题 4 条轨迹，每次 update 共 16 条轨迹；共执行两次 update，使用前 8 道训练题，共 32 条训练轨迹。数据准备了 16 道训练题和 4 道验证题，train/val 的图表 scene_id 不重叠。数据由服务器生成，冻结签名为 `6732fb48dcf873739dbbf6c1504d568536870126bc26b4b89509ef7b6b94327a`；本机 Pillow 版本不同产生的文件不能替代该次输入。

上游源码改动仅涉及三处：入口的两行记录 hook、跨 DP rank 的 padding/验证元数据宽度修复、无 critic 的 GRPO 指标条件修复。评分源码、GRPO loss 与 optimizer 更新规则保持原样。运行包装负责收据、参数抽样和存档；为 24 GB 显存使用 V0、0.55 缓存预算、关闭 prefix cache/chunked prefill，并在 wake_up 前释放空闲 allocator block。它是 3B 工程适配，不是原论文全部设置。

## 实际训练与恢复

| 项目 | step1：native_stop1_10 | step2：native_resume2_01 |
|---|---|---|
| 训练题索引 | 0–3，每题四条轨迹 | 4–7，每题四条轨迹 |
| 非零有效 advantage token | 3273 | 3315 |
| 真实工具成功步骤 | 9 | 11 |
| 已验证 Solver 步骤 | 25 | 23 |
| 实际 Repair 步骤 | 2 | 5 |
| 参数更新证据 | update 后保存完整 checkpoint，且恢复续训成功 | 每个 rank 的参数抽样均观察到变化，抽样值有限；grad_norm=2.954097270965576 |
| 进程状态 | checkpoint 写完后指标汇总报错，failed | 正常退出，completed / returncode 0 |
| 完整 checkpoint | step1 保存，12 个 worker 分片及 dataloader/RNG | 未新增；step2 最终模型未持久化 |

step1 的失败不能改写为成功：原生 GRPO 无 critic，指标代码仍引用 `values`，错误发生在更新、验证和 checkpoint 保存之后。该步缺少最终原始 GPU 指标与完成收据；事后从真实 batch 补算的数据指标另存为 `processed/recomputed_data_metrics.json`，没有补造 GPU 更新指标。

step1 完整 checkpoint 位于服务器：

```text
/data1/yangbin/SCOR_v01/solutions/v01/outputs/native_stop1_10/checkpoints/global_step_1
```

实际检查共 46,302,239,618 bytes（约 43.1 GiB），包括四个 rank 的 model/optimizer/extra 共 12 个 `.pt` 分片、`data.pt`、driver RNG 与 `scor_complete.json`。恢复执行了原生 model/optimizer/scheduler/worker RNG/dataloader 加载，以及 driver RNG 加载；训练签名一致，下一批样本为 4–7，未重算 0–3。恢复后重新 rollout 并完成真实参数更新。

恢复运行的[验收收据](../../solutions/v01/outputs/native_resume2_01/processed/verification.json)中，`native_chain_observation_passed=true`，未发现缺失的工具、Verifier、Repair 或有效更新证据。参数抽样变化只能证明实际发生更新，不能解释为所有参数都已改变。vLLM 采样器的逐位 RNG 恢复尚未验证。

## 验证题结果

原始模型、step1 和 step2 的四道验证题均为 **3/4 正确（0.75）**。未观察到准确率提升；同四题反复评测不能作为泛化或统计显著性证据。

准确率直接来自原生逐样本 `acc`，见 `raw/validation_step_*.json`。上游按解码后的文本 prompt 聚合时可能把不同图像的同文问题合并，因此不采用该聚合的 `mean@k` 作为四题整体准确率。部分 `final_answers` 为空，原生评分会回退到解码文本；本次未验收严格 boxed 输出稳定性。

## 真实 batch 的离线 reward 对照

[compare_rewards.py](../../solutions/v01/src/compare_rewards.py)加载真正进入 updater 的 step1 batch，用原生 GRPO 函数重新计算 advantage，结果与保存的输入一致。随后仅将 row0 的 return 从 1 改为 −1（delta=−2），整组重算：

| 同组轨迹 | 原 advantage | 修改后 advantage |
|---|---:|---:|
| row0，修改 reward | −0.354946 | −1.079794 |
| row1，reward 不变 | 1.410661 | 1.283191 |
| row2，reward 不变 | −0.932072 | −0.390471 |
| row3，reward 不变 | −0.123644 | 0.187074 |

同组四条轨迹的训练 advantage 均改变；row3 的 224 个有效训练 token 发生符号翻转。其他十二条轨迹所属 group 未改变。结果与输入 hash 保存在[分析 JSON](../../solutions/v01/outputs/native_stop1_10/processed/reward_counterfactual_step_000001.json)和同名 CSV，可从服务器 raw tensor 重建。

这证明了这批数据中组标准化对 reward 修改的联动影响。它没有执行新的梯度更新，没有检测模型后代污染，也没有识别真实 Verifier 错误，因此不算正式 Gate 通过。

同一初始模型、同一保存 generation 的 GPU 错误分支已准备。`native_corrupt1_01` 在资源检查时发现 GPU4/5 被其他任务占用，只有 GPU6/7 空闲，状态为 `preflight_failed`；没有启动该分支的训练，未干预其他任务。待四卡空闲时用新的 run_id 执行。

## 失败现场与测试

| 尝试 | 失败原因或结果 | 对应处理 |
|---|---|---|
| native_stop1_01 | Ray UNIX socket 路径过长 | 缩短项目临时路径 |
| native_stop1_02 | 自动 V1 引擎、低预算 KV cache 分配失败 | 按原生权重同步接口固定 V0 |
| native_stop1_03–04 | 权重同步后的 vLLM wake_up OOM | 调预算至 0.55；wake 前释放未用 allocator block |
| native_stop1_05 | DP rank 输出宽度不同，拼接失败 | 共用 padding 与元数据宽度 |
| native_stop1_06–07 | 重复图像 token 与 placeholder 数不匹配；关闭 prefix cache 仍失败 | 关闭 chunked prefill 后通过 |
| native_stop1_08 | 项目记录器的 Torch 临时文件名被拒绝 | 用文件对象原子写入；实际保存/加载测试通过 |
| native_stop1_09 | 项目参数探针的 float 索引舍入越界 | 改整数索引；大分片回归测试通过 |
| native_stop1_10 | 更新与保存完成后，原生无 critic 指标引用 values | 修复指标条件，保留 failed 状态与可恢复 checkpoint |
| native_resume2_01 | 恢复并完成 step2，正常退出 | 原始收据已保存 |
| native_corrupt1_01 | 四卡资源不足 | 预检停止，GPU 对照未启动 |

环境安装阶段还经历了精确 xgrammar 版本不再提供 PyPI wheel、CDN 下载缓慢和本次初选 math-verify0.3.0 接口不匹配。最终 xgrammar0.1.16 从官方 tag 编译，依赖固定版本；FlashAttention 官方 wheel 的 ZIP/传输 SHA256 校验通过；math-verify0.7.0 恢复了原生接口，不改评分规则。原失败日志保留。

服务器[测试日志](../../solutions/v01/outputs/server_preflight_20261010/raw/native_tests_05.log)为 **10 passed in 4.40s**，包含六项结构/完整性测试和四项原生奖励、整组 advantage、真实工具、Torch 收据与大参数索引检查。本机六项通过，原生模块因未装 Torch 跳过；不把跳过项算通过。实际 CUDA 矩阵乘法和 BF16 FlashAttention kernel 也已通过。

## 当前边界与下一步

- 已完成：本项目开发、四卡核心链路小规模试跑、非零 group advantage、真实参数更新、checkpoint 保存与恢复后续训、实际 batch 的离线 reward 对照。
- 尚未完成：同轨迹 GPU reward 更新分支，目前需四张空闲卡；论文的 SFT cold-start、7B/8B benchmark 数值复现；六个研究 Gate。
- 存储限制：余盘约 32.8 GiB，保留 step1 完整 checkpoint；恢复运行未保存最终模型。全量试验前需要足够 checkpoint 空间。
- 解释限制：3B 指令模型、合成图表、两步训练和四题验证只验收工程执行。未主张性能提升、SCOR 有效性或长期污染。

JSON/CSV/log 与 checkpoint 完成证明已同步本机；完整 model/optimizer 分片、生成 tensor 和 updater batch 仍在服务器。本机完成证明不能用于直接恢复模型。所有运行使用独立编号，失败记录没有被后续成功覆盖。
