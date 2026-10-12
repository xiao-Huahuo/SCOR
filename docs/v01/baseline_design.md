# v01：四卡机制 baseline 设计

日期：2026-10-10（Asia/Shanghai）。状态：**2026-10-10原始设计；2026-10-12六项真实证据及累计预算冻结通过，正式D三个seed完整32步/五次评测/fullS32正常验收，E实际同S7/U8配对更新及完整S8审计通过**。本文件依据 [idea.md](../RSI-Agent/idea.md)、原生训练代码和已有试跑证据制定；后续实验细则见 [gate_protocol.md](gate_protocol.md)，审查见 [design_review.md](design_review.md)。

执行遵守用户最新提供的 [科研开发规范.md](../../科研开发规范.md)，包括新增的强制代码正确性审查；依赖实现的P0/P1缺陷或关键证据缺失必须闭环后，才能进入正式实验。

用户已确定两项约束：**v01 就是自己的 baseline；后续实验最多使用四张 RTX 4090**。模型、数据和规模可以调整，以机制对比为主。下文的候选模型、数量和阈值是本次设计建议，不代表用户已经确认或实验已经预注册。各节保留设计日的候选与建议；实际固定条件见 [baseline_freeze.md](baseline_freeze.md)，校准修订时序见 [baseline_report.md](baseline_report.md)。后续Gate的独立数量与判定仍待各版本冻结。

## 1. baseline 要为后面解决什么问题

v01 的交付是一个可比较、可恢复、能持续生成训练数据的正常训练对照：

```text
视觉任务 → Solver → 实际工具执行 → 共享参数的 Verifier
       → 必要时 inner-loop Repair → 最终 training groups
       → 原生奖励合成 → GRPO → 新模型继续生成下一轮数据
```

保留这个闭环，才能研究 idea 中的 `错误 reward → 参数偏移 → 后续数据变化 → 后续更新偏移`。只保留一次推理、固定数据训练或离线改 advantage，都不足以替代它。

先建立正常训练与最小实验接口，六个 Gate 再检验现象。HED、CSS、TRA、RR 的实际算法开发排在验证之后。v01 不承担原论文绝对分数复现，也不以 Gate 出现阳性结果作为 baseline 验收条件。

| 版本 | 目标 | 本次状态 |
|---|---|---|
| v01 | 自己的 Agent0-VL 类机制 baseline、记录/恢复/分叉/重放基础 | 固定3B/data08已冻结；D三个seed完整32步已正常验收；E实际同S7/U8配对更新及完整S8审计通过 |
| v02 | Gate 1：错误影响是否保留 | 规划 |
| v03 | Gate 2：是否经自生成数据传播 | 规划 |
| v04 | Gate 3：是否造成能力损失 | 规划 |
| v05 | Gate 4：Direct Correction 是否足够 | 规划 |
| v06 | Gate 5：污染集合是否值得选择性处理 | 规划 |
| v07 | Gate 6：Oracle-SCOR 的效果与成本上限 | 规划 |

上述 Gate 版本只有各自获得证据后才进入下一步；本次没有创建这些版本的源码目录。正式 baseline 另算 v02、Gate 排到 v08 的旧规划取消。

## 2. 当前资产与尚缺的能力

已有 Qwen2.5-VL-3B-Instruct 在四卡上的两次原生update执行（首轮LR=0，参数变化证据来自第二轮）、工具/Verifier/Repair、完整 step1 快照和恢复续训。四题评测均为 3/4，无性能提升证据；最终 step2 模型没有保存。单 reward 修改的证据来自离线 batch，GPU 配对分支尚未执行。[试跑报告](pilot_report.md)

现有入口是有意受限的工程试跑：`check_profile` 限制最多8步；generation 重放只支持同配置、同初始模型的 step1；保护点不支持 step0；保存器含固定的3B空间门槛；验证题没有独立统计能力。它们不能直接冒充后续完整实验框架。

保留 `run_pilot.py`、原始配置、旧 hooks 与旧输出的含义。新的 baseline 入口独立开发，不通过去掉旧入口的限制来复用历史 run_id，也不让新模型继承旧 training signature。

## 3. 模型与四卡拓扑

**优先考察 Qwen2-VL-2B-Instruct，全参数 GRPO**；已有3B保留为已验证的工程参照。2B 是官方指令 VLM，当前固定的 vLLM 0.8.2 支持 Qwen2VL 架构；本地固定上游也包含 Qwen2VL 的视觉模型加载和 attention 适配。这支持兼容性尝试，不能替代实际训练、权重同步或恢复验收。[官方模型卡](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)、[vLLM 0.8.2 支持列表](https://docs.vllm.ai/en/v0.8.2/models/supported_models.html)

采用2B的理由是降低整条训练与修复路径的资源需求，而非只降低一次推理的显存。若2B无法形成正常工具/Verifier/Repair闭环，应先查格式、上下文和任务难度；修复后仍不具备任务能力才改模型，并在正式 Gate 前重新冻结。

| 项目 | 初始候选 | 依据与限制 |
|---|---|---|
| 训练 | 4卡 FSDP；全参数；原生 AdamW/GRPO | 优先延续已验证框架；四卡显存不能简单视为一个96GB空间 |
| rollout | TP=2，4卡上两份 rollout replica | Solver/Verifier/Repair 分时使用同一模型 |
| reference | 固定初始模型；参数 CPU offload | reference 与可训练的 Verifier 是两个概念 |
| optimizer | CPU offload，gradient checkpointing | 降显存，不等于缩小完整快照 |
| group | 每轮4个问题，每题4条轨迹，共16条 | 保持组完整、组内归一化、固定数据顺序 |
| mini/micro batch | 每轮一个全局 optimizer step；每卡 micro=1；ppo_epochs=1 | 记录实际步数，不能只看配置值；上游会按 n 和 world size 改写 mini batch |
| 学习率 | 5e-7，原生cosine、warmup=0、完整32步horizon，沿用当前 KL/entropy 配置作为起点 | 先校准健康性，随后所有分支一致；scheduler 按完整 horizon 初始化 |
| 推理适配 | V0、eager、prefix cache/chunked prefill关闭；预算0.55、max_seqs=16作为起点 | 新模型须重新测峰值，不宣称旧值自动适用 |
| 初始长度 | prompt上限2048，每次256，当前候选4轮/总响应3072/context8192 | 2026-10-11因工具后角色切换和完整Repair上下文容量修订；cal01截断率为0，不以截断严重为理由 |
| 训练采样 | temperature=1、top_p=1、top_k=-1；建议 repetition_penalty=1 | 减少采样概率解释上的额外变换；与旧配置1.05的变化明确记录 |
| 评测 | 固定 greedy、n=1；另有固定诊断 probes | 评测随机流独立，不消耗训练采样状态 |

LoRA、冻结视觉编码器、BF16 optimizer state、量化训练均不默认为必要前置。若全参数方案无法满足资源预算，再单独设计与验收备选；所有对照使用相同的参数更新范围，不能假定 adapter 快照和原生 vLLM 权重同步已经支持。

**只用最多四卡，分支串行。** 每个分支使用同样的 world size、TP、精度和 batch。当前只有两卡空闲时等待资源；不能把两卡分支与四卡分支混作配对结果，也不占用他人的 GPU。

## 4. 奖励条件与 oracle 的边界

现有原生奖励不是单一的答案正确率。它把工具执行、Verifier score/confidence、折扣和 inner repair penalty 组成过程分，再加入 `alpha_out × ground-truth outcome`。Verifier 输出等模型生成 token 也进入训练 mask，工具 observation 和插入提示不进入 policy loss。共享参数因此确实可能使 Solver 和 Verifier 同时变化。[原生奖励实现](https://github.com/aiming-lab/Agent0/blob/f775b5101e62fe92976831adf4a21a38fcc0a767/Agent0-VL/verl/workers/reward_manager/agent0.py)、[原生 rollout](https://github.com/aiming-lab/Agent0/blob/f775b5101e62fe92976831adf4a21a38fcc0a767/Agent0-VL/verl/workers/rollout/vllm_rollout/vllm_agent0_rollout_spmd.py)

本设计建议 **主协议先保留混合奖励 `alpha_out=1`**。这与已跑通链路接近，并允许对一个确知的终局正确性项做错误注入。各项奖励和训练 mask 完整记录，不能称为无标签自奖励训练，也不能将工具“执行成功”写成“答案正确”。

主实验 Clean 指“没有这次人工注入”，其余原生 Verifier 的判断未被 oracle 全部替换。Full Rollback Oracle 只消除指定历史事件的注入，并重新执行后续正常训练；它不是所有未来 reward 都完美的模型。

`alpha_out=0` 的共享 Verifier 自奖励设置是单独的候选扩展，须校准、另冻结，并设计可审计的过程奖励错误。不能在主 Gate 阴性后临时去掉真实答案奖励，再把新结果冒充原协议。人工终局奖励误判也不能直接证明自然 Verifier error 的完整反馈链。

oracle 答案在主协议中供原生终局 reward 使用；额外的审计标签、注入位置、counterfactual history 和污染集合只由实验控制器/评测器读取，不进入 Solver/Verifier prompt、工具输出或训练选样。自然错误研究时，再单独定义延迟证据的可见时间。

## 5. 任务、数据与校准

先使用可程序判定的视觉任务，延续读图数值后的求和、乘积、大小差三种题型。图像中的数字不抄入问题文本；工具仅执行模型写出的代码，不读取 latent 答案。这样保留视觉推理与实际工具路径，同时可以客观区分错误答案、错误计算与格式失败。

建议四个独立 split：train 1024、calibration 64、固定 probe 128、held-out evaluation 256。它们是设计预算，不是已生成数据。按原始 scene/base-instance 划分，再生成版式或问题变体；同一场景的不同题型/版式不得跨 split。所有 split 记录 image、问题、答案、checker 与生成器哈希。

训练数据足以提供每轮4题的固定 task schedule；默认不根据某个分支的失败、reward 或学习进度调整选题。Gate 2 研究的是 trajectory/reward 的内生变化，不同时引入题目选择变化。后续 curriculum 实验另立协议。

probe 含两类：正常解题行为，以及固定正确/错误候选解答上的 Verifier 判断。候选上下文、trainable token mask、工具状态和行为类别定义在校准期固定，不能用受污染模型的输出重新挑“最有差异”的 probe。

校准只看 calibration 与工程收据。建议采用以下健康规则后再冻结：

- 能在图像/答案 checker 上超过机会或固定低基线；没有接近全错或全对的题型。初始参考区间为10%–90%，按题型分别报告。
- 显式终局答案解析率建议≥90%；Verifier JSON 解析率建议≥90%；上下文/响应截断率建议≤5%。原生 fallback 得分单独记录。
- 有工具执行和自然触发的完整 Repair 样本；至少20%的题目 group 出现非零组内 advantage，梯度和参数有限。遇到连续3轮全部 zero-advantage，暂停查因。
- 同一次校准应同时包含正确与错误候选；不能通过强制打开 Repair 标记或额外 reward 伪造链路健康。

这些是待冻结的工程筛选建议，非已证实的统计阈值。未达到时修订任务/长度/格式后另开校准 run，记录全部尝试；不看正式 held-out 结果选配置。baseline 不要求显著涨分，但若无法生成有效训练信号，就不能直接进入 Gate。

## 6. 一轮、一个 group、一个状态分别是什么

为避免 idea 的 `G_t` 和实际 batch 混用，实现明确区分：

- `G[t,q]`：同一个任务的 n=4 条 trajectory，GRPO 归一化单位。
- `U[t]`：第t轮全部 b=4 个 groups，进入一次 outer update 的训练单元。
- `S[t]`：完成t轮后的完整训练状态；update k 从 `S[k-1]` 开始，得到 `S[k]`。

idea 的“第k轮之前状态”映射为这里的 `S[k-1]`。初期 CSS 决策作用在 `U[t]` 上，任何 group 有不可复用的关键轨迹，保守重生对应整个训练单元。细化到 group/partial reuse 是后续扩展，不能拆散 group 后沿用旧 advantage。

完整状态包含模型、optimizer moments/step、scheduler、reference身份、训练计数、dataloader/schedule游标、driver/各rank RNG、采样流与工具环境。只恢复模型权重、重置 Adam 或从头 warmup，会改变因果对照。

主 horizon 建议 T=32、注入 k=8、检测时刻 d=T；即错误发生后继续24轮。它只能说明此任务与 horizon 下的保留/传播，不能称无限期影响。horizon=8的诊断只做工程与成本测量，不当正式 Gate；更长 horizon、其它 k/delay 作为预先列明的后续敏感性实验。

## 7. 最小接口与数据契约

v01 提供少量明确操作，直接调用原生 rollout、reward、advantage 和 updater，不先搭建四个 SCOR 模块。以下接口已接入独立run_baseline.py/native_adapter.py；当前引擎实际S0/S7恢复、U1/U8原生/重放及重复评测下一unit已经通过；D三个seed完整32步已验收，E实际配对更新及完整审计也通过；CLI见版本README：

| 操作 | 输入/输出 | 必须保持的约束 |
|---|---|---|
| `initialize(spec)` | 冻结配置 → S[0] | 完整 horizon 的 optimizer/scheduler；固定模型/processor/reference |
| `generate(state,tasks,stream)` | 当前状态 → U的 trajectories/Verifier/tool证据 | 原生闭环；完整group；记录真实采样调用 |
| `score(unit,revision)` | unit → 各 reward 分量与 token reward | revision只影响声明的分量；inner repair成本保留 |
| `update(state,unit,replay_mode)` | 原生 GRPO → 下一个state/更新收据 | 全组重算advantage；记录每个真实optimizer step |
| `snapshot/restore/fork` | 完整state ↔ 持久快照/新run | 分叉前验证所有state身份；不加载异配置分片 |
| `replay(unit,mode)` | 原记录 → 验证后的训练输入 | 明确哪些字段冻结、哪些在当前模型重算 |
| `evaluate(state,split,stream)` | 模型 → 逐case结果与固定probe分数 | 不改训练状态、选样和随机流 |

raw 记录使用 JSONL 事件及保留形状/dtype的 tensor 文件；CSV/图表由 processed 脚本重建。核心记录字段如下：

| 记录 | 关键内容 |
|---|---|
| RunSpec | version/run_id、父run/状态、模型全部文件hash、源码/依赖/数据/checker/prompt hash、horizon、seed与stream规则、硬件拓扑、奖励/重放/快照协议 |
| TrainingUnit | t、task/group/trajectory稳定ID、图像/上下文/工具状态、完整token及mask、角色段边界、原始Verifier文本与解析值、tool输入/输出、repair实际发生位置 |
| RewardReceipt | 原始各分量、注入/修订分量、最终token reward、完整组mean/std/epsilon、进入updater的advantage、正常评分与oracle评分的来源 |
| UpdateReceipt | optimizer实际步数、LR/scheduler位置、有限梯度/参数、model-score与采样分数来源、clip/KL/支持范围诊断、单元hash、计算完成与持久提交状态 |
| RevisionCertificate | 被修订稳定ID、发生k/发现d、旧值/正确值/证据、可见时间、目标选择规则、人工或自然来源 |
| CheckpointManifest | model/optim/extra/data/RNG全部文件hash与字节数、已提交step、父状态、world size、兼容身份、保护理由和完整证书 |
| Evaluation/CostReceipt | case级显式答案/fallback/格式错误/正确性、行为类别、各角色token/call/time、工具失败、同步耗时、分配GPU数、保存/加载成本 |

uuid/row位置只做内部索引。跨run使用 `task_id + occurrence + trajectory_slot` 配对；图像、token、mask、组成员及数据顺序也要核对。只匹配文本 prompt 的现有 step1 replay 检查对未来视觉实验不够。

签名分三层：基础学习条件身份（模型/数据/GRPO/拓扑）、实验协议身份（注入/分支/重放/seed）、状态完整身份（父状态和已完成路径）。允许合法实验分叉，但仍拒绝误恢复；不能把 corruption 加进一个万能签名后，让 Clean checkpoint 无法作为 Corrupt 的合法父状态。

## 8. 历史轨迹重放：必须先说清训练算子

原生训练器在生成后重新调用当前 actor 的 `compute_log_prob`，把结果写成 `old_log_probs`，再执行 GRPO。已有保存字段证明了“真正进入 loss 的分母存在”，没有证明保存了所有采样调用的实际概率；temperature、repetition处理、推理精度和上下文都会影响这一点。[原生训练器](https://github.com/aiming-lab/Agent0/blob/f775b5101e62fe92976831adf4a21a38fcc0a767/Agent0-VL/verl/trainer/ppo/ray_trainer.py)

建议主重放协议为 **native reanchor**：固定历史图像/token/工具observation/mask/group/reward，在当前修复模型上重新计算 loss 分母与reference分数；reward修订时全group重新计算advantage，再调用原生loss和optimizer。每个outer update只在开始时计算一次分母，不能在每个micro batch刷新。

这是“强制历史数据重放”的干预算子。对历史轨迹，它并非从当前policy采样的on-policy更新，也没有保证对当前分布无偏。Gate 1 的固定数据控制和 Direct Correction 都明确使用这个条件，结论不能自动推广到任意off-policy更新方法。

同时保存 `historical_actor_logp`、`current_anchor_logp` 和实际 `sampling_logp` 的不同来源；禁止统称 behavior logprob。若希望做源policy分母的敏感性对照，必须先采集各角色每次调用的实际采样上下文/概率，并验证与训练条件的对应；clip本身不提供历史数据可复用保证。

v01 必须完成“同状态、同unit、无修订的重放与原生update一致”测试。Gate 4/6 还必须报告重放支持/clip诊断，并做可执行的分母协议敏感性检查；如果修复差距主要来自重放实现或分母选择，不能写成后代污染的独立证据。TRA 以后可用这些数据研究 false reuse，而不是用高似然直接证明安全。

## 9. 分叉、随机性与无操作对照

每个seed block训练一个共同prefix，固定U[k]，从同一个 `S[k-1]` 分出Clean/Corrupt；只改目标reward。其后的 matched streams由 `seed,t,task,slot,role,inner_step,repair_attempt` 生成请求seed，不包含branch名。额外clean branch用独立stream测自然波动。

请求seed不是逐位复现保证。分支行为不同会改变调用数，GPU/vLLM也可能非确定；必须记录请求身份、实际调用顺序和输出hash。固定数据实验直接读同一个完整unit，不能通过重新采样同seed冒充数据相同。

先做四个无操作对照：未注入/注入delta=0；同状态同unit的原生/重放；不中断/恢复后的固定下一unit更新；多次评测前后下一次训练rollout是否变化。任何超出预定数值容差的偏差都先查实现，再估计stochastic floor；不能把实现故障归为正常随机性。

## 10. 代码组织与版本继承

保持开发规范的版本隔离。建议未来v01源码增量如下；设计日仅规划；2026-10-11已创建并测试入口/记录/恢复/重放/干预/评测模块，当前验收见baseline_report.md：

```text
docs/v01/
  baseline_design.md      # 本设计
  gate_protocol.md        # 为v02–v07准备的实验协议
  design_review.md        # 需求审查与风险
  baseline_freeze.md      # 通过校准后才写的冻结记录
  baseline_report.md      # 实际baseline运行后才写的报告
solutions/v01/
  README.md
  src/
    run_pilot.py ...      # 保留现有工程入口
    run_baseline.py       # 新的正常训练入口
    native_adapter.py    # 原生phase接口、采样调用与角色记录
    records.py           # 稳定ID、签名、不可改写收据
    replay.py            # 明确的历史unit重放
    interventions.py    # 单事件注入与证书；不实现HED
    evaluate.py          # 逐case评测与固定probe
    checkpoint_store.py # 按新模型实测预算、完整state分叉/恢复
    configs/             # 待验证候选与冻结配置
    tests/               # 关键一致性/故障/泄漏检查
  inputs/<dataset_id>/
  outputs/<run_id>/{raw,checkpoints,processed}/
```

Gate vXX 在自己的 `src/` 中保留一份固定baseline引擎，并增加该Gate入口/配置/分析/测试；自己的 `docs/` 有方案、冻结、审查与报告。不得 `import solutions.v01...`，也不在v01目录堆六组开关和Gate业务代码。

继承时记录baseline release commit、engine文件hash和原生patch hash；各版本的模型/数据与单位定义保持可比。数据和分叉快照在本版inputs物化为校验过的副本或只读硬链接，拥有独立路径；小manifest记录来源/依赖。只读硬链接只能复用不可改写内容，禁止原地更新共享inode。

`.runtime/`只放共享依赖、原始模型和可重建的上游checkout。原生修改由各版src内的适配/patch清单产生，runtime副本记录版本和hash；不把科学实验逻辑藏进runtime，也不能让串行Gate静默覆盖另一版本的上游适配。无需先发布一个通用框架或建跨版本共享业务包。

修复引擎bug时先记录影响范围，再对需要复核的版本做同一修订、新run_id。旧版本仍被引用时不归档、不删输入依赖；已结束版本归档时按规范同时迁移docs/solutions并写版本说明。

## 11. 快照与真实资源预算

间隔法控制快照份数；每份大小由实际参数、dtype、optimizer state等决定。新模型不能直接复用3B的55/50GiB门槛，更不能直接降低门槛跳过空间检查。[现有策略](checkpoint_policy.md)

先在小规模试验中测完整快照Q、每轮raw增量R、模型新增下载M、保存/加载峰值与耗时。FP32参数加两份FP32 Adam moments的粗估为 `12 × 参数数` bytes，再加buffer/元数据；最终以完整文件实测为准。只存HF/BF16权重不能代替训练恢复快照。

建议普通保存间隔J=8，最近一份滚动保留；分叉前 `S[k-1]`、检测/最终结果状态及仍被其它实验引用的状态明确保护。初始S[0]需新增可验证的初始化状态协议，现有 `--checkpoint-pin-step 0` 不支持，不能直接使用。

单分支的空间检查至少满足：

`剩余空间 ≥ 新模型M + 原始记录预算 + 已保护状态预算 + 滚动旧快照Q + 正在写的新快照Q + 安全余量`

例如保留一个分叉状态和一个滚动状态，新写时已有三份Q共存；另有检测状态或多个未消费分支终态时还要增加。**串行运行只减少并发显存，不自动减少历史快照和raw的累计占盘。** 每次分支启动根据资产依赖清单计算实际峰值，分析完成/依赖物化前不删旧状态。

仅作预算例子：如果新模型实测Q=25GiB，三个槽位就需75GiB，尚未计模型、raw、检测状态和余量。设计日余盘约32GiB、3B快照43.1GiB；2026-10-11该旧快照已授权退役，当时余盘926.804GiB，cal01实测S0约9.10GiB/S1约25.56GiB。因此2B即使降低单份大小，也不能预先宣称现有空间足够完整Gate。需要先落实本项目获准的存储空间；新运行不删除无验证备份的被依赖快照，不假定可写其它盘。

快照完成采用原子提交和完整文件校验，先确认新状态可恢复再清理本run旧快照。跨run输入/保护点不能按“最近N份”误删。间隔内计算收据保留，但模型未提交；中断后新run从最后完整状态重算未提交段，汇总器按实验单元身份去重。

当前3B引擎真实完整S0/S7恢复与固定下一U1/U8更新精确一致已经验收；通用sampler逐位恢复未知。正常32步中的滚动间隔替换及全部终态仍待D；历史pilot/旧2B与CPU模拟不替代这些实际结果。

不可重算的原始响应/指标每步提交；已持久化模型阶段不重复计算。间隔内未持久化模型段的重建与阶段恢复明确区分：保存重建来源、原单元身份和新attempt ID，核对固定输入，禁止把两次计算当成两次独立实验。若采样不能按固定输入/流可靠重建，正式恢复只能到已经验证的边界。

## 12. v01 的实施顺序与退出条件

| 阶段 | 实际工作 | 退出条件 |
|---|---|---|
| A 兼容性与容量 | 固定2B revision/全部文件hash；原生多模态/工具/Verifier/Repair；一次GRPO；实测内存/完整快照/raw | 任务能力健康，更新有限，容量预算成立；否则停在校准 |
| B 完整状态 | 新model预算、周期保存、实际中断恢复、晚期分叉、S[0]协议、固定下一unit/no-op对照 | 真实GPU state一致性通过；不以证书文件替代分片 |
| C 记录与重放 | 角色/采样/奖励分量记录；稳定ID；同状态unit重放；显式答案与逐case评测 | 原生与重放等价、数据/答案边界清楚、指标可重建 |
| D 正常baseline | 独立校准后冻结；建议3个预列seed各32轮，评测0/8/16/24/32 | 全部完成或按停止规则报告失败；原始收据/最终state齐全，无选择性保留 |
| E Gate准备 | 同prefix同U[k]的正确/错误reward各一次真实更新；校验只有目标分量改变 | 干预接口与分叉有效；不要求错误长期保留 |

**D通过就可对导师说 baseline 已立起来；E通过可进入Gate 1。** “已立起来”指正常训练对照与恢复/评测可靠，不能说六个Gate已通过，也不要求当前任务涨分。若只有A的一次正常更新，仍应称工程试跑。

建议baseline seed列表为20261011/20261012/20261013；Gate推断使用事先列明的5个新seed blocks（例如20261101–20261105），不复用校准结果当正式重复。数量必须结合实测单步时间和预算最后冻结；若只完成两seed，只报告探索性差异。

每次试验启动前检查共享GPU/磁盘/环境身份；超过资源预算、非有限数值、状态不完整、oracle泄漏、group错位立即停止。统计结果不足以判定时记“不确定”，不无限增加seed或修改误差强度寻找阳性。完整开发、验收与Gate判定细则分别按本文件和 [gate_protocol.md](gate_protocol.md) 执行。

2026-10-11校准修订：2B尝试未通过协议与容量，当前模型为固定Qwen2.5-VL-3B-Instruct；cal05实测完整S0/S1，cal06初始显式61/64、Verifier解析127/128，仍因求和20/22、差值19/21超过.9 headroom拒绝。cal07候选题型定义为三值之和、三值乘积、最大值减其余两值之和，所有split同步checker与问题，重新生成数据身份；不降低门槛，不据held-out选择。本文前述2B/range属于早期候选，最终冻结以实际通过六项证据的候选为准。

正式运行前已冻结的资产依赖规划（证据见baseline_freeze）：三seed均保护S0/S7；首个seed另保护S8，供D之后E与原生U8更新作实际对照，另两seed保留S0/S7和滚动终态。预算要同时计入所有已保留校准及三个consistency分支、三个完整baseline终态和E额外错误分支；正确分支复用首seed nativeU8/S8，不能只按单run峰值判断空间足够。

2026-10-11 cal09校准候选：max_pixels32768/min3136、原生slow processor统一Torch与vLLM实际pixel张量；视觉原生FlashAttention明确deterministic=True，环境完整pip/Python hash进入学习身份。任务/checker、原生GRPO/AdamW、四卡/TP2、full32与验收阈值不变。源码/数据/模型和环境以同身份六证据通过后冻结；实际结果及阴性见[baseline_report.md](baseline_report.md)。E正确分支复用第一正常seed已完成native U8与保护S8，错误分支从同S7/同U8执行一次GPU更新，避免重算已完成控制。

2026-10-12实际状态：六项同身份冻结证据和累计存储预算已通过；[正式冻结记录](baseline_freeze.md)及原始JSON SHA固定。三个正常seed均已完成32步、五次256case held-out评测和完整S32，全部return0；E从第一seed同S7/U8执行一次真实更新，固定输入/唯一terminal+1/全组adv/其它组/完整S8审计全部通过。六个Gate仍未执行。
