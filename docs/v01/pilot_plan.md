# Agent0-VL 小规模复现与验证

日期：2026-10-10（Asia/Shanghai）。这是试验启动前制定的方案；当前执行结果见 [pilot_report.md](pilot_report.md)。核心链路与恢复续训已完成，配对 GPU reward 对照因空闲卡不足未启动。

## 用户要求与验收

1. 在当前项目开发：代码、配置和说明保存在 `solutions/v01/src/`；上游代码与依赖置于 `.runtime/`。
2. 尝试服务器四张 4090：先核实实际型号、空闲显存、共享任务和工作目录；一次不超过四张卡。
3. 小规模复现：保留视觉输入、共享 Solver/Verifier、真实工具执行、Verifier 输出、内循环 Repair、原生奖励合成、GRPO 更新，以及更新后模型重新产生数据。
4. 小规模验证：先验收正常 rollout、有效 group advantage、参数更新和完整 checkpoint 恢复，再尝试受控 reward 修改；不把 smoke test 当作六个 Gate 或论文结果。

## 范围与研究条件

优先复用官方实现，不重写训练框架。固定上游 commit、模型版本、数据、随机种子和实际执行配置。最先尝试适合同系列的小模型配置；如资源允许，再试论文使用的 7B。缩小模型或数据的试验明确标为机制适配与工程验证，不能称为论文数值复现。

数据采用少量具备确定性答案的视觉题，用于检查真实链路与奖励注入。该数据不代表正式 benchmark；不据此得出泛化、后代污染长期存在或 SCOR 优越性的结论。训练中的 Verifier 过程奖励、终局 ground-truth 奖励与审计证据分别记录。

## 执行顺序

1. 连接与环境检查；收据写入本版输出。已存在主机指纹必须匹配；新指纹按 `服务器.txt` 要求先核实。
2. 固定官方源码，检查训练入口、数据 schema、奖励、存档和恢复逻辑。
3. 本地验证配置和数据准备；执行独立工具沙箱与奖励计算检查。
4. 服务器运行短训练，保存 rollout 原文、group 信息、实际 reward、behavior logprob、配置与版本；检查有限数值、非零有效 advantage 和参数更新。
5. 中断/恢复试验，核实完成的更新不重复执行、模型/optimizer/scheduler/随机状态恢复且配置一致。
6. 在单个明确更新边界修改 reward，使用同一前置状态比较未经修改与被修改的更新；条件满足时继续短程 on-policy 训练。该结果只作为后续 Gate 的准备证据。

## 记录与停止规则

每次尝试使用新的 `run_id`，保留原始日志与失败现场。每个阶段原子写入配置哈希、状态和完成标记。模型、依赖、配置变更另开尝试并记录原因。

遇到 SSH 指纹未验证、无空闲 GPU、工作目录权限不明时，不启动远程训练。遇到显存不足、数值异常、空 group advantage 或恢复状态不完整时，记录失败并修复；不扩大运行规模。缩小配置仍应保留研究所需的原生链路。

## 开发期间的决策记录

- 2026-10-10：用户授权尝试首次 SSH 连接，本次观察到的主机密钥保存到项目专用 known_hosts；未声称已由管理员核验。
- 实际服务器有八张 24 GB RTX 4090，计划最多使用空闲的 4、5、6、7，避开已有任务。
- 个人目录所在盘初始剩约 95 GiB。3B 原生 FP32 模型与 Adam checkpoint 约 45 GB，先保存一次完整 checkpoint，再以不产生第二份 checkpoint 的方式验收恢复后继续更新；最终模型未持久化的限制必须报告。
- 首次环境安装失败：vLLM 0.8.2 的精确依赖 xgrammar 0.1.16 在官方 PyPI API 返回 404。该版本官方源码 tag 仍存在，改为在独立目录编译相同版本；保留原失败日志，不升级 vLLM 或 Transformers 来绕过兼容性核查。
- 原生 CPU 集成测试发现：最初锁定的 math-verify 0.3.0 缺少源码要求的 `errors.TimeoutException` 和 `boxed_match_priority` 接口。临时导入适配未解决完整接口问题，随后检查官方 0.7.0 wheel，确认同时提供这些接口。依赖改为 0.7.0，并撤销临时评分源码适配，恢复原生评分代码。旧失败日志保留；重跑真实奖励与 GRPO 集成测试，不能用本机跳过的测试代替。
- `native_stop1_01` 首启在训练前失败：Ray UNIX socket 路径超过 107 bytes。临时路径缩至项目 `.runtime/r`，模型和训练逻辑不变。
- `native_stop1_02` 成功四卡加载 3.75B 参数，但自动启用的 vLLM V1 引擎无法在初始 0.35 显存预算下分配 KV cache。原生权重同步代码使用 V0 接口，因此显式设置 `VLLM_USE_V1=0`，并将缓存预算调至 0.75，重新检查实际峰值。没有 reward、参数更新或 checkpoint 结果产生，不能将这一轮记为训练成功。
- 后续显存峰值诊断将预算降至 0.55，并在原生 vLLM wake_up 前释放未使用的 PyTorch allocator 缓存。跨 DP rank 输出改为共同 padding 宽度；重复图像 GRPO 关闭 chunked prefill 后通过。真实 tokens、mask、评分和更新规则保留。
- `native_stop1_10` 完成原生 update 与完整 step1 checkpoint 保存，但随后在无 critic 的 GRPO 指标汇总中引用未定义 `values`，任务失败。只修复指标的 critic 条件后，`native_resume2_01` 从该 checkpoint 恢复，继续用下一批数据完成 step2，正常退出。补算的数据指标明确标注为事后分析，不当作丢失的原始 GPU 指标。
- 同一实际 batch 的离线 reward/advantage 检查完成；配对 GPU 更新试验 `native_corrupt1_01` 被资源预检拦住，尚未产生 GPU 错误分支。未扩大任务到论文全量规模或六个 Gate。
