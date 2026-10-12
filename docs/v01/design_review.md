# v01 深度设计审查

日期：2026-10-10（Asia/Shanghai）。审查对象为 [baseline_design.md](baseline_design.md) 与 [gate_protocol.md](gate_protocol.md)。以下第1–6节保留2026-10-10设计审查的原始状态；最新真实实施进度见文末及baseline_report，不能把历史“尚未实现”当当前结论。

## 1. 用户要求逐条对应

| 用户要求 | 设计落实 | 验收/当前状态 |
|---|---|---|
| 先深入设计，结合idea和后续Gate | 每Gate列假设、竞争解释、分支、指标、判定、失败后动作 | 两份设计已写入；研究结果尚无 |
| v01就是自己的baseline | v01正常闭环与实验基础；v02–v07为Gate1–6 | 旧v02 baseline/v02–v08路线取消；未创建未来版本实现 |
| 后续只有四卡 | FSDP4/rollout TP2候选；所有branch串行；预算包含修复与oracle构造 | 未承诺四卡能完成任意模型/规模；2B须实测 |
| 可以换模型，以机制对比为主 | 2B优先候选，固定全部方法的模型/训练范围/GRPO/评测 | 原论文7B/SFT/绝对分数不再是前置；模型仍未冻结 |
| 保留Agent0-VL核心链路 | 视觉、共享Solver/Verifier、实际工具、inner Repair、原生奖励、GRPO、后续生成 | 3B工程证据保留；新baseline尚待验收 |
| 按开发规范组织各版本 | 各版docs/src/inputs/outputs独立；固定引擎快照、输入物化、来源hash | 本次只新增v01文档；未跨版本import业务代码 |
| checkpoint使用间隔法 | 定期滚动+分叉/检测保护点；按峰值和依赖预算，不按份数平均 | 原3B策略仍原样；新模型/step0/跨run保护尚待实现 |
| 使用服务器遵守服务器.txt | 设计沿用共享账号/个人目录/最多四卡/不动他人任务 | 本次未连接服务器；不打印或发布凭据 |
| 更新最新科研开发规范 | 用户新正文导入项目，保留既有服务器第9条；新增强制代码正确性要求落实到开发验收；AGENTS/CLAUDE均指向实际规范文件 | 已更新；原始来源与hash记录在CURRENT_STATE |

## 2. 挖洞式审查

P0表示可能破坏核心结论/评测；P1表示实施前必须补齐；P2表示解释或局部分析限制。下面区分已在设计中修正的主张与仍需实际验证的问题。

| 级别 | 问题 | 处理与状态 |
|---|---|---|
| P0 | 把模型分叉当成能力损害 | Gate2/3分离；同held-out、clean-clean、case级终局评测。设计已修正，实验待执行 |
| P0 | 只恢复权重，Adam/scheduler或数据游标不一致 | 完整state与相同full horizon；无操作/晚期分叉GPU验收为前置。尚未按新协议验收 |
| P0 | 把旧old_log_probs称实际采样概率，或重放历史轨迹当on-policy | 明确native reanchor干预及三种分数来源；Gate4/6分母与支持敏感性。无偏性不承诺，关键验证待实现 |
| P0 | 给修复方法泄漏未来oracle或held-out标签 | 固定可见时刻/修订证书；独立评测；GT混合训练与额外oracle字段区分。须做实际泄漏检查 |
| P0 | Gate6缓存Full Rollback数据被当作当前policy数据；漏计生成费 | 6A数据上限与6B真实生成分开；成本转记和oracle研究费用单列。设计已修正，实际可行性未知 |
| P0 | 各自路径的gradient差/文本变化被当精确污染集合 | 同完整参考state的有限update与suffix替换；局部proxy不能冒充oracle；组合充分性Gate6验证。真实唯一集合仍不可直接观测 |
| P1 | 2B只证明支持架构，却声称训练/快照在32GiB能运行 | 模型候选、不直接改配置；实测完整Q/raw/内存/权重同步；受保护状态共存预算。存储资源尚未落实 |
| P1 | 现有pilot入口≤8步/step1重放/固定3B门槛不足 | 保留旧入口；新增baseline入口与state协议。尚未开发 |
| P1 | 只有两个seed或同组4条轨迹被当多次重复 | 按seed block配对，建议5组、固定T主终点；区间宽则不确定。数量/统计法待最终冻结 |
| P1 | 为追阳性事后换题、换alpha/阈值/错误强度或增加seed | 独立calibration、预列方案、全部尝试和不合格目标记录。冻结文件尚不存在 |
| P1 | 原生长度/格式/Repair触发不足造成小模型假阴性 | 任务能力与格式/truncation健康筛选、完整Repair样本；只在calibration调配置。新2B健康性未知 |
| P1 | 稀疏集合仍需全部重训；搜索/IO可能吃掉收益 | 全部update计费、先测成本下界、Fixed/Adaptive-W对照、6B端到端成本。收益未知 |
| P1 | Gate5替换试验漏计一侧克隆，参考状态重建费用缺失 | 续接审查已改为两侧共H(H+1)次update；H=24为每seed600次，另计重建/floor/评测/IO。预算公式已手工枚举核对，实际耗时待测 |
| P1 | 单点oracle替换漏掉联合影响；hybrid路径改变集合 | 明确参考路径范围、边界单元保守处理、Gate6组合验证；不把残余影响相加称已证明安全 |
| P2 | 原生混合reward被描述为纯自奖励；工具成功等于正确 | alpha_out=1明确标注，分量记录；自然Verifier机制后续另测。解释范围已限定 |
| P2 | 同文本不同图像合并评测；fallback读到Verifier数字 | 用稳定case/image ID，Solver显式答案与fallback分列。旧四题结果不改写；新评测待实现 |
| P2 | sampler seed等于逐位恢复，或把MFU=0用于成本 | 请求seed/实际输出记录、固定unit重放、allocated GPU-hours；采样逐位复现仍未验证 |
| P2 | 模型revision路径名当完整模型身份 | 冻结全模型/processor/tokenizer文件hash与依赖；合法分叉身份和恢复兼容身份分层。新签名待实现 |

## 3. 假设与决断审查

已由用户确定：v01自有baseline；最多四卡；可换模型/配置；机制对比；版本隔离与服务器规则。设计不能把这些再写成待审批项。

本次建议但未冻结：2B全参数；原生混合奖励为主；每轮4题×4轨迹；32轮/k8/d32；train/calibration/probe/eval为1024/64/128/256；校准健康比例；三baseline seed、五Gate seed；2个百分点容差、20%成本省幅；native reanchor主重放；间隔8步。任何修订须写原因、新hash和时间，并在正式运行前冻结，不能追溯为旧run配置。

关键保留的未知：小模型是否具备正常共享Verifier/Repair能力；真实快照/raw大小和可用资源；历史重放与实际采样分布失配；有限horizon功效；集合交互；真实选择性修复能否省成本。上述未知只被设计成可检验问题，没有被写成已解决。

范围选择：先用可程序判定视觉任务及人工终局reward误判，减少模型/环境与审计混杂。这能验证历史错误的一般路径机制，不能直接代表自然Verifier错误、无标签自奖励、开放环境或所有VLM。保留正常共享Verifier过程分，使机制链路完整；自然错误与参数分离干预在后续研究。

## 4. 按科研Review维度检查

| 维度/权重 | 本设计能改进什么 | 仍需什么证据 |
|---|---|---|
| 问题科学性25% | 关闭/开放数据反馈、完整状态配对、Direct与Descendant分离 | 六Gate实际效应、替代解释与阴性结果 |
| 创新性30% | 把修复对象落到历史训练路径；保留窗口等强对照 | 直接相关工作与正式算法比较；不能用六Gate本身宣称新颖性已成立 |
| 实验可信度20% | 状态/奖励/数据契约、独立评测、配对seed、oracle成本透明 | 新模型实际恢复、重放等价、采样/格式与泄漏检查 |
| 结果分析25% | 预列主要终点/容差/停止规则，记录有限范围与资源失败 | 完整逐seed结果、区间、失败与成本frontier |

不对尚未执行的设计打“通过率”或以总分替代证据。v01最先应补的是新模型资源/任务健康和完整状态、重放、评测验收，而不是提前实现HED/CSS/TRA/RR。

## 5. 本次完成与未完成

完成：以idea和原生实现为依据的baseline设计、六Gate协议、代码/版本/存储组织、验收与审查；README、CURRENT_STATE和变更记录同步。

未完成：新模型下载与兼容性试验、候选配置冻结、正式baseline入口/数据/记录接口、真实周期恢复/晚期分叉/重放、正常baseline运行、六Gate及SCOR四模块。现有3B事实、源码、配置、签名和运行产物保持原样。

## 6. 新规范要求的代码正确性验收

用户新规范“代码开发规范第5条”要求从设计到代码、从代码到设计审查，并以反例/手推/参考实现验证，不以可运行或已有测试数量替代正确性。该条本次已导入；下面列未来实现的独立判据，**不是已通过的测试结果**。

| 调用链/对象 | 独立审查与反例 | 必须保留的真实证据 |
|---|---|---|
| initialize → scheduler → prefix → fork | 用完整T与误用短prefix horizon对照，检查LR轨迹；在非零Adam moments时比较完整/仅权重恢复 | 生效config、参数组/optimizer step、scheduler状态、固定下一unit的更新差异 |
| native generate → role segments → updater masks | 含不同图像但相同文本prompt；插入tool observation、非法Verifier JSON、提前终局、长Repair及截断 | 每角色实际输入/输出边界；trainable mask；图像hash；异常路径和实际policy loss使用位置 |
| reward revision → GRPO | 对4项小group手算sample std/epsilon和全组advantage；零方差、只改一项、给全组同加常数；与固定原生函数互证 | 原tensor/修订tensor/组员ID、手推参考、原生输出和差异；其它group不受影响 |
| replay → compute_log_prob → actor update | 同状态同unit复现原生；刻意用相同prompt不同图像、旧mask、错group、不同source模型；当前分母与source分母区分 | 完整payload hash、模型评分来源、实际采样参数、clip支持诊断、拒绝错配的异常收据 |
| snapshot → commit → retire → restore | 新快照少分片、保存中断、缺optimizer、step0/late fork、不同world size、旧保护点被引用、磁盘写满 | 完整分片hash/字节、写盘提交顺序、实际恢复、旧状态仍存在、失败与重建来源 |
| evaluation → parser → aggregation | Solver答案与Verifier confidence相反；缺显式答案、不同图像同prompt、重复case、评测前后训练RNG | case级答案来源、fallback/格式指标、checker参考、聚合单位与训练状态不变检查 |
| gate controller → history → cost analysis | no-op、空/满污染集合、顺序打乱、缓存oracle费用=0、缺失败seed | 极端策略与DC/FR一致性；实际update/rollout/IO账单；计划表与汇总覆盖所有run |

配置审查必须覆盖最终Hydra展开值，而非只看候选JSON：batch/n/world size归一化、ppo_epochs、total steps与数据容量、model dtype、KL/entropy/reward项、sampler参数、长度与token mask、save/keep/pin及空间单位。重要判断对应源码函数/位置、真实收据和修复回归；尚未写的代码不虚填位置。

seed审查覆盖driver、data sampler、各rank actor/reference初始化与vLLM每角色请求；仅设置driver的random/torch seed不算全部生效。人工主错误须有有效显式错误答案，格式缺失不替代语义错误；把这一反例加进目标筛选与证书检查。

审查由独立判据完成，不等于必须增加代理。本次未改训练代码，不重新宣称旧测试通过了新协议。正式baseline和Gate依赖的实现若仍有未闭环P0/P1正确性问题，不启动依赖实验；科学假设本身的未知由Gate验证，不混同为代码缺陷。

## 2026-10-12实施同步

2026-10-12实际状态：六项同身份冻结证据和累计存储预算已通过；[正式冻结记录](baseline_freeze.md)及原始JSON SHA固定。三个正常seed均已完成32步、五次256case held-out评测和完整S32，全部return0；E从第一seed同S7/U8执行一次真实更新，固定输入/唯一terminal+1/全组adv/其它组/完整S8审计全部通过。六个Gate仍未执行。
