# 六个 Gate：与 v01 baseline 衔接的实验协议

日期：2026-10-10（Asia/Shanghai）。状态：**设计提案，六个 Gate 均未执行**。版本分配为v02–v07；本文件先保存在v01，冻结后各Gate在自己的版本文档中固定条件与判定。基础链路、资源和状态定义见 [baseline_design.md](baseline_design.md)，研究问题来源为 [idea.md 第10节](../RSI-Agent/idea.md)。

## 1. 假设、竞争解释与共用条件

| Gate | 可证伪假设 | 必须排除的竞争解释 |
|---|---|---|
| 1 | 单次错误update的影响在固定后续数据下仍保留 | optimizer/RNG不一致；固定数据实际不同；只是极短暂数值偏移 |
| 2 | 错误模型生成的数据进一步放大分叉 | 正常随机rollout；任务schedule不同；只有措辞不同 |
| 3 | 分叉造成实际能力损失 | 只改参数/文本、不改能力；显式答案解析失效；评测选择偏差 |
| 4 | 修正起始事件并重放历史数据，仍不足以恢复 | 重放bug、分母协议、重置Adam；给Full Rollback额外证据/训练预算 |
| 5 | 有意义污染集中于可选择的非连续/少量单元 | 简单连续窗口已覆盖；局部gradient不同但末态无影响；多个弱影响累积 |
| 6 | 理想选择性修复接近Full Rollback且省成本 | 预先生成oracle数据未计费；混合路径失配；漏算全部重训/存档/审计成本 |

初始主条件建议：同一2B候选最终冻结模型、原生混合奖励、四卡同拓扑、每轮4题×4轨迹、T=32、k=8、d=T、固定题目schedule。32轮只是有限horizon；另做长程实验才扩大解释。主错误类型先用一次终局false-positive，不同时扫模型、优化器、多错误率和多delay。

跨方法共享初始模型、S[k-1]、U[k]、reference、optimizer/scheduler、task schedule、seed配对规则、held-out/probes和可见修订证据。常规训练到同一个T，发现错误后的各方法恢复到同一个逻辑T。Continue Training就是此前照常训练得到的S[T]，不是给它额外未来题来与别的修复终态比较。

所有分支串行使用四卡。可复用跨Gate资产须校验配置/engine/data/state哈希，物化到本版inputs并记录来源；不能因为run名字相似就复用。正常clean history、corrupt history和Full Rollback history各有独立身份。

## 2. 错误如何注入，修订如何发生

**主干预在inner-loop结束、最终unit已固定之后，GRPO之前进行。** 它只改一个trajectory的终局reward分量，不改token、图像、工具结果、Verifier文本、Repair路径和其它reward项。

建议false-positive：从U[k]中具有有效显式Solver终局答案、且客观终局得分为0的轨迹，按预先固定的稳定ID顺序选择第一条，将该分量改成1；alpha_out=1时总return加1。缺答案/截断/格式失败不作为主语义错误目标，不能用fallback碰巧得0替代。false-negative（1→0）是预列的次要实验。旧试跑的任意delta=-2仍是工程证据，不混入这个主协议。

目标选择不得查看后续损害。如果k轮没有合格目标，记 `no_eligible_target`；不临时往后找一个“更容易造成损害”的k。报告全部计划seed与不可用seed，结论限定于存在目标错误轨迹的条件。是否预设额外seed补充，必须在运行前规定上限。

保存未注入/注入reward tensor、目标ID与分量、组内mean/std/epsilon/全部advantage；保证只有目标分量变，**同group的所有advantage必须重算**。算法调用固定上游实际使用的 `core_algos.compute_grpo_outcome_advantage`，不擅自换成另一个同名Agent0 helper或忽略std约定。

修订证书在d=T交给所有修复方法，只揭示同一个历史错误及其正确值。其它oracle污染集合仅Gate5/6可用；Full Rollback与Direct Correction不读取held-out标签来选修复策略。人工reward override是受控错误，不能称原生Verifier自然产生的误判。

## 3. 共用测量、统计单位与判定

| 测量 | 定义与用途 |
|---|---|
| P[t] | held-out按case的显式最终答案正确率；fallback结果另列。答案取自Solver终局，禁止从最后Verifier JSON的confidence提取数字 |
| D_score[t] | 在固定图像/前缀/候选token/角色mask上，两模型对应log-score差的RMS；先按case/角色归约再平均。不称完整policy KL |
| D_data[t] | 相同task上生成结果的行为效果类别分布差异，如答案、工具成功/异常类别、Verifier接受/拒绝、是否Repair；使用固定总变差统计，忽略自由措辞差异 |
| Repeated error | 固定错误类型在独立probe案例上的发生率；同一道被重复评测的题不计多个独立样本 |
| Failure | 工具执行失败、格式/超时/截断、异常终局和task失败分别报告；不能以任一字符串变化定义失败 |
| B_stoch | 同完整起点、独立未来采样流的clean-clean对照；不同seed block还覆盖训练随机性 |

`D_score`使用固定候选的模型评分，既可测Solver也可测Verifier；它不是采样概率，也不因保留一个teacher-forced文本就证明行为等价。`D_data`从实际on-policy记录计算；轨迹slot只表示采样位置，不宣称两条不同轨迹逐token具有因果对应。

建议每Gate事前固定5个seed blocks；每block共享prefix和注入unit，再执行需要的分支。4条同group轨迹不是4次重复；一条训练路径上的24轮也不是24个独立seed。

主终点为T；0/k/k+8/k+16/T曲线用于解释持续性，不逐轮挑最显著结果。先报告每seed配对差，再报告配对seed均值的95%区间（方法在冻结时固定；若采用t区间，明确小样本分布假设）。只有5个seed时区间可能很宽，需据实记“不确定”。按case的重采样仅作固定评测集敏感性分析，不代替独立训练seed。

阈值需要校准后冻结。建议起点：能力差异/恢复容差 `epsilon_P=0.02`（2个百分点）；行为分数阈值依据独立clean校准的波动量与任务可接受偏移共同决定；有用成本下降 `s_min=0.20`。这些数字目前是提案。校准样本不足时不能宣称95%stochastic quantile准确；同时报告floor估计的不确定性。

每Gate只预列一个主判据，其它指标为支持/冲突证据。顺序检验、敏感性分析和多指标结果如实标注，不能把任一个偶然阳性当主Gate通过。结果有三类：支持、否证/达到无实际差异的证据、不确定。统计不显著不等于证明不存在效应。

## 4. Gate 1 / v02：固定后续数据，影响是否保留

先生成无注入的clean history作为固定数据源，保存完整U[k+1:T]及其reward。源branch在k之后的结果不用于选择注入目标或调阈值。

```text
同一S[k-1] + 同一U[k]
  ├─ 正确终局reward → Clean-fixed
  └─ 一条错误reward → Corrupt-fixed
两者之后都读取同一U[k+1:T]，按native reanchor执行更新
```

固定的是实际图像、token、observation、角色mask、group组成、顺序和reward；当前actor分母/参数/optimizer沿各自分支演化。后续不重新调用Solver、Verifier或inner Repair生成数据。这是关闭数据反馈的机制干预，不是通常的on-policy RL。

主检验是末态D_score是否仍超过预定数值容差/固定数据clean-clean floor与最小实际偏移；后段曲线与P差异辅助判断自然覆盖。不能因为注入后一步模型不同就认定“长期保留”。输出包括单位逐字段相同的证明、reward局部变化和组内advantage变化、末态probes及所有失败。

若影响在声明horizon内已落入无实际差异范围，记录该条件下自然覆盖；若样本不足则不确定。不直接进入昂贵修复算法；可以按事前规划检验其它delay/错误强度，但属于独立结果。

## 5. Gate 2 / v03：恢复on-policy，是否进一步放大

在同一seed block比较四个核心分支，并增加独立clean未来stream：

| 数据方式 | Clean | Corrupt |
|---|---|---|
| fixed | 共用固定history | 共用固定history，仅k的reward错误 |
| online | 每轮自己生成数据 | 每轮自己生成数据，仅k的reward错误 |

clean-online另开至少一个独立stream branch估计自然分叉；fixed方式也检查无操作数值floor。所有branch输入相同task schedule，唯独trajectory/工具/Verifier/Repair/reward可以随policy内生变化。

同一个D_score衡量fixed/online的模型行为分叉：

`A[T] = (D_online[T] - B_online[T]) - (D_fixed[T] - B_fixed[T])`

floor与效应都是测量值，保留负数，不先截成0。建议主判据为配对A[T]区间下界超过事前冻结的最小放大量；同时必须看到实际D_data超过clean-clean数据分叉，确认存在数据路径变化。参数距离增加或文本变长不够。

如果online只是随机分叉，或者fixed/online交互不成立，不能声称已证明Descendant Contamination。此Gate研究trajectory/reward反馈；它未单独识别“Verifier漂移造成未来误判”的因果通道。后者需要独立固定Verifier等干预，排在正式模块实验。

## 6. Gate 3 / v04：传播是否伤害能力

优先复用已校验的online模型/记录，避免为测同一现象重复训练；无完整终态资产时必须重建并核对，不能从旧acc猜测结果。

主效应为 `DeltaP[T] = P_clean[T] - P_corrupt[T]`，使用固定held-out。建议支持判据：配对区间下界超过epsilon_P，曲线及独立probe错误方向一致，差异超过clean-clean正常性能波动。固定任务family分别报告，不能只展示受损family。

Repeated-error、工具/行动失败及Verifier对固定错误解答的接受率是预列支持指标。若行为分叉但能力差异不足，记录“有分叉，未验证实际损害”；若只有预列次要指标受损，明确对应能力维度，不换成主准确率损害结论。

不要把初始P=0.75、四题前后不变的现有工程结果当成此Gate。若检测不到损害，优先判断评测灵敏度/统计预算是否足够；不能按正式held-out结果重新挑题。

## 7. Gate 4 / v05：Direct Correction 能否解决

检测发生在d=T，所有方法知道相同历史reward修订。比较三个定义明确的状态：

| 方法 | 从哪里出发 | k之后怎么处理 |
|---|---|---|
| Continue Training | 历史Corrupt的S[T] | 无额外修复；该历史已正常继续训练到T |
| Direct Correction | 完整S[k-1] | 用同U[k]和正确reward更新；历史Corrupt U[k+1:T]/reward全部native reanchor重放 |
| Full Rollback | 完整S[k-1] | 用同U[k]和正确reward更新；之后每轮按当前clean模型重新生成、评分、更新到T |

Direct Correction不是在S[T]上再打一针corrected gradient，也不是只对目标trajectory做SFT。它恢复Adam/scheduler并按时间顺序重放全部历史更新；其额外成本包含所有这些update。

先确认Full Rollback实际恢复了Gate3测得的损害；否则这个修复参考自身不成立。主差距为 `Gap_DC = P_FR[T] - P_DC[T]`，辅以固定probe和失败率。若区间下界大于epsilon_P，可支持Direct Correction存在实际不足；若区间上界≤epsilon_P且行为也在容差内，则Direct Correction已足够；跨阈值为不确定。

必须做三项有效性检查：历史数据重放无操作一致；分母/clip/支持诊断；重放分母协议敏感性。actual sampling概率无法取得时如实记录限制，不能假装做了importance-corrected更新。如果DC差距消失于一个合理重放协议，先解释这个混杂，不能把主要损害归因于必须重新生成的后代数据。

DC足够时，SCOR复杂选择的必要性降低，不继续用更多模块“证明”它。可以保留历史错误检测/直接修订作为研究结果。

## 8. Gate 5 / v06：如何定义可检验的oracle污染集合

**同时拥有Corrupt和Full Rollback history，不等于自动知道C[k]。** 两条不同参数路径上的gradient差异混合了状态差异和数据差异；raw字符串不同也不等于值得重生。

主设计用完整counterfactual history构造一个可复核的有限horizon替换试验。对每个 t>k：

1. 恢复同一个完整参考 `S_cf[t-1]`，包含相同optimizer/RNG。
2. 两份克隆分别更新counterfactual U_cf[t]与历史U_hist[t]；相同native reanchor、group单位及奖励协议。
3. 先记录真实Adam更新后的局部probe差异，不能仅比较裸gradient欧氏范数。
4. 两份克隆之后都强制使用相同counterfactual U_cf[t+1:T]/reward，重放到T；测末态D_score和能力差异。
5. 用独立clean group替换同一参考位置并走相同suffix，估计这个替换试验自身的stochastic floor。

把步骤4得到的末态行为影响定义为本协议的 `I*[t]`，阈值为预先冻结的task偏移与步骤5的随机floor。阈值上方的单元进入C*；边界不确定的单元保守进入候选集合，并单独报告。

这是**在指定counterfactual参考路径和固定suffix上的oracle替换影响**。它不是所有可能repair路径的唯一真实因果集合，亦非已证明的future amplification公式。若只完成局部probe/gradient扫描，记 `local proxy`，不宣称oracle集合已验收；不能把估算的放大因子代入后称“精确oracle”。

按上述每个位置的两份克隆都执行替换update与完整suffix，单个seed的重放update总数为 `2 × sum(1,…,H) = H(H+1)`（H=T-k）：H=8为72次，H=24为600次。五个seed仅这部分就需3000次，另加参考状态重建、随机floor对照、评测和IO。若S_cf[t-1]未持久化，必须从完整保护点和冻结unit重建，并计入实际费用；不能把恢复参考状态视为免费。因此先用H=8测成本与恢复流程；这只是诊断。H=24的主试验必须按实测预算完成，资源不足则Gate5未完成，不能用短诊断替代。

报告 `|C*|/H`、index分布、最长连续段、覆盖集合的最短suffix、累计影响及不确定单元比例。优先比较Fixed-W/Adaptive-W：窗口长度选择只用校准或相同oracle信息，不用主held-out挑最优W；最终在匹配恢复容差下比成本。

稀疏但全部集中在一个短后缀，支持简单窗口而非复杂CSS。若集合几乎覆盖全部历史，选择性重生空间很小；不把任意小于100%的比例称有意义稀疏。建议候选筛选条件为稳定少于一半单元，且实测成本下界可能节省≥20%；最终判据与Gate6的效果/成本联合。

单点替换不能排除多单元交互或弱偏差累积，C*本身也随参考状态而变。Gate6必须检验整个选择集合的组合充分性；未列入集合的总偏移不能未经证明直接相加成安全上界。若组合失败，先确认集合/支持问题，再判断科学假设。

## 9. Gate 6 / v07：理想选择是否有价值

保持idea的oracle实验，但分两层报告，防止轨迹和成本解释越界。

### 6A：counterfactual缓存数据的理想效果

从S[k-1]用正确reward修正U[k]；之后按时间顺序：t不在C*用历史unit/reward，t在C*用Full Rollback缓存的counterfactual unit/reward。全部update基于当前hybrid state，遵守相同replay协议。

此处的“正确数据”来自Full Rollback policy，它未必是当前hybrid repair policy会生成的数据。6A衡量理想数据供给下的效果上限，不证明当前policy下trajectory可复用，也不直接证明真正on-policy SCOR有效。

先检验三个极端：全部用历史数据应复现Direct Correction；全部用counterfactual数据应复现Full Rollback（在已验收的重放数值容差内）；C*为空/满时决策、group和reward均符合预期。失败先修实现/重放。

主要比较 `Gap_oracle = P_FR - P_hybrid` 与固定probe差距。接近需要non-inferiority式判定：差距区间上界≤epsilon_P；“差距检验不显著”不能当等效。Normalized Recovery只有FR确实优于Corrupt且分母超过预定小量才计算，保留大于1/小于0的真实值。

### 6B：给定理想集合，当前模型实际生成

使用相同候选C*，不取缓存替代真实rollout；需要Regenerate的单元由当前修复模型实际执行Solver、Verifier、工具和inner Repair，原生评分后更新。其它单元使用已声明的历史重放，并记录支持/clip/状态诊断。

初期保守以完整训练单元做Reuse/Regenerate，避免partial-reuse group混合引入新算法。Correct分支在有独立可审计reward与reuse判据后另测；不能把6A数据的“正确”当成TRA/RR已实现。

6B同样检验“全部Regenerate”与真正Full Rollback的一致性，并使用配对streams/独立重复处理随机差异。C*在hybrid路径可能失效；如需增加保守再生成，要记录新增单元及费用，不继续声称完美固定集合。

**6A通过只支持oracle数据上限；6B接近恢复且实测省成本，才支持进入实际CSS/TRA/RR的开发。** 6A失败时先审计操作性oracle集合与重放条件，不能直接宣称所有选择性修复均不可能。

## 10. 成本怎样算，才能在四卡上成立

所有修复方法都重走k到T的update。SCOR主要省生成、验证、工具与某些评估，不在本协议中省掉所有重训。

令H=T-k，每单元update成本为C_U，完整生成/验证/工具成本为C_G，选中重生m个单元，粗估：

```text
Full Rollback ≈ (H+1) C_U + H C_G + 恢复/存档成本
Direct Correction ≈ (H+1) C_U + 数据读取/恢复/存档成本
SCOR ≈ (H+1) C_U + m C_G + CSS/TRA/RR/probes + 恢复/存档成本
```

实际报告按role生成token/calls、Verifier调用、tool调用、repaired trajectories、regenerated完整groups/units、update次数、同步时间、加载/保存时间、allocated GPU-hours及总wall time拆分。GPU-hours按实际分配卡数×占用时间；不使用RTX4090原生MFU=0推导计算效率。

若C_U与IO占绝大部分，即使m=0也无法达到目标省幅，Gate6上限已经不够；不要先开发CSS再发现理论省幅很小。普通baseline的共同训练成本、人工事件审计成本、修复额外成本、oracle构造成本分别列出。

6A的oracle groups已经付过生成/验证成本，缓存读取不能当作零成本生成。它的当前运行GPU-hours是“缓存实验实测成本”；按Full Rollback原始ledger给采用的groups转记生成费用后，才可给“理想选择的预计部署成本”。这个预测与6B真实端到端GPU-hours分开。

oracle集合构造本身可能跑过完整回滚甚至大量替换试验。其真实研究成本完整披露；在理想可行性上限中可把完美集合视作给定，但结果不能说“包含集合搜索后仍省20%”。实际HED/CSS/TRA/RR的全部费用要在正式主实验重新计入。

建议6B支持判据：恢复差距区间上界≤epsilon_P，行为也在预定容差内；在同口径repair cost上节省下界≥s_min，且无新增明显失败。只达到准确率而支持范围异常、或只靠忽略oracle费用省成本，不作为通过。

## 11. 运行组织、停止规则与后续模块

每个实验单元用新run_id，父状态/引用history写manifest，原始结果不可改写。按seed串行运行所需branches，及时评测、保存必要终态，依赖物化后才按显式策略处理非保护状态。失败与 `no_eligible_target` 写入总计划表，不能只汇总成功分支。

资源预检失败不启动GPU；group/视觉输入错配、非有限值、修订范围越界、完整state缺失、oracle泄漏立即停止。无操作对照失败先修工程。阴性或不确定结果按上述判据报告，不以加大corruption、换主指标、增加未计划seed继续追阳性。

| 证据停在哪里 | 后续合理动作 |
|---|---|
| baseline可靠，Gate1不足 | 报告该horizon下覆盖/不确定；按预列条件补验证，不开发复杂修复 |
| Gate2不足 | 不声称数据反馈污染；研究直接修订即可 |
| Gate3不足 | 不声称能力损害；检查评测功效与适用任务 |
| Gate4表明DC足够 | 优先直接修订/重放，不增加选择模块 |
| Gate5连续窗口足够或集合很密 | 强化窗口/Full Rollback对照，限制CSS动机 |
| Gate6理想上限无优势 | 先排集合/重放/成本错误；若有效阴性，不继续堆HED/CSS/TRA/RR |
| 六Gate支持 | 再把oracle换成实际HED/CSS/TRA/RR；重新验收false revision/false reuse/漏检与全部成本 |

正式模块阶段继续保留Continue Training、Direct Correction、Reward Relabel + Replay、Historical Advantage Recompute、合理的Influence/Unlearning、Fixed/Adaptive-W和Full Rollback。自然Verifier错误、frozen Verifier、frozen Solver、参数分离、其它模型/任务/优化器用于检验机制边界；它们不全塞进当前v01，也不由人工终局错误结果直接替代。
