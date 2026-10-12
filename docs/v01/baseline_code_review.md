# v01 baseline代码一致性与反例审查

审查日期：2026-10-11；最终验收与文档核对更新于2026-10-12。审查范围为独立 baseline 引擎及其原生调用链，依据《科研开发规范》代码开发第5条执行设计→代码和代码→设计核查。当前12个引擎文件的SHA与已完成的cal11及三个真实GPU对照完全相同；旧15个pilot源码保持原貌。未发现阻止此引擎正式运行的未解决P0/P1。累计存储预算已真实通过并写six-witness freeze；三次正常32步和E实际配对均已完成，v01 baseline A–E已验收，六个科研Gate尚未执行。

## 逐句需求、代码与实际验收

| 用户要求/已确定方案 | 实现及反向核查 | 独立判据与实际结果 |
|---|---|---|
| “v01就是自己的baseline” | `run_baseline.py`独立入口，模型/数据/源码/环境/协议/父状态身份；不取消旧pilot约束冒充新实验 | cal11完整8个原生outer update及S0/S1/S7/S8；正式D三seed×32/五次256case/全S32正常完成 |
| “一次最多4张卡”及只用自己的目录 | `run_baseline.check_profile`、`preflight.inspect`、`record_gpu.record`；ROOT从版本源码推导 | 真实只用物理4–7，FSDP4/rollout TP2；源码/环境/模型/输入/输出均在服务器SCOR目录内；不操作他人任务 |
| “v01以及之后的文档如果有不合适的，也要同步” | 根/版本README，baseline设计/实施/报告/本审查，Gate协议与design review | 2B失败、3B最终候选、full32 cosine、Gate v02–v07和未完成项分开；不宣称Gate已通过 |
| “开始写并跑baseline”且不可降级为MVP | `native_adapter.install`、精确固定上游patch、原生actor `update_policy` | 图像→Solver→实际Python→共享Verifier→自然Repair→混合reward→GRPO/AdamW→新模型继续生成；cal11工具201次、自然Repair22次、32/32 groups非零；三正常run已各完整32步，五次评测和全S32/return0齐全 |
| 完整horizon/一次真实global更新 | `run_baseline.main`、原生FSDP scheduler及actor optimizer、adapter `optimizer_step` | 原生constant/warmup0首轮LR0反例已修为原生cosine/full32；实际8步optimizer计数1–8、used LR>0、全参数与持久buffer/Adam/scheduler有限、实际model SHA变化 |
| 四split和图像统计单位/不泄漏 | `prepare_baseline_data.prepare/verify`、`records.value_hash`、`evaluate.case_result/aggregate`、`probe.evaluate_verifier` | train1024/cal64/probe128/eval256的scene/case隔离、PNG SHA和实际视觉张量；期望答案/候选正确标签仅用于checker/评估，未放入Solver输入；正式held-out效果没有用于候选选择 |
| 只接受显式Solver整数终局 | `evaluate.integer_answer/solver_answer`及角色记录 | 小数、指数、分数、后缀、空boxed、末尾非法答案、Verifier/tool数字等反例；case重复拒绝；显式与native fallback分列，不把工具结果冒充Solver输出 |
| 实际train mask/角色边界 | `apply_baseline_instrumentation._append_tokens`与原生actor使用multiturn_mask | 根据语义角色独立重建mask，核对真实token切片和连续attention；训练tool/错role/token/attention缺口反例拒绝；cal11所有实际groups核验通过 |
| S0及非零Adam晚期完整状态 | `checkpoint_store.required_names/certify/commit/validate/retire`，adapter `save/load` | model/optim/extra各4分片、data/driver RNG/state/worker完整证书；缺分片、缺worker、同size损坏、粗mtime、错learning/外run/保护点反例；真实S0及S7恢复全worker参数/持久buffer/Adam/groups/scheduler/RNG和driver/cursor精确相同 |
| 恢复后不重复已完成批/随机流 | adapter `restore_driver_and_cursor`→原生加载器 | 真torchdata懒state_dict会消耗driver RNG的反例已闭环；先物化并核对游标、最后恢复RNG。CPU下一批0–3/28–31与下一次随机数通过；GPU真实S0/S7恢复及固定下一单元更新通过 |
| 原生/历史同unit更新等价 | `replay.load_generation`→原生`compute_log_prob`/reference→原生GRPO/actor | 六tensor/必需角色metadata/mmInputs、完成标记/group/payload SHA、dtype/shape/bytes；错图/slot/mask/缺字段反例。仅兼容原生counter同值int32→int64。真实U1、U8所有13训练tensor及全部non-tensor输入精确相同，更新后完整model/buffer/Adam/groups/scheduler和worker RNG精确相同 |
| 单个语义错误terminal outcome+1 | `interventions.false_positive`→原生reward→原生whole-group advantage | 只选显式错误、native acc0、非截断的第一stable ID；不看未来结果；无eligible明确记录。独立手算sample std及epsilon1e-6与原生一致，只一个terminal token改变、整目标组adv重算/其它组不变；delta0 negative-zero字节反例通过。实际delta1 E已从同S7/U8执行，唯一[5,260]terminal+1与全组adv、其它组及全S8审计通过 |
| 评测隔离及无操作对照 | adapter `evaluate_once/evaluate`、实际两个initial评测和delta0 | 两次真实64case initial评测、driver/worker RNG和data cursor核验；随后真实native U1的全部输入和完整更新状态与未增加评测的cal11精确相同，fullS1及正常退出；不只是比摘要或随机seed |
| 所有关键参数实际生效 | launcher/resolved_config、8份worker config、HF/native-vLLM processor及小GPU数学参照 | slow处理器min3136/max32768，真实grid[1,12,12]/pixels(144,1176)/36 merged tokens；Torch/vLLM tensor类型、形状、字节相同。HF-fast与vLLM-slow同grid却差.015008的反例已闭环；不能以constructor kwargs替代行为 |
| deterministic视觉及原生compiled entropy | adapter薄wrapper直接转发原生FA deterministic=True、原生torch compiled entropy | 8个worker均实传；BF16原生kernel前向/重复梯度和独立dense float64参照通过；entropy解析值及导数GPU1passed6.88s。完整U1/U8实际state精确证明特定对照，不推广为任意sampler逐位恢复 |
| 六项真实冻结门槛 | `baseline_acceptance.freeze`及launcher baseline分支 | 空/失败/异身份/正在运行/错参数/错source/错ENV反例；所有evidence路径与SHA、model/data/profile/environment和预列seeds必须一致。原生健康/恢复/重放/评测隔离四项已通过；完整累计预算已凭六份完整归档及实测future写入峰值通过 |
| 每步持久化、失败保留、成本可追溯 | records原子不可改写保存、Tracking hook完成标记、checkpoint双次内容hash、`baseline_acceptance.costs` | raw逐unit立即保存，模型只在已认证边界恢复；失败及partial不冒充完整。物理CSV是2秒采样下界且含设备总使用量；组件计时有重叠不能相加，GPU小时是分配卡数×wall，非利用率 |

## 主接口与参数核对

固定上游为`f775b5101e62fe92976831adf4a21a38fcc0a767`。实际patched原生trainer在reward后重算全组advantage，再调用原生actor，评测/保存后写metrics及完成标记；数据游标加载后global step自增一次。原生core_algos GRPO以轨迹reward总和、组sample std和epsilon1e-6归一化，保持原实现。原生actor loss、AdamW、reference及vision FlashAttention算法未替换。新增工具后的Solver cue使用实际stdout，不读取GT；原生_get_template_tokens已经chat包装，不再次包装Verifier/Repair。

正式候选：Qwen2.5-VL-3B-Instruct revision `66285546d2b821cf421d4f5eb2576359d3770cd3`；三位整数可见图像data08；四轮×单次256、总response buffer1536/context6144；batch4×n4、micro1、global一次更新、full32 cosine/warmup0/lr5e-7；native mixed reward alpha_out1。cal seed20262001，正常seeds20261011/12/13。learning源码/参数/完整pip与Python环境身份锁定；CPU21后唯一profile变化为49152→32768，已由cal11实际生效值与全部GPU对照覆盖，源码及测试未改。

## 实测证据与缺陷闭环

服务器`baseline_cpu_20261011_21/raw/all_tests.log`：82passed/1GPU诊断skip，14.15s；其`raw/source_hashes.json`记录源身份。本机50passed/3native模块skip，0.69s；skip不算原生通过。原生entropy GPU另行1passed6.88s。`baseline_processor_probe_20261011_04/raw/same_pixels.json`证明最终像素预算的实际张量一致。真实GPU证据在cal11 `processed/acceptance`四个已完成witness及其原始路径/SHA：

- `native_loop_health.json`：14项规则全部通过；训练截断6/128=.046875，初始每题型准确率18/22、18/21、15/21，保持原门槛。
- `actual_state_restore.json`：S0两次、非零Adam S7一次的实际完整GPU加载，正常完成后有认证终态。
- `same_unit_replay.json`：S0/U1与S7/U8真实固定输入、完整更新状态精确一致。
- `evaluation_rng_isolation.json`：两次initial评测、delta0与未来真实native U1/fullS1精确一致。

已闭环P1包括显式答案误解析、完整证书遗漏worker、重复chat framing、lazy loader随机流顺序、历史counter原生cast、dtype同值漏洞、delta0负零变化、fast/slow像素不一致、vision deterministic参数未实传、环境身份缺失、空/失效freeze。所有失败校准run的raw/source、日志、证书及小元数据保留。冻结前归档曾在lossless备份全部member SHA验证后回收展开副本；2026-10-12用户授权清理后，旧大权重、失败半写和旧备份已退役，当前六份完整状态见[storage_cleanup.md](storage_cleanup.md)。

## 未验收项与解释边界

D三seed×32及E同S7/U8真实delta1配对全部通过，最终CSV/成本/报告已重建，Git实施提交5573536已发布并核对远端。这些结论依据实际GPU运行和原始收据，不以CPU通过或calibration替代。产品UI不适用于此CLI/GPU任务；用户已允许使用真实CLI/原生/恢复冒烟验收。

cal11固定Verifier正确和错误候选的接受率均为0，是能力阴性观察；高JSON解析率不代表判断有效。native mixed reward包含outcome，不能称纯自奖励；历史native reanchor不是on-policy或无偏估计。teacher-forced actor score、vLLM返回分数和sampler分布不同来源分列，通用采样逐位恢复未验证。题型和pixel修订是校准期决断，不能冒充原始预注册或用户逐项确认。当前工程健康和可比较性成立不代表科研Gate、长期污染、泛化提升或SCOR算法有效。


2026-10-11后处理/串行入口补充审查：新增`analyze_baseline.py`和`run_baseline_sequence.py`，不改变12个native engine的任何字节、profile/model/data/environment。D固定三个seed各32步online，首seed保护S7/S8，E复用其nativeU8/S8并从同S7单delta1；任何失败保留scene，不覆盖旧attempt，monitor在raw实际创建后启动。D分析要求全部unit完成标记/位置/LR、五次256case held-out及fullS32；E独立typed输入、stable semantic-error/native0/nontrunc目标、唯一terminal+1、whole-group Python sample std/epsilon1e-6与其它组不变，容差atol1e-6/rtol1e-5在E前指定。原生9反例及CLI通过（analysis_cpu_20261011_02：9passed2.99s），无新增未解P0/P1。immutable补充review为cal11 processed/analysis_sequence_review_20261011.json；原native review原文/SHA仍保存在raw，未来GPU结果不能由此CPU检查代替。

2026-10-12冻结后核查：六项证据、profile/model/data/ENV/source SHA及预列seeds已由真实freeze固定；后处理与成本helper的最终审查为`processed/analysis_sequence_review_20261011_03.json`，CPU04实际12passed3.17秒，当前local engine12逐SHA匹配freeze。D前两个seed所有32unit、五个256case评测及全S32正常；后续结果只报告，不用于调参。

2026-10-12最终CSV导出审查：新增`export_baseline_summary.py`和五个独立反例，只读已完成D/E及原始cost/设备CSV。按三预列seed顺序、all32和five256case、count/accuracy一致性验收；独立手算三seed百分比点均值/范围，拒绝missing/repeated seed与missing评测，不按轨迹或重复case伪增n。E需actual GPU/full state finite和序列control ID一致；物理记录必须覆盖全部选择的卡，logical/per-worker phase覆盖必须一致；wall/组件/角色/采样峰值的不同单位分列。服务器`baseline_summary_cpu_20261012_01`实际配对9+成本3+汇总5共17passed5.20秒，raw source SHA及processed/review保存；12个frozen engine SHA全matched。没有新增未解P0/P1；完整真实D/E输出仍须终态后验收。

2026-10-12最终实际验收：三个预列seeds20261011/12/13各32步、五次256case及完整S32/return0通过。E从相同完整S7/Adam/RNG/cursor、读取相同U8；固定九训练tensor/全部visual-role-group metadata typed exact，唯[5,260]terminal outcome0→1、总return1→2，四mutable tensor按唯一reward和全目标组标准化变化，Python sample-std独立oracle匹配、其它三组字节不变。实际step8/LR4.432526133406842e-7、四rank模型相对正常S8均不同、全参数/optim有限、全S8/intentstop8/return0；实际paired audit passed。最终export CLI成功生成15行results/4行costs/32行roles/32行paired adv，三seed描述性n3，components重叠/GPU allocation/device sampling界限保留；case/update/cost和SHA实际复核，12引擎与原15pilot字节保持。工程A–E无未解P0/P1或必需未完成项；六科研Gate仍未执行，不推广为长期污染/能力损害/SCOR有效。

## 2026-10-12用户授权清理后的资产范围

本文件前述校准/controls/备份状态描述属于清理前的执行事实；实际原始证据与冻结JSON未改。用户明确清理后，仅保留三个正式S32、首seed S7/S8和E S8共六份完整训练状态；旧校准/对照/重复初始及中间状态大权重、失败半写、2B和旧压缩/本机备份与下载残片已退役，证书及small metadata保留。当前旧点不能完整恢复，cal06本机下载不再是待完成项。3B/环境/data08/所有raw与processed保留，结果与验收结论不变；详见[storage_cleanup.md](storage_cleanup.md)。新的校准/恢复复核若需要旧完整状态须另立新运行，不能用历史证书冒充现存模型。
