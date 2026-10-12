# v01 baseline实施与验收清单

更新：2026-10-12。用户要求继续此前全部工作、同步不合适的v01及后续文档、自主解决阻碍。以下为当前验收，后面的时序记录只表示当时状态。

| 需求/阶段 | 代码或文档 | 当前实际验收 |
|---|---|---|
| 规范、用户TODO逐句对应、四卡与个人目录 | 科研开发规范、baseline_code_review、preflight/launcher | 当前实现双向审查无未解P0/P1；最多物理4–7，只写SCOR |
| 同步v01/后续不合适文档 | 根/版本README、design/implementation/report/review/Gate协议 | 当前配置与历史失败分开；v02–v07六Gate仍仅规划 |
| A原生持续训练闭环 | launcher/adapter/native pinned patches/profile | cal11 8实际updates、201工具、22自然Repair、14健康checks全部True；共享3B/原生GRPO/AdamW未替换 |
| B完整初始与晚期状态/恢复 | checkpoint_store/adapter | 真实S0两次与非零Adam S7一次全worker/driver/cursor精确恢复；fullS1/S8正常提交 |
| C独立数据/完整记录/同unit replay | prepare_baseline_data/records/replay | train/cal/probe/eval scene隔离；U1/U8全部13typed tensor及视觉/nonTensor输入、完整updated state精确 |
| C随机流/无操作 | adapter/evaluation/false_positive delta0 | 两次initial评测、delta0之后未来native U1/fullS1与原对照精确相同；通用sampler bitwise未知 |
| 累计state/raw/写入峰值预算 | launcher/checkpoint_store/storage witness | 冻结前cal05 S1本机备份及六份同server压缩副本完整验证；freeze累计预算420.5494≥418.6016GiB通过；旧备份已在2026-10-12授权清理，当前保留六份正式状态 |
| D正常baseline | run_baseline_sequence/run_baseline/baseline_acceptance | 六witness真实freeze通过；三预列seed各32步/五次评测/fullS32正常return0，D已验收 |
| E同S7/U8单reward真实配对 | interventions/replay/analyze_baseline | 9配对+3成本原生反例审计通过；D之后真实GPU delta1、同S7/U8、唯一terminal+1、全目标组adv/其它组和全S8审计passed，不要求科研阳性 |
| 原始输出、终态、成本、Git和报告 | summarize/costs/CSV/README/state/history | 校准/controls及三个正常run原始记录完整；D/E与全部CSV/最终成本报告已验收；实施提交5573536已发布，后续清理和三处同步见storage_cleanup |

产品UI不适用于此次CLI/GPU任务，已获用户豁免；实际CLI、GPU闭环及恢复冒烟已执行。不能以校准8步、CPU通过或只启动任务代替D/E验收。冻结后保持12个原生引擎源、profile/data/model/ENV不变；新增分析及串行入口不改训练算法或这些身份。

## 历史时序记录（当前状态以以上表为准）


更新日期：2026-10-11（Asia/Shanghai）。用户要求同步v01和后续不合适的文档，写并运行baseline，遇阻碍自主解决。所有实现逐条对应原设计A→E；未验收事项保持未完成，六Gate仍在各自版本执行。

| 用户要求/阶段 | 实现位置 | 真实证据/剩余验收 |
|---|---|---|
| 统一服务器目录、规范和安全资源边界 | 科研开发规范首要9/10；server_migration.md；CURRENT_STATE.server | 已迁移并核验；只写/data1/yangbin/SCOR，最多4卡，不动他人 |
| 同步v01/后续文档 | 两份README、baseline_design、gate_protocol、design_review、pilot_report、baseline_report | 修正旧scheduler/状态/资源描述；Gate版本尚无实现，不提前称通过 |
| A：固定小模型/原生闭环 | prepare_model.py、configs/baseline_2b_candidate.json、run_baseline.py、native_adapter.py | 固定官方2B文件hash；cal01实际运行但健康拒绝；修订cal02四卡运行中 |
| B：完整状态/恢复 | checkpoint_store.py、native_adapter.py、archive_checkpoint.py | S0与非零Adam完整状态真实保存；缺分片/同size损坏/保护点反例；新引擎实际恢复和晚期fork待执行 |
| C：四split数据和记录 | prepare_baseline_data.py、records.py、native adapter/apply hooks | train1024/cal64/probe128/eval256 scene不重叠；真实角色、seed、图像/工具/token/mask记录；正式held-out未用于调参 |
| C：同unit重放/单reward干预 | replay.py、interventions.py | 错图拒绝及整组adv手算/native互证通过；实际GPU无修订等价和奖励配对更新待执行 |
| C：独立评测 | evaluate.py、probe.py、native_adapter.evaluate | Solver整数显式答案、fallback分列、case聚合；固定正确/错误Verifier诊断和teacher-forced分数；GPU RNG/游标隔离运行中 |
| D：正常baseline | run_baseline.py、baseline_acceptance.py | 六项证据同身份且全部通过才freeze；3个预定seed×32步与终态待执行 |
| E：Gate准备 | 完整S[7]/U[8]、false-positive证书、真实配对更新 | D之后准备；不在v01提前实施六Gate或四SCOR模块 |

2026-10-11最新服务器预检：约926.804GiB余盘，GPU0–7空闲，只用4–7。此前32GiB/75.87GiB是历史资源快照，不覆盖原记录。cal01的S0已有lossless本机归档且全部member SHA验证；S1/S2传输中断，完整服务器副本保留，未称本机备份完成。

候选修订在独立校准期进行：原生constant scheduler在warmup0时首轮LR0，改用原生cosine/fullT32；增加角色chat framing与tool后Solver提示；使用4轮/3072/context8192给完整Repair留容量。cal01实际截断0，不以截断严重作为修订理由。阈值采用设计第5节的工程建议：初始各题型正确率10%–90%、Solver/Verifier解析至少90%、截断至多5%、至少20%group非零adv、实际工具与自然Repair、参数/optimizer有限且真改变。阈值在cal02结果读取前确定，不能事后降格。

全部版本40项服务器测试通过（2026-10-11 CPU06），其中新baseline契约/原生17项。续接独立审查补小数/指数/空boxed解析、worker状态指纹缺失、空/失败/异配置freeze拒绝；当前新增源码尚需再同步服务器与回归。源码身份变更后重新开校准，不混旧状态。

按规范做设计→代码和代码→设计、参数生效值及独立反例审查。未闭环P0/P1阻止正式实验。此任务是CLI/GPU工程，产品界面冒烟不适用；实际CLI、原生链路与恢复必须执行，测试通过不能替代GPU验收。


2026-10-11独立审查修订：固定原生_get_template_tokens已经apply_chat_template，cal02额外包装导致重复user/assistant，已自主中止并保留S0、原始角色/工具日志和精确引擎源码。移除重复包装，直接复用原生模板；tool后Solver返回提示保留。新入口每run保存raw/source及profile；当前本机47passed/2native模块skipped，server修复回归和新cal03准备中。此前cal02运行中表述为历史进度，不作为验收。


最新（cal05）：2B initial64盒式答案0/64/Verifier39/128、更新1–3后step4 OOM，完整S0/S1保留；按授权换固定3B、原生compiled entropy、统一processor min3136/max65536、4轮/1536/context6144。CPU70passed/1GPU诊断skip、独立GPUentropy1passed；cal05真实四卡full32调度/stop1进行中。不得将模型更换或像素预算说成原方案预注册，全部在calibration确定、held-out未看。后续同状态恢复/重放和3seed32仍待验收。

cal05真实完整stop1结束但格式/Verifier拒绝，cal06完整S0/S1、initial64显式61/64和Verifier127/128，final评测尚在执行；headroom题型0/2超过.9。cal07本机已修三值求和与最大减其余两值之和，反向checker检查及问题措辞测试通过（local49passed/3native模块skip）。源引擎未覆盖运行中的server cal06；结束后同步、新dataset07和全server回归，再进行完整校准及恢复/重放。

cal06完整正常stop1/return0，health重建仅task_headroom失败，其余13规则通过。final64准确54/64、显式62/64、Verifier128/128；Tool13/自然Repair1、非零group4/4、训练/评测截断0、usedLR5e-7、grad2.471236、全参数4shards变化且有限；物理采样最大22479MiB。完整wall2086.555s/四卡2.3184GPU小时，native unit119.357s、保存1033.284s、评测722.863s，重叠项不相加。

cal07启动data07后，真实torchdata懒游标恢复probe发现state_dict会物化iterator并消耗driver随机流；原load先restore RNG再检查游标，存在P1。已只中止本run标签进程，raw/source/16265099208bytes部分状态保留，无完整证书/评测。首次中止脚本误用旧系统Python，在任何signal/写入前因Path API报错；改隔离3.11后实际中止，事实存abort收据。

修订先物化并核对cursor，再最后restore driver RNG；实际S0/S7 CPU loader的下一批0–3/28–31与下一次随机数通过。CPU18完整78passed/1GPUskip11.68s；重放实际counter仅允许num_steps/num_repairs的同值int32→int64原生转换，改值/浮点、mask/图像/slot仍拒绝。持久buffer指纹/NaN检查、driver/data实际hash、acceptance源身份加入；delta0与repeat-initial-evaluation仅consistency，用于设计第9节无操作对照。

cal08新engine新run于2026-10-11 07:55:06UTC启动，PID657013/资源657014，data07 signature23986c40e196e274457ed02376b82cf5ce3557c7a245e53e099cf8bf27bf32f2，fullT32/stop8/pin1,pin7，预检734.57GiB和GPU4–7空闲。正式未验收，实际S0/S7恢复/U1/U8等价、重复评测下一rollout、六证据冻结、3seed32/E仍待完成。

最新执行为cal09：data07不改、max_pixels32768、原生slow pixel一致性、视觉deterministic实传、完整environment身份及严格dtype/zero no-op，已通过81项server CPU和独立小GPU诊断；详见[baseline_report.md](baseline_report.md)。完整GPU恢复/U1/U8、六证据freeze、三seed32及E仍未完成，不能降级为工程演示。

cal09最新：initial64正确57/64、显式61/64、Verifier127/128、截断1/64，仍sum22/22和差值19/21超过headroom上界。真实完成更新1–3，S0/S1完整；step1grad2.441516/usedLR5e-7/参数探针真变有限。已于09:21:36UTC只中止本run，全部raw/source/fullS0S1保留，不能把未保存模型区间当可恢复state。

cal10候选数据独立scene为三位整数100–999，512×512图、字号24、bar高度floor(.28*value)保证柱顶/标签在画布内，三值和/三值乘积/最大减其它二值和公式不变。Legacy15 source未改；baseline自己的render/checker/收据一致，新immutable visual_baseline_08 signature8b38d437b78fe97caa74a47ba3584eebcd0c89e3b55d5ea9a09f1963d46cb8a1，正式held-out效果未读。统一slow processor预算min3136/max49152（196px/49vision tokens待实际probe）；Tool成功但No output时要求真实print(result)，不把裸Python表达式当打印结果；终局用无逗号的十进制整数，防大数格式失败。CPU21实际82passed/1skip14.15s，本机50passed/3skip0.69s。

服务器剩634.063GiB；累计正式/consistency/恢复资产预算必须重新核算，先为不再依赖的旧校准快照做lossless完整backup并逐member SHA验证，验证后才能退役server副本。本机可用289GiB，旧partial archives不能假称complete。所有正式冻结、GPU恢复/replay和三个32步正常baseline/E仍待完成。

cal10完整initial64为56/64（.875），显式61/64（.953125），Verifier117/128（.9140625），截断0。family0 19/22与family2 18/21满足规则，family1 19/21仍超过.9上界；已自主仅停本run，certifiedS0/raw/source与partialS1保留，没有已完成unit。cal11同source、同data08、同seed/full32/四卡，只max_pixels49152→32768，所有门槛保持。backup cal05 S1的全source SHA已通过，durable gzip正在生成，尚未本机完整member验证，也未退役源。

## 2026-10-12用户授权清理后的资产范围

本文件前述校准/controls/备份状态描述属于清理前的执行事实；实际原始证据与冻结JSON未改。用户明确清理后，仅保留三个正式S32、首seed S7/S8和E S8共六份完整训练状态；旧校准/对照/重复初始及中间状态大权重、失败半写、2B和旧压缩/本机备份与下载残片已退役，证书及small metadata保留。当前旧点不能完整恢复，cal06本机下载不再是待完成项。3B/环境/data08/所有raw与processed保留，结果与验收结论不变；详见[storage_cleanup.md](storage_cleanup.md)。新的校准/恢复复核若需要旧完整状态须另立新运行，不能用历史证书冒充现存模型。
