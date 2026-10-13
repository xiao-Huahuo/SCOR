# v02 / Gate1

检验一次终局 false-positive 在完全相同的后续历史数据下是否持续到32步。[实施协议](../../docs/v02/gate1_protocol.md) 固定五组 seed、主指标、对照和停止规则。

正式冻结与工程验收已完成，五组T32正在执行；首个完整paired block已验收（1/5），五seed主统计及科研判定尚未完成。`src/` 包含 v01 原生链路的版本内副本和 Gate1 审计/测量入口；旧版本保持原貌。`inputs/` 保存经核验的跨版本资产与固定探针，`outputs/<run_id>/` 保存 immutable raw、完整 checkpoints 和可重建 processed。下方进展按时间保留，以末尾最新收据为准。

当前状态：最终native09的133项检查、真实GPU中断恢复与连续S9状态/评分等价、初始probe128/P256、生产CLI及八项冻结证据均已验收。源码15+6冻结，首seed三分支T32与实际退役独审闭合；seed20261012前缀重建、S0退役及clean T32/source等价/实际去重已核准，corrupt分支已启动。详见[实施与验收记录](../../docs/v02/implementation.md)。

入口为 `src/run_gate1.py`（单attempt）和 `src/run_gate1_sequence.py`（五组序列）；`export_gate1.py` 重建统计与辅助CSV，`export_gate1_costs.py` 重建全量费用，`plot_gate1.py` 使用独立绘图环境导出PNG/SVG。所有科研结果必须来自完整原始证据；当前合成排版测试不是正式结果。

服务器Python环境不全局安装patched `verl`。单独运行 `import_gate1_assets.py` 或 `gate1_probes.py` 的准备入口，须从项目根目录显式设置 `PYTHONPATH=solutions/v02/src:.runtime/upstream/Agent0_baseline_264348cb5893/Agent0-VL`；本次固定补丁身份来源于 `apply_baseline_instrumentation.py` 的SHA前12位。正式sequence/export候选入口已在父进程设置这一精确路径，实际收据见[独立审查](../../docs/v02/gate1_workflow_review.md)。

2026-10-13：因原累计空间实际不足，工作流保持clean→corrupt→repeat并将clean/source完整末态独立验收及去重提前。纯early proof不包含后续退役收据，paired/export重建时严格核对early与clean complete，最终只退役repeat。成功终态清理中断从原proof/plan续做；clean续做前再次认证保留source S32，plan/complete须完全相等并绑定本sequence/seed/commonS7。ENGINE15未变；native09全部133检查通过，新增16个生命周期/guard病例明确是小状态夹具，不是正式科研结果。原native08夹具导入别名失败收据保留。新累计13槽预算保留48GiB及原raw/eval额度并另核验各原launcher guard，实际budget/新codeReview/正式freeze仍待验收。

2026-10-13最新验收：新05独立代码审查（611a596d）与存储证据（baa5a621）已实际通过，八项证据齐备；预算实际free677054451712bytes高于严格累计需求666741017387bytes。CPU正在完整重核所有唯一证据SHA（包括探针及24大分片），随后写入新冻结文件并进行独立实际复核；正式五组T32尚未启动。冻结预审还闭合了历史P0混配漏洞：三组映射严格绑定原草案27d811，引用项和导入SHA同源，重签反例拒绝。早期草案由本机原字节补齐部署，不伪装成既有服务器freeze或用户预注册。

2026-10-13正式启动：冻结文件b36195b5129d8fe33fb26d808434bfd7db8e9b8e0de21f69c139af19fd164d3a经实际1021份唯一证据全SHA与validator通过，独立实际复核无P0/P1（独审另核995small，26large引用主代理全核验）。fresh GPU4–7/累计空间预检通过。五种子T32序列gate1_20261013_01的父进程2886355于2026-10-12T22:32:30.807780UTC启动，先完整核验导入资产，再串行GPU分支；启动不代表T32验收或科研结论。源码15+6锁定，失败保留现场并重新核查资源后才恢复。

2026-10-13首个正式U8验收：clean seed20261011的实际完整S8已认证，原生U8、固定probe128/P256与四rank全状态/driver RNG/cursor比较通过；根代理完整哈希审计收据为e682c641，独审另核小证据及fresh原生单位，未重复大权重哈希。第8–15步原始完成记录已提交，第16步评测在进行，最新完整恢复边界仍为S8。为保证滚动删除S8后的审计可追溯，cert及全部非model/optim共9个小成员（167667bytes）已保存到raw/audit_S8/global_step_8，追加810e306b映射收据而不改原审计；该目录仅元数据，不可直接恢复。五组T32、配对Q_s/95%seedCI、成本与图表报告尚未验收。

2026-10-13首条clean T32已完整验收：seed20261011的25个原生更新与7/8/16/24/32固定probe128/P256均实际通过，完整S32提交、正常return0，stage accepted ccd178bb。序列实际认证完整保留source S32后逐字段比较，worker参数/buffer/Adam/groups/scheduler/RNG、driver RNG/cursor/position/step完全相等，early clean_source_dedup证明bb6010a8已固化。退役plan/complete字节一致（ff12ff5d），仅移除重复clean的8个model/optim分片，46,302,164,640bytes；其余小成员、证书、全部raw与指纹保留，该退役点不可直接恢复。独审实际核准25单位/5点评测/退役生命周期，去重363份证据中362份small重核SHA，910MB固定输入及大权重认证明确沿用实际启动器/序列/helper运行。旧六完整状态、共同S7及原failed工程S8保持。首corrupt已从共同S7实际完整恢复；U8注入、repeat、paired五组与科研结论尚未验收。

2026-10-13首条corrupt T32已完整验收：seed20261011的25个原生更新及7/8/16/24/32五点固定probe128/P256均实际通过，完整S32提交、正常return0，stage accepted 237a6bf4。只有第8步原row5/cert08553的终局outcome从0改为1；9–32步无再次注入，固定输入和原生GRPO规则保持。首次U8完整状态与原保留E8实际全哈希认证并逐状态比较一致（125f1d1a），其9小成员持久保存使滚删S8后的审计仍可核验；元数据不可直接恢复。独审核准整条25单位/5点评测、319小证据及完整S32身份，模型物理认证明确引用已完成的启动器/序列。完整corrupt S32长期保留。226小JSON共11,644,239bytes已同步本机且SHA通过。首clean-repeat已启动；其floor、首个paired组及全部五seed主统计仍待验收。

2026-10-13首个完整paired block已验收（1/5）：seed20261011的clean/corrupt/repeat共75个真实原生更新、15个分支评测点正常完成，paired proof288dc35e已生成并独审。独立重算逐token→候选均方→case/role RMS→双role平均，四个非零D与实现最大差1.39×10⁻¹⁷；T32 D_score=0.05543707034 nat/token，floor=0，扣冻结ε_D=.01后的单seed Q=0.04543707034。该值仅描述一个训练种子，不给seed CI或五seed科研判定。S0曲线来自一份真实common测量的复用。repeat plan/complete70497951同字节且绑定原paired，仅退役自身8个model/optim共46,302,164,640bytes；9小成员及全部raw/指纹留存，退役点不可直接恢复。原clean early/退役、旧六完整状态、原failed工程S8、source/commonS7和完整corrupt32保持。752份paired小证据与362份退役小证据经独审SHA核准，大认证引用实际运行。序列已接续seed20261012前缀重建；其余四块、真实五seed统计/图表/成本报告与最终验收未完成。

2026-10-13第二个固定种子前缀已验收：seed20261012从共享base按原历史重放U1–7，完整S7提交并正常stop7/return0，accepted0a80371b、历史S7证明7c37b5b8通过。新S0/S7的四rank参数/buffer/Adam/groups/scheduler/RNG及driver/cursor/position与历史认证小元数据逐字段相等。prefix工程评测是cal64、Verifier diag128与固定probe128/768；正式P0另绑定冻结旧source12真实P256，不用cal64替代。自身S0的canonical退役plan/complete同字节96b39d62，142条证据覆盖，9小成员保留但不可直接恢复；完整newS7保留。独审fresh7原生单元及128份去重小证据SHA通过，本范围无未闭P0/P1，大权重认证引用实际wrapper/parent。98小文件3,646,372bytes已本机SHA同步。clean12已启动并实际恢复共同fullS7，四rank状态指纹、driver RNG和cursor一致（d35c1bce）；其T32、第二paired及剩余固定块仍未完成。当前完整paired进度仍为1/5，五seed统计、图表、成本和科研结论待验收。

2026-10-13第二个种子clean已完成实际U8及S8：起点和第8步的128/768固定探针、P256与RNG/cursor保护验收通过（1d305a43/331a67f1）；完整S8正常提交，checkpoint46,302,326,608bytes，保存666.9228秒。U8无注入，12固定张量与19元数据匹配；9认证小成员保存至raw/audit_S8/global_step_8（marker8ea1ea62），实际验证为不可直接恢复，9262dd3a的23条证据使用持久路径。独审fresh单位及39份去重small SHA通过，无本范围P0/P1；大物理认证引用实际native launcher，不声称旧source8完整状态相等。U9–15的7单位/77证据经根复核通过，第16步评测正在进行。81小文件4,052,244bytes已本机逐SHA同步。T32、第二paired及全部五seed结果仍待验收。

后处理边界另有真实工程反例：五计划种子仅一有效时，原绘图入口因缺ε字段失败。原plot与全部冻结身份保留；隔离候选8aae1b3e代数恢复既有CSV阈值，独审16 CLI及n1/n5视觉QA通过，仅合成工程证据。待正式训练结束后新增独立plot_gate1_reviewed.py入口并记录后处理SHA，原35和codeReview05嵌套证据继续可复核。最终真实科研图尚待完整导出及QA。

2026-10-14第二个种子clean T32已完整验收：seed20261012的25个无注入原生更新与7/8/16/24/32五点128/768/P256保护通过，正常return0，accepted c8c3e7e3。实际认证保留source12完整S32后核对全部worker/driver/cursor/position，末态与历史完整指纹精确相等，early eb0fa7d4保存。退役plan/complete同字节bc927710，仅移除自身8model/optim共46,302,164,640bytes，9small/cert与全部raw保留；该clean32不可直接恢复，共同fullS7及原source32仍可用。独审核准311/322/362份去重小证据，大认证引用实际native/parent。S8滚删后9262的23持久证据及8ea1的9成员/mapping仍有效，本范围无未闭P0/P1。232小文件10,997,953bytes已本机逐SHA同步。corrupt12曾因外部GPU4/5占用等待，原定4–7空闲后自动启动；同完整S7恢复已核准（a2eadbc2），唯一U8注入和T32仍待验收。完整paired进度仍1/5，五seed科研主统计与报告待完成。

2026-10-14第二种子corrupt U8已验收：同完整newS7实际恢复（a2eadbc2），起点128/768/P256及两role固定评分距离0（fb578b3c）。实际唯一first target row7/certf5fb，显式答案1356而GT−158，原终局0且未截断；只在terminal位置567增加1，native_info/native_returns保留。独立样本std GRPO oracle最大误差0，其它三组12行不变，7dbb输入审及25去重small SHA通过。完整S8正常提交（ae3e83cf/state2f8c1acf），46,302,326,608bytes、保存696.4625秒。第8步128/768/P256/RNGcursor通过1d8c49fa。9小成员167,797bytes保存至raw/audit_S8/global_step_8，marker c087fecf与23条持久证据1414fd78已独审；加probe8的28条证据中27small SHA核准，普通full restore实际拒绝metadata-only目录。大认证引用actualnative，不二次模型/cache哈希。U9以后固定历史继续，无本范围P0/P1；actual S16滚退后的持久性、corrupt T32/repeat12/第二paired及五seed统计仍待验收。65小文件4,017,416bytes已本机逐SHA同步。
