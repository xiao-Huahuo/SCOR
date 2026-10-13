# Gate1 实现与验收记录

2026-10-12开始记录，2026-10-13更新。正式冻结已验收，五组T32正在执行。本记录按时间保留工程进展，较早的待办状态以末尾最新收据为准；不把启动任务视为完成。

`run_gate1.py` 保留 v01 原生 AdamW/GRPO、3B、data08、32步 scheduler 和四卡拓扑，通过明确兼容契约读取旧完整状态及固定历史。执行身份由15份源码构成；旧身份单列。新恢复参数 `--recovery-from` 指向原 attempt，要求同 seed、分支、冻结、输入、源码、学习参数和最后完整提交点。已提交的 U8 保留在原 attempt，续跑从 U9 开始，禁止重复注入。失败及未完成文件保留，成功单位不补造完成标记。

`gate1_contract.py` 逐字段核验后续 token、图像、角色 mask、group、顺序、reward、advantage 和 reference；当前 actor anchor 单列。第8步只允许一个 terminal outcome 从0变1，使用上游实际 GRPO 函数重算整个 group，并验证其它 group 不变。`gate1_probes.py` 保存运行前固定的 boxed 正确/加一 Solver 候选与 Verifier 接受/拒绝候选，实际 padding 不进入评分。

`run_gate1_sequence.py` 固定五个种子，按 source、共同 S7、clean、corrupt、repeat 串行执行。无目标按原计划报告。它捕获 S0 的认证小元数据，保留0/8/16/24/32评测来源；元数据不能冒充可恢复检查点。完整的来源、单位、末态、恢复链和逐 case 评测由 `gate1_validation.py` 核验。六份旧完整状态受保护；新的临时状态须在真实验收和依赖闭合后退役。

`export_gate1.py` 从所有计划种子的不可改写原始证据重建主统计、逐 case、按 family 和 false-positive/格式/工具辅助诊断。主统计单位是独立训练 seed，保留负 floor 校正值。`export_gate1_costs.py` 单列工程控制、失败/重建、新实验与复用 v01 的既有费用。`plot_gate1.py` 验证五个曲线点、末态与主均值一致，导出 PNG/SVG 及数据、源码 hash；均值和区间采用同一 Q_s 量纲。

已完成的实际证据：

- 646文件跨版本导入及来源 SHA/typed units 验证。
- boxed 校准64题、正式探针128题及各自 Verifier 正误上下文的固定输入构建。
- boxed 工程控制 `gate1_control_s7_stop8_20261012_02`：实际全状态 S7 恢复、固定 U8 更新、完整 S8/完成标记、正常 return0；更新后四 rank 参数、Adam、scheduler、worker RNG 与旧 native clean S8 相同。
- 同 S7 两次隔离评分的 D_score=0。这是该控制的实测数值观察，不是噪声95%分位数。
- 审查后源码的服务器原生检查107项通过（36.06秒）；原始日志与源码 SHA 在 `native_cpu_reviewed_20261012_03` 收据中。最终工作流修订随后在隔离目录完成117项原生回归（32.81秒）；第一次隔离目录漏放参考源码的8项读取失败日志保留，补齐12份字节一致参考后重跑通过。
- 实际两个自有 CPU 心跳进程验证了按 UID、环境标记、start time 绑定 PIDfd 的停止/终止边界。服务器 Python/libc 没有 pidfd_open 包装，使用本版 `gate1_pidfd.c` 从系统头文件调用内核接口；未改 GPU Python 包环境。这不替代 GPU 中断恢复证明。
- 合成数据的图表排版与两个拒绝反例，以及成本账本的四个反例通过；合成图不属于科研结果。

审查后源码的 clean 控制 `gate1_control_clean8_20261012_03` 已正常停止于8、完整提交S8，U8后四rank训练状态与旧clean S8相同。队列03随后在CPU验收阶段因父进程缺少实际上游 `Agent0-VL` 导入路径失败；后续污染控制未启动，原失败日志与收据保留。队列04只修复父进程导入路径，先完整复核并复用已有clean控制，45份原始证据通过；随后以新04 attempt IDs执行真实不完整U9中断、恢复与连续对照及S0/P256。它仍只用物理4–7，服务器源码在队列期间锁定。正式CLI的部署导入路径正在追加独立审查；实际GPU恢复比较、存储预算、八项冻结证据、五组T32和科研结果报告仍待完成。

预算审核曾指出两个问题，现已在代码与反例中闭合：旧短候选工程评分没有新版 candidate mask 字段，须依据原固定 tensor 进行明确的 legacy engineering 验收；prefix 到受保护 S7 时，旧 keep1 策略还会留下 S0，须在完整前缀验收后记录其非保护初始权重退役，或增加预算。旧短候选及boxed两个真实控制已分别完整验收通过；prefix0退役整链经实际native小状态测试通过。真实GPU新控制结束后的权重退役与当时存储预算仍须验收，未提前删除。

独立审查记录见[gate1_workflow_review.md](gate1_workflow_review.md)。当前八项正式证据中，独立代码审查和clean阈值校准已有真实收据；完整GPU恢复、初始控制和累计预算仍待闭合。

部署反例追加：独立审查在无harness PYTHONPATH的真实GPU Python3.11.15复现了正式sequence/export父进程缺`verl`。本机只改sequence的三行路径设置；隔离fresh正例读取真实16row U8及native单位验收通过，export共用同一设置，93项本机回归通过。live源码在queue04结束前不改；正式冻结必须记录新sequence SHA及新的独立审查witness。此前03审查收据原样保留，不能用于声称新候选已经部署。

15:29UTC实际进度：三项新验收收据已落盘并通过，分别是645导入成员加manifest共646文件/96个原生单位、实际全S7恢复、固定clean U8与原clean S8训练状态等价。第一次证据脚本误把646当成员数，在守卫处失败且没有生成passed收据；修正口径并绑定原import SHA后以新02尝试完成，旧失败不覆盖。新sequence隔离候选的完整服务器原生117项回归通过（8.42秒），live部署仍等当前队列结束。

污染工程控制于15:25:59.216754UTC真实SIGKILL（return−9）：全S8和U8完成标记已提交，真实U9奖励已写出，U9完成标记不存在；按UID、运行环境标记及start time绑定的91个自有进程被终止。唯一完整恢复点S8及失败现场保留。新的recovered9 attempt已启动；恢复与连续运行的完整状态/评分比较尚未完成。1850份小型收据已同步本机并逐SHA核验，模型和大型trajectory tensors留在服务器。

2026-10-13 Asia/Shanghai（18:12UTC）实际恢复闭环：continuous控制正常return0且两单位严格验收通过；`engineering_reviewed_20261012_04.restore_comparison.json`完整状态比较passed，四rank全部参数/buffer/Adam/group/scheduler/workerRNG及driverRNG、data cursor、位置逐项相等，两role的固定score差及D_score均0（64题calibration）。它验证工程恢复机制，不能当作正式128题探针或T32科研结论。原中断完整S8及partialU9保持，后续初始S0/probe128/P256控制已接续启动；正式预算、最终源码部署与八项冻结仍待闭合。

02:36（Asia/Shanghai）证据物化：实际恢复witness已通过并绑定139份去重证据，双方两单位validation原共140条逐SHA重核，原failed S8/partial9保留。新增只读审查发现早期历史reference entry缺run_id，当前接口真实KeyError；从原证书和四份core metadata生成新输入reference_metadata_20261013_01，9项validate_reference整链通过，旧输入不改。另将Verifier验收从标签覆盖补为签名parquet、实际raw prompt token与actor上下文的GT/GT+1逐行核验。初始控制fullS0/S1已出现，S1评分/评测/完成标记和终态验收仍待完成；正常临时权重尚未退役，预算/正式冻结/五组结果未完成。

03:09（Asia/Shanghai）实际工程队列闭合：initial控制正常return0，S1 score/evaluation/completion真实提交，controller validation passed、一个原生单位和44份证据。队列19:06:22.728693UTC完整结束，GPU4–7各11MiB/0%使用。主代理接续部署已独立核准f97及完整S0物理内容核验，随后固化probe_inputs/newcodeReview、正常临时状态退役和累计预算；这些操作及正式五组T32仍未以启动代替通过。

03:45（Asia/Shanghai）最终输入与代码审查已固化：新probe_inputs通过63份SHA证据，包含完整S0物理核验、9份非model/optim metadata capture、实际128题/768候选、P256及S1终态；新codeReview通过168份证据，部署35份源码与native07完全一致，117原生检查/5个重签语义反例/实际生产导入来源均核准。七项witness已完成。仅六成功工程控制的大分片退役正在CPU串行进行，首个恢复9控制已完成，连续9控制接续；原failed完整S8及六旧完整状态不在退役集合。

预算已独立按真实checkpoint plan/keep1/protected及sequence次序重建：新增峰14、最终11、453新updates和89评测点，raw预测36.40625GiB并留48GiB总余量；这是既定成功路径，新增正式失败/恢复祖先会额外占用，必须重新检查累计资源后才resume。最终storagewitness固定已审计算器和退役操作SHA，并实际核验六退役收据/大分片消失/小metadata留存和七保护状态全集，再测free。退役尚未全完成，实际budget/八项冻结/正式五组与科研结论仍待完成。1434份新增small记录66,884,790bytes已逐SHA同步本机，无大模型/optimizer/generation/training tensors下载。

03:56（Asia/Shanghai）退役保护门实际发现证据遗漏：恢复9、连续9两个正常控制已完成大分片退役，initial控制在helper检查覆盖范围时拒绝。实际唯一缺项为native checkpoint0收据（capture存在后validate_run要求追加），初始S0/S1大分片完整且没有退役plan。原日志/源码/旧proof及失败原因保留；新runtime02候选补齐此收据，并验证复用已完成两项、为待办使用新proof/status/launch IDs；独立审查尚在进行。ENGINE15、六workflow及native参数不改；预算/正式冻结仍等待完整闭环。

04:13（Asia/Shanghai）独立审查已闭合操作P1：真实旧proof与当前validator差集只有checkpoint0，补齐后覆盖完整需求；正式source/prefix的真实调用顺序capture→terminal validation→accepted，paired proof保留完整branch evidence，未发现同遗漏。新的controller02/storage02已逐SHA部署、controller02以独立PID2738221及新日志/运行收据启动，核验复用已完成两项并继续四待办；原01失败现场及初始完整权重保留。仅操作层修补，不修改ENGINE15/workflow6、旧proof或原始实验记录。完整退役、actual预算和冻结仍待执行验收。

04:47（Asia/Shanghai）六个成功控制退役已实际全部完成，七份保护状态及全small/raw留存；最终strict storage操作在物理/来源校验后测得free约630.614GiB，低于652.589GiB计划，实际failed预算收据和日志保留，未生成storage通过收据或正式冻结。正在依据项目实际空间与全部raw/eval/log/cache尺寸重新核查未冻结AI资源提案，不能删保护状态或降低五组T32及评测要求。54份既有evaluation JSON最大647,677bytes，原每点评测256MiB是粗估；具体资源修订仍须完整独立审核及新实际测量。

2026-10-13：因原累计空间实际不足，工作流保持clean→corrupt→repeat并将clean/source完整末态独立验收及去重提前。纯early proof不包含后续退役收据，paired/export重建时严格核对early与clean complete，最终只退役repeat。成功终态清理中断从原proof/plan续做；clean续做前再次认证保留source S32，plan/complete须完全相等并绑定本sequence/seed/commonS7。ENGINE15未变；native09全部133检查通过，新增16个生命周期/guard病例明确是小状态夹具，不是正式科研结果。原native08夹具导入别名失败收据保留。新累计13槽预算保留48GiB及原raw/eval额度并另核验各原launcher guard，实际budget/新codeReview/正式freeze仍待验收。

2026-10-13最新验收：新05独立代码审查（611a596d）与存储证据（baa5a621）已实际通过，八项证据齐备；预算实际free677054451712bytes高于严格累计需求666741017387bytes。CPU正在完整重核所有唯一证据SHA（包括探针及24大分片），随后写入新冻结文件并进行独立实际复核；正式五组T32尚未启动。冻结预审还闭合了历史P0混配漏洞：三组映射严格绑定原草案27d811，引用项和导入SHA同源，重签反例拒绝。早期草案由本机原字节补齐部署，不伪装成既有服务器freeze或用户预注册。

2026-10-13正式启动：冻结文件b36195b5129d8fe33fb26d808434bfd7db8e9b8e0de21f69c139af19fd164d3a经实际1021份唯一证据全SHA与validator通过，独立实际复核无P0/P1（独审另核995small，26large引用主代理全核验）。fresh GPU4–7/累计空间预检通过。五种子T32序列gate1_20261013_01的父进程2886355于2026-10-12T22:32:30.807780UTC启动，先完整核验导入资产，再串行GPU分支；启动不代表T32验收或科研结论。源码15+6锁定，失败保留现场并重新核查资源后才恢复。

2026-10-13首个正式U8验收：clean seed20261011的实际完整S8已认证，原生U8、固定probe128/P256与四rank全状态/driver RNG/cursor比较通过；根代理完整哈希审计收据为e682c641，独审另核小证据及fresh原生单位，未重复大权重哈希。第8–15步原始完成记录已提交，第16步评测在进行，最新完整恢复边界仍为S8。为保证滚动删除S8后的审计可追溯，cert及全部非model/optim共9个小成员（167667bytes）已保存到raw/audit_S8/global_step_8，追加810e306b映射收据而不改原审计；该目录仅元数据，不可直接恢复。五组T32、配对Q_s/95%seedCI、成本与图表报告尚未验收。

2026-10-13首条clean T32已完整验收：seed20261011的25个原生更新与7/8/16/24/32固定probe128/P256均实际通过，完整S32提交、正常return0，stage accepted ccd178bb。序列实际认证完整保留source S32后逐字段比较，worker参数/buffer/Adam/groups/scheduler/RNG、driver RNG/cursor/position/step完全相等，early clean_source_dedup证明bb6010a8已固化。退役plan/complete字节一致（ff12ff5d），仅移除重复clean的8个model/optim分片，46,302,164,640bytes；其余小成员、证书、全部raw与指纹保留，该退役点不可直接恢复。独审实际核准25单位/5点评测/退役生命周期，去重363份证据中362份small重核SHA，910MB固定输入及大权重认证明确沿用实际启动器/序列/helper运行。旧六完整状态、共同S7及原failed工程S8保持。首corrupt已从共同S7实际完整恢复；U8注入、repeat、paired五组与科研结论尚未验收。
