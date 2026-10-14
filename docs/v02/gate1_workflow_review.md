# Gate1 工作流独立审查与验收记录

日期：2026-10-12。范围：v02 工作流、恢复、验收、来源、导出与临时状态生命周期。明确保留 v01 冻结源码及六份完整状态。本记录不宣称正式五组实验、GPU恢复或科研结论已经通过。

## 逐项实现与验收对应

| 请求/协议 | 实现与可核验结果 |
|---|---|
| 五组20261011–15、T32/k8，不按后续损害选目标，无eligible不补seed/换k | gate1_evidence固定SEEDS及冻结检查；eligibility只读U8；全部五组源/eligibility收据；导出拒绝缺组、重复组、错误source seed。 |
| 同一S7/U8、固定U9:32/native reanchor | 三分支origin parent/history/learning相同；25个真实训练组的schema、reward、原生whole-group GRPO、generation、tensor/metadata、optimizer/finite、completion、评测/RNG逐项重核。 |
| 完成stage复用，恢复不混配置，不重算已提交边界，partial不能冒充完成 | 新stage plan/accepted证据收据；--resume校验后复用；新_aNNN attempt；--recovery-from从前attempt最后完整且model_snapshot_committed=true的点之后继续。完整状态之后尚未保存的段可重放，不能把这些raw update假称模型已提交。 |
| corrupt S8以后不能重复U8或改变目标 | 对前attempt seed/phase/branch/freeze/probe/import/learning/source/horizon/最后完整点严格一致；恢复protocol记录原intervention，future段没有再次injection。 |
| fresh source/preparation也遵守freeze | 传入--freeze就完整核验；每个source/prefix都传冻结；新source显式preparation/evaluation，取得真正256题初始P。 |
| rebuilt S7及clean/source S32实际核对 | 历史metadata import/checksum；所有worker参数/buffer/Adam/group/scheduler/RNG、driver RNG/cursor/position核验；source S32仍须完整；clean末态不一致立即停并留weights。 |
| 0/8/16/24/32辅助曲线 | 新source/prefix已认证S0的全部非model/optim小成员立即保存raw，含worker extra-state/RNG、小configs，明确metadata-only。共同S0实际log-score经score_distance计算，注明fork前共同起点，不伪称三次独立测量；P0读取实际256题案例。 |
| prefix stop7留下S0不能漏入预算 | 只有prefix terminal0、完整protected S7、S0与S7历史指纹均通过、小metadata/RNG已保存后，原子记录plan，复用canonical retire(newest7,keep1,protected=[])删除唯一新S0；不改canonical helper/S7；complete可恢复。 |
| 临时clean/repeat/control严格退役 | owner、terminal0、accepted IDs、明确空pending依赖、全部SHA、末态及真实anchor验证；仅本运行model/optim；plan→unlink/fsync→complete，半删除可续做。control S0从完整base+seed重建；S7派生从真实ancestry+history重建。formal corrupt与interrupted/failed不退役。 |
| 旧uniform候选工程兼容不放宽正式schema | 仅old phasecontrol/branchcontrol/freeze=None、原源码SHA一致、原Final answer0/1缓存；实际fixed_inputs.pt重核token/metadata hash、全一response/attention masks、无padding、每行同candidate tokens及logp长度；不补写旧rawmask、不作正式评分。 |
| raw重建导出，退休证书不能冒充restore | 重建paired proof、source stage accepted SHA、5组及case/scores/metrics/completion；退休后仅验证metadata和明确receipt；默认restore仍逐文件检查全部model/optim/extra-state。 |
| 配对训练seed tCI扣floor/epsilon，不截负值 | CSV同时有corrected_D及Q_s；primary.mean/lower/upper已为Q_s=D−floor−epsilon；decision与Q区间0边界一致。 |
| 原始辅助诊断逐seed/step/branch导出 | auxiliary_diagnostics.csv：独立fixed wrong-context Verifier重复false-positive率、correct acceptance/parse、Solver explicit/truncation与tool/repair计数；heldout_families.csv逐family准确率；不改主检验。 |

## 缺陷闭环

- P0：原paired proof直接passed=True，未充分绑定25unit、source、finite、评测身份及原生GRPO。现重建原始DataProto/状态并SHA绑定全部证据，invalid拒绝proof/retirement。
- P1：corrupt从S8/S16/S24无合法CLI组合。现显式recovery、继承origin intervention、future不重复注入，source/learning SHA一字不忽略。
- P1：sequence无重启复用与恢复。已用真实小CPU子进程验证failed→S8恢复→U9起续→再次复用不启动进程；旧S8 SHA与partial9保留。
- P1：preparation绕过freeze。现冻结传入即核验。
- P1：metadata引用混淆restorable checkpoint；毒path/祖先symlink/错误seed/缺large shards。现metadata-only schema与full restore分开；默认full仍完整验证必需状态。
- P1：临时早删、S0假称由S7重建、半删除无法继续。现base/S7 ancestry明确，owner/evidence/dependency/terminal严格，原子收据可续做。
- P1：stop7/pin7的keep1会留下S0，漏两份initial权重。现prefix0专用验收并调用原canonical退役。
- P1：导出未覆盖全部source/metrics/scores/heldout身份。现重建并拒绝篡改，source S32必须完整。
- P2：epsilon bool误作float容差。现精确type拒绝bool。
- P2：Q_s与corrected_D区间坐标混用。现primary/endpoint明确Q_s；主代理绘图使用对应坐标。

## 实际验证和限制

- 本机最后全套：.runtime/venvs/scor-local/bin/python -m pytest -q solutions/v02/src/tests → 93 passed、2 native模块skipped、3.29s；compileall及3个CLI help通过。
- 新pure测试70项（原23），含路径/证书/rank/finite、恢复继承/提交边界/实际进程中断、freeze/source/五组导出、证据改动、mask/padding/heldout GT/RNG、base S0、原子退役中断、prefix0仅删自身并保S7、auxiliary原始比率。
- 新native模块14项：9项实际DataProto/GRPO异常；1项完整materialize→validate_reference→metadata_fingerprint→compare_rebuilt→capture→prefix0退役→重复resume调用链（只隔离该测试不涉及的训练unit审核，不mock metadata指纹/比较/删除）；4项旧uniform缓存真实tensor兼容/拒绝。
- 主代理服务器旧同步版本实际107 passed/36.06s，receipt为.runtime/gate1_jobs/native_cpu_reviewed_20261012_03.{json,log}。此后新增legacy mask、prefix0、全部small成员capture及测试仍须server回归，不能把107冒称最终helper已全通过。
- 本代理未启动GPU、同步server、删除真实模型、commit/push或修改README/CURRENT_STATE/CHANGE_HISTORY；状态由主代理统一记录。
- 待主代理完成：最终helper serverCPU回归、新15实际GPU恢复/连续等价与S0/P256控制、输入/历史metadata import、真实预算/freeze、五组formal、图表与科研结论。

## 冻结与调用接口

ENGINE15已锁，唯一由本代理改动的learning source是run_gate1.py，SHA c0c79a795201aff878a56bbab51a1ee16341149d68e94e19e3f7c0ce4fb9e85b。
仅新增启动冻结/显式恢复/辅助evaluation split，其他14未修改。身份仍包含新SHA；旧boxed控制保留旧execution身份，可当标明旧source的独立数值校准，不改成当前learning身份。当前需新15 S7/U8、真实corrupt中断9/恢复9/连续9全state/RNG/cursor/输入相等、新S0/P256。已中断工程S8完整保留；按实测峰值预算，不扩大failed退役例外。

workflow_source_sha256精确6文件：gate1_evidence.py、gate1_state.py、gate1_validation.py、gate1_references.py、run_gate1_sequence.py、export_gate1.py。

初始字段：
- initial_state_control={run,scores:{path,sha256},evaluation:{path,sha256},snapshot}：新15 seed11实际S0、formal固定probe128及P256；snapshot使用完整状态或明确合法退役小metadata。
- initial_evaluations={"20261011":{path,sha256},"20261012":{path,sha256},"20261013":{path,sha256}}：old source真实256 eval0物化v02 inputs；可附source_manifest:{path,sha256}与import_manifest:{path,sha256}，均验SHA及source seed/data身份。
- control退役proof：passed=true、accepted_run_ids、pending_dependents=[]、覆盖validate_run(run).evidence的所有path/SHA。依赖子控制先验收/退役，再退役其已完成父控制；interrupted父S8保持完整。

## 追加：真实部署导入路径审查

2026-10-12独立审查在服务器GPU Python3.11.15、删除PYTHONPATH、空CUDA、fresh进程下复现：原正式sequence父进程在U8 eligibility以及训练后的native单位验收会缺`verl`，结果export也会沿同一调用链失败。训练子进程的PYTHONPATH不会传回父进程，测试harness路径不能代替生产入口验证。此P1已在本机候选sequence增加三行精确patched checkout导入路径；export通过原有sequence import共用它，ENGINE15、pip环境和native参数不变。

实际隔离收据为 `.runtime/gate1_jobs/cli_native_import_20261012_01/receipt.json`：旧入口真实失败，修复版载入patched264348包、16row真实U8，native unit验收与export共用函数通过；两log和源码SHA均经独立只读复核。本机93项通过、2个native模块跳过（3.39秒）。修复候选SHA为`f97edc43c34b4b1cdca7bdef49acad42b56477b3c5e296ba0ea4a62ec2ad7cfa`。当前工程queue04仍使用锁定旧workflow；必须等队列结束后同步，并以新SHA和新审查witness正式冻结，不能改写旧03收据的事实。

独立实际检查确认成本CLI只读取JSON，小state的driver_rng/data也是可直接读的dict，两者不受此缺包影响。已有importer和probe缓存已成功构建；其裸CLI仍须显式设置本次固定patched包路径，例如从项目根目录使用 `PYTHONPATH=solutions/v02/src:.runtime/upstream/Agent0_baseline_264348cb5893/Agent0-VL`。不安装新的pip包，不将这一操作配置解释为学习参数变更。

## 追加：实际恢复与输入层独立核准（2026-10-13）

独立只读审查已核准新源码seed11 S7→9工程恢复：双方validation各两个实际原生单位，合计140条证据SHA匹配；真实完整S9比较passed，四rank参数/buffer/Adam/group/scheduler/workerRNG、driverRNG、cursor和位置相等，cal64两角色及D_score均0。mandatory recovery witness随后物化为139份去重证据，原interrupted/−9、partial9、无原cert/completed9及完整failed S8保留。PID记录没有逐PID另存UID/start_ticks为明确P2，实际锁定PIDfd代码与双marker控制经过核验，不伪造追溯字段。

审查真实发现旧reference_metadata_20261012缺run_id，当前validate_reference会KeyError。已从原始证书/四core metadata构建新的reference_metadata_20261013_01，全部9项实际核验通过，旧目录保留。正式调用必须使用新ID，元数据仍不代表旧权重可恢复。

Verifier的标签覆盖不足以证实其实际context，已在运行时验收逐行补核签名parquet、base绑定、GT/GT+1上下文、raw_prompt_ids及实际actor prompt文本、token/rightpad masks。独立fresh原生CPU审查实际重跑128base/768候选记录及P256/GT/image/family/scene/aggregate/RNG/cursor核验，通过5个先重签payload/manifest的语义反例：错误base绑定、正确标签下错误raw context、正确rawIDs下错误actor context、rightpad含非pad token、padding被计入candidate mask。实际审查代码/log/JSON为.runtime/gate1_jobs/initial_inputs_independent_20261013_01，JSON SHA96d92aca524da15f2d191d07659082f62e68836332fece6d3fe43aa592de60c5。该审查没有GPU任务、模型hash重跑或改动真实输入。

最新sequence候选完整native CPU回归117项通过（8.42秒），证据native_cpu_final_workflow_20261012_07；fresh生产导入反例/正例另有CLI01。live old sequence仍在queue04期间锁定，新f97仅在隔离候选验证；终态后实际部署必须逐SHA匹配测试候选，再生成引用本记录不可改写快照的新codeReview04。完整probe_inputs需要初始控制终态/S1评测和主代理新增S0物理全成员SHA核验；累计预算、正式冻结及五组T32科研结果仍有独立验收项。

实际工程终态追加：19:06:22.728693UTC queue04全部completed。初始control return0、fullS0/S1及score/eval/completed1齐全，controller原生单位和44份证据验收passed。主代理接续部署与额外S0全成员hash；本审查的输入层收据仍只按其18:44观察scope使用。恢复、initial、codeReview和六成功control退役三个运行时物化/生命周期脚本经独立只读审查无新增P0/P1；源15/6不作额外算法修改。

## 2026-10-13：提前clean验收、恢复入口与完整容量审查

仅sequence改为 `44802f39d163f978acfaab2d1689f7a52697102b90b72f0820605643d32adf29`；ENGINE15、state/validation/reference/export实现不变。保持clean→corrupt→repeat。完整clean25单位/评测/S32与保留source S32实际等价后保存独立early `clean_source_dedup` proof，物理权重依赖已闭才退役clean。此时paired未验收；后续两分支由共同S7/history驱动。最终paired/export重建early并核对clean complete，最终仅退役repeat。early不纳入之后才生成的退役收据，paired不纳入依赖自身的repeat收据，避免循环和不可变重建差异。

独审发现两处P1并修复：pending清理resume原helper虽能恢复删除，但须在继续删除clean前重新认证保留source S32并匹配早期certificate/path/state signature；所有plan/complete必须各自绑定当前sequence/seed/owner/proof/commonS7，两者同时存在必须完整字典相等。缺原early、foreign proof、错误anchor、损坏source、不等价或failed clean拒绝，保留未删除的权重。复用已成功阶段不会新启动GPU训练。

独立审查代理实际核准native09 JSON SHA `8b9d2f83944092f4f6233781d0e37be2782f4e4000ca58f7f7b2721b5f097c37`、log SHA `0430da827168dff24a6cb1c2bff0d37dd3f7b1084372e3364934c42d4491feb1`，133 passed/75.88s/rc0，35 stage源码及12份隔离/实际v01 canonical逐SHA匹配，test SHA `bcdeed46f61a6ac67398553946c95a1669863f24c43d23d89b6b68e0f9fc3d9c`。新增16病例使用真实小torch状态/cert/物理unlink，包含clean及final-repeat first-unlink中断→execute原proof续做→paired重建完全相同；训练validate_run/validate_unit、目标/S0等明确使用夹具，不能宣称新增真实25-unit/GPU实验。native08的130pass/1夹具别名失败完整保留，修正夹具后没有降低生产校验。

预算85f9c80f候选保留原48GiB和32MiB/unit/256MiB/eval，19stage独立枚举为peak13/final11/453updates/89eval。全物理峰需求654342307909bytes；将全部48GiB余量预先扣除后，原launcher累计要求最紧为666741017387bytes（约620.951GiB，seed15 repeat，retained11）。独审使用实际model manifest/profile编译原guard AST验证正例与六个内存反例：只改weight_bytes、bool、改safetensor bytes并重签、改文件hash并重签、换profile、换workflow均拒绝。签名须重算files digest、weight_bytes严格为int且等于两safetensors认证总和7509337976，profile固定35c489。此预算仅成功路径；新增失败祖先需fresh累计资源检查再resume。

CLI02操作e45050b3已独审可执行：native09 exactcandidate，fresh GPU-venv进程无PYTHONPATH/CUDA；bootstrap去除反例与当前真实16row U8/原生unit/目标/同export函数正例，source/import及unit小证据入新收据。实际CLI/actualfree以及新witness仍以之后收据为准。当前代码P0/P1已闭合，尚无正式T32科研结果。

## 2026-10-13：正式首个 clean U8 实际独审

原full U8审计e682c641已核准：fresh validate_unit8、17份小证据及真实completion/optimizer8/finite通过，新S8 cert d980fa62的run/learning/parent与旧保留S8 cert b9766de5的来源正确。两边小metadata中的全部worker参数/buffer/Adam/groups/scheduler/RNG指纹、driver RNG/cursor/position/step精确相等。大分片认证明确引用根代理210bbe9e操作的实际全SHA运行，独审未重复重哈；不把小指纹检查冒充重新读取模型。

独审指出S16 canonical keep1会删除整个新S8目录，使原收据证书路径失效（P1）。根代理在删除前实际保存cert及全部非model/optim共9成员167667bytes，并追加810e306b durable binding；原e682收据不改，source或fullS8权重不pin。独审逐项核准9成员SHA、原path/SHA至captured path/SHA映射、marker、18份durable evidence、原full审计SHA不变，并实际确认完整恢复校验拒绝该metadata-only目录，P1闭合。此时S16证书尚未出现；本核准只覆盖实际U8与证据生命期，不是T32末态或整个配对组。

2026-10-13追加实际S16阶段审查：fresh核准units9–16全部8个原生单元与probe16，93份收据证据一致（8b9e3bbc及585c32aa）。S16 owner/learning/parent、完成标记和checkpoint收据一致；未重新哈希S16大模型。原本run S8目录已真实滚删，810e306b mapping、9成员及18份durable evidence仍全部SHA通过；原old clean S8和failed工程S8全部成员存在且尺寸正确。范围内无P0/P1，只核准units9–16/probe16与已提交S16的小证据，不构成T32末态或配对组验收。

## 2026-10-13：首条实际 clean T32、early proof 与退役独审闭环

fresh validate_unit8:32共25个原生单位（275条evidence），validate_probe7/8/16/24/32共5点（25条evidence），128base/768固定候选/P256及RNG/cursor保护门通过。freeze b36195、current21/raw15/profile/seed11、clean无注入/native_reanchor8:32一致；stage accepted ccd178bb的25单位、learning/parentS7/replay及324份证据正确，319small实际SHA核对并覆盖300单位/探针证据。新S32 cert4bd0022c与保留source32 cert5e4bbecd的worker参数/buffer/Adam/groups/scheduler/RNG、driver/cursor/position/step逐字段相等。大模型物理认证明确引用实际launcher/sequence运行，独审未重复重哈。

实际early proof bb6010a8完整绑定该accepted、source32及共同S7、freeze和physical-clean-weight依赖范围；paired组仍pending。退役plan与complete完全同字节/SHA ff12ff5d，只移除8model/optim共46,302,164,640bytes，另外8成员与certificate均SHA有效。475evidence去重363，其中362small fresh SHA通过；唯一910,107,314-byte fixed_inputs沿用已实际probe/冻结认证，不重复hash。完整恢复检查实际拒绝retired clean32。共S7、旧六完整状态与原failed工程S8全部成员存在、尺寸及小证据保持，退役clean32全指纹仍与source32相等。

本阶段无未闭P0/P1。corrupt仅核准同S7/history/freeze启动身份及实际恢复；不能将本审查扩大为其U8注入、repeat/paired五组或持续性科研结论验收。

## 2026-10-13：首条实际 corrupt T32 整条独审

实际S7起点76e7e00、U8输入acf3343及完整S8审计125f1d1a均核准：只原row5终局位置260增加1，独立GRPO advantages/returns最大误差0，其余三组12行不变；共同S7两role固定评分距离0。物理全S8与原E8完整状态相等，9小成员marker1172c2c0及永久证据路径通过，full restore实际拒绝小副本。S16滚删原S8后，125f审计的全部持久证据仍有效。

整条fresh25 units通过，只有U8 injected true；U9–32各12固定tensor/19metadata匹配、revision none。五点probe128/768候选/P256/RNGcursor核准。stage accepted237a6bf4的25单位、learning102347、共同S7、history、origin_manifest及324evidence覆盖严格一致，319small fresh SHA全部通过，5次910MB cache引用未重复哈希。S32 cert30058c4e/state81d2d62b/owner/4workerfinite/driver/cursor/next33与completed32/cp32一致；大S32认证采用实际wrapper/parent完成的验证，独审未重复大权重hash。范围内无未闭P0/P1，完整corrupt32保留。repeat、paired及五seed研究结论尚未验收。

## 2026-10-13：首个实际 paired block 与 repeat 退役独审闭环

paired288dc35e严格覆盖三份accepted、75更新、15分支评测点与原first target08553、同freeze/source21/raw15/profile/learning/commonS7/history。1335条证据去重后752small fresh SHA全部通过，唯一910MB固定输入不重复哈希。独立逐token均方、候选等权、case/role RMS与两role平均重算8/16/24/32非零D及role值，最大差1.39×10⁻¹⁷，floor均0。S0实际核对current control的768候选/P256与历史0小状态指纹，明确为common一份测量复用。repeat、retiredclean及保留source32的小全状态指纹相等。

真实repeat plan/complete同字节/SHA70497951，绑定原paired288dc35e，只移除其global_step32的4model/4optim共46,302,164,640bytes；9small/certificate有效，ordinary full restore实际拒绝。退役475行证据去重后362small fresh SHA通过，唯一固定cache不重哈。原clean earlybb6010/退役ff12完全不变；七个protected状态（old6+原failed工程8）的证书、成员尺寸、小SHA保持，failed原−9/无completed9/cert9不改；corrupt32完整成员仍在且无退役收据。大物理认证引用实际wrapper/parent/helper，未二次模型hash。

单seed T32 D=.05543707034、floor0、Q≈.04543707034仅为描述；ε=.01保持冻结AI提案来源与userapprovedFalse。范围内无未闭P0/P1，序列实际完成1 block，五seed区间/结论未完成。prefix12当时只核准preparation启动身份，不能提前核准S7或prefix初始退役。

## 2026-10-13：第二种子prefix重建与实际S0退役独审闭环

accepted0a80371b、rebuild7c37b5b8与同字节retirement plan/complete96b39d62均实际核准。fresh7个原生单元无注入，seed12/source21/raw15/profile/freezeb361/horizon32/stop7/pin7一致；新S0/S7完整小指纹逐字段等于历史reference。两个固定probe128/768及cal64/diag128、RNG/cursor保护通过；正式P0单独核准冻结旧source12真实P256，未用cal64替代。

只退役自身S0目录，fullnewS7保留。142条证据覆盖accepted/checkpoint0/历史比较/capture，128条去重small SHA通过；初始9小成员仍在，capture及旧reference实际拒绝完整恢复。旧六状态、失败S8、source32及首corrupt32保持。大物理认证引用实际wrapper/parent，未二次哈希大权重。范围内无未闭P0/P1。clean12只确认启动身份，未扩大到T32、paired或五seed科研结论。

根代理另核实际clean12 resume_loaded a601d0c8与共同S7 certificate/smallfingerprints，11条证据的d35c1bce确认四rank状态、driverRNG/cursor/learning/freeze一致；不声称通用sampler bitwise。交接文件一次误覆盖已从成功历史操作的文本恢复并对照事前12500字符完全一致，实验源码和raw未受影响；恢复凭证保存在私有runtime。


## 2026-10-13：完整五计划种子仅一有效时的绘图P1与隔离候选

原CLI实际n=0/1/2分别return0/1/0：paired_interval(n1)省略epsilon_D，原export统计尾段保持省略，plot_gate1.py:42产生float−None TypeError。真实原反例仅执行原统计AST与实际绘图CLI，合成夹具不替代模型/原始结果验收。

原绘图a8d0a93b与全部学习15/工作流6/旧35保持。隔离候选8aae1b3e在n1缺字段时从已审核CSV的corrected_D_T−Q_s恢复原ε，检查finite/positive及pair一致，figure_manifest记录ε和来源。独审16个实际CLI条件全部符合预期，f0ec6816收据覆盖6正例和10拒例；n1/n5实际PNG视觉QA b1658066通过。负校正/负Q保留，n1均值有效但CI不可用且uncertain，n2有CI仍uncertain，n0无估计。均值、CI、阈值与判定未调整。本候选范围无未闭P0/P1，尚未部署。

codeReview05的nested evidence还绑定原live plot a8和stage9旧副本，因此最终保留原路径和35文件历史身份；等全部正式训练完成后新增plot_gate1_reviewed.py作为独立后处理入口，另列新SHA并做实际位置CLI验收。原冻结export与全部nested证据继续可重建。当前GPU学习不依赖绘图入口，继续原冻结实验；最终真实结果导出、绘图和报告另行验收。

## 2026-10-13：第二种子clean实际U8、起点评测与持久小证据

真实op539f3d50与前审597d2550仅差移除unused import。独审fresh U8的11条原生证据、12固定张量和19元数据通过，revision none。restore7的flags/driver/cursor以及probe7/8的128/768/P256保护通过。U8 receipt9262dd3a、durable marker8ea1ea62正确，39条去重small SHA有效；9capture/mapping和完整恢复拒绝实际通过。

U8 receipt的23条evidence全部使用持久捕获路径，后续S16滚退后的实际复核仍待。S8大物理认证引用实际native launcher，未重新模型/910MB cache哈希；未声称source8全状态相等、T32或paired验收。范围内无未闭P0/P1。操作入口一次错误手填SHA常量在capture前被守卫拒绝，改用真实部署SHA539f后成功，未改实验raw或训练状态。

## 2026-10-14：clean12 T32、source32等价、early proof与实际退役独审

accepted c8c3e7e3、early eb0fa7d4与plan/complete bc927710均真实核准。fresh25 validate_unit(expectedFalse)逐个12tensor/19metadata/revisionnone匹配，五点128/768/P256/RNGcursor通过；source21/raw15/profile/freezeb361/seed12/learning5c/commonnewS7/history一致。accepted324行、early335行、retire475行分别核对311/322/362份去重small SHA，910MB固定缓存不重复哈希。

新clean32 certf857ccaf/state5b7a871b的小全状态与实际保留oldsource12完整S32及新reference32（certa480ef06）逐step/workers/driver/cursor/position相等。source完整成员尺寸保持；大物理认证引用实际native/parent。plan与complete同字节并绑定原early SHA，仅自身8model/optim共46,302,164,640bytes消失，9small保留，普通full restore实际拒绝。

原S8/S16目录均已滚退；原9262的23条持久证据、8ea1的9成员/mapping仍全SHA有效，不依赖失效checkpoint路径。newprefix7/source12full32、old6+failed8、首corrupt32与首paired288d保持。本范围无未闭P0/P1。日期steering后root对过时S16路径的只读FNF无写入、无passed收据，不计训练失败。corrupt12只观察启动，恢复/注入/完整分支/paired及seed CI另验收。

## 2026-10-14：corrupt12起点、唯一U8及认证小证据保全实际独审

restorea2eadbc2、initial7fb578b3c与input7dbb2763核准。25条去重small SHA通过，4×4 generation、8固定tensor和19metadata匹配；原定firsteligible row7/certf5fb为显式1356 vsGT−158、nativeoutcome0、nontrunc，仅terminal567增1，native_info/native_returns保持（目标usedreturn增1）。独立样本std GRPO全部adv/returns最大误差0，其它三组12行不变。S7双role D0/P256/128768/RNGcursor保护通过。

实际committed8/S8 owner/learning/parent及finite小状态一致，1414fd78的23evidence与probe8共28条，其中27small fresh SHA通过，910MBcache不重复hash。9capture及mapping与仍在的S8小指纹相等，c087fecf marker有效；ordinaryfull restore实际拒绝metadata-only。最终evidence仅依赖持久路径。probe8 1d8c49fa的128/768/P256/RNGcursor通过。

本范围无未闭P0/P1。大S8认证引用actual native；actualS16滚退后持久复核尚待。未声称旧source8全状态等价、T32、paired或seed CI。


## 2026-10-14：corrupt12 U9–16、point16及S8滚退后持久证据闭环

fresh8个无再次注入原生单元通过，各12固定tensor/19metadata及revisionnone一致。point16 ac8969e2的128/768/P256/RNGcursor保护有效。117条去重small SHA核准，910MBcache和模型不重复哈希。

S16 owner/learning/parent、completed16与cp16一致，小成员SHA及完整成员尺寸有效；物理认证引用native。原S8实际已滚删，原1414的23条证据、c087 marker/9members/mapping仍可复核；postroll da09fa98的36evidence全部使用永久raw/capture路径，ordinaryfull restore仍实际拒绝metadata-only。old6+failed8/newS7/source32/corrupt11保持。本范围无未闭P0/P1，不构成T32、paired或seed CI验收。


## 2026-10-14：corrupt12 U17–24、point24和第二次滚退闭环

fresh8个无再次注入单元及point24 cd47a8c2的128/768/P256/RNGcursor保护通过，129条去重small SHA有效。S24 owner/learning/parent、completed/cp24一致，小成员SHA及完整成员尺寸有效；物理认证引用native，未二次模型/cache大哈希。

原S8/S16均已滚删，fdec1d3c的14条、da09fa98的36条及原1414的23条永久raw/capture证据仍可复核，c087的9成员/mapping通过，ordinaryfull restore实际拒绝metadata-only。old6+failed8/S7/source32/corrupt11保持。本范围无未闭P0/P1，不构成T32、paired或seed CI验收。

## 2026-10-14：corrupt12完整T32及实际accepted独审

accepted29aec5b3严格绑定fresh25单元，只有U8原row7/f5fb注入，U9–32的12固定tensor/19metadata/revisionnone通过。五点128/768/P256/RNGcursor有效；source21/raw15/profile/freezeb361/seed12/new共S7/history匹配。324条证据中311条去重small fresh SHA有效，模型/910MBcache不重复哈希。

S32 cert5f05f8fa/state858d5960/owner/learning/parent、finite小状态与完整成员尺寸一致，完整S32保留且无退役收据。物理认证引用已实际native/parent。最终S24滚退后，U8/S16/S24/S32的23/36/14/15条永久证据仍有效，c087 capture9及mapping通过，ordinaryfull restore实际拒绝metadata-only。old6+failed8/source32/newS7/corrupt11保持。本范围无未闭P0/P1；repeat12只观察启动，不包含其restore/floor/paired或seed CI验收。

2026-10-14第二种子clean-repeat U8已独立验收：共同fullS7恢复、实际7/8点128/768/P256与RNG/cursor保护通过，两role评分floor在这两点实测均0。U8无注入，12固定tensor/19metadata匹配；完整S8认证后9小成员167,796bytes持久保存（markere83c13a1），605b2470的23条证据不依赖原checkpoint8目录。独审核准fresh11单元证据及31份去重small SHA、9成员mapping/当前S8小指纹一致，并实际确认普通full restore拒绝metadata-only；大模型与cache认证引用真实native，不重复哈希。本范围无未闭P0/P1。62小文件4,032,987bytes已本机逐SHA同步。T32后续floor、实际滚退后证据复核、第二paired和五seed统计仍待，完整paired进度仍1/5。

2026-10-14 clean-repeat12第16阶段已独审：U9–16八个无注入原生单元、点16的128/768/P256与RNG/cursor保护通过，双role floor16实测均0；88单元证据及118去重small SHA核准。完整S16证书d42cd5a8/statec7fd2f8f提交，保存707.4981秒、46,302,326,498bytes，原S8已真实滚退。ec18d987的36永久证据、原605b的23证据及e83c的9成员/mapping继续有效，普通完整恢复实际拒绝元数据目录。本范围无未闭P0/P1。136小文件6,317,308bytes已下载并核对服务器SHA，原62个U8成员亦重新核准。下载辅助脚本首次TarFile迭代错误发生在本机写入前，修正后通过，失败记录保留。后续T32、第二paired及五seed结果仍待。

2026-10-14 clean-repeat12第24阶段已独审：U17–24八个无注入原生单元、88证据及131去重small SHA核准，点24的128/768/P256与RNG/cursor保护通过，双role floor24实测均0。完整S24证书d6b21eb3/statead6fd888提交，保存667.4974秒、46,302,326,610bytes，原S16确已滚退。db81a3b9的14永久证据、原U8的23条、S16的36条及e83c的9成员/mapping持续有效，普通full restore拒绝metadata-only。本范围无未闭P0/P1，后续T32及第二paired仍待。

2026-10-14 clean-repeat12 T32已完整独审：25个无注入原生更新及7/8/16/24/32五点128/768/P256、RNG/cursor保护通过，正常return0，actual accepted9b4abda2/324证据。逐token比较五点双role floor均为实测0。完整S32证书9f1c8ee1/state9af38c0e提交，46,302,326,466bytes、保存660.8018秒，完整worker、driver RNG、cursor和position指纹与保留source12及retiredclean12精确相等。独审核准整链394去重small SHA及accepted中的311small；原23/36/14条永久证据和9成员/mapping经全部滚退保持有效，full restore拒绝metadata-only。物理认证引用native/parent，本范围无未闭P0/P1。231小文件10,539,279bytes已本机核对server SHA。第二paired及repeat权重退役仍待，当前完整paired进度仍1/5。

2026-10-14第二个完整paired block已验收（2/5）：seed20261012配对proof5899a6a9覆盖75个原生更新及15个分支评测点、1343条证据，762去重small SHA经独审。逐token→候选均方→case/role RMS→双role平均的独立重算最大差2.77556×10⁻¹⁷。T32 D_score=0.05540106903、floor=0，扣冻结ε_D=.01后的单seed Q=0.04540106903；ε保持AI研究提案身份，这两个已完成种子的描述数值尚待完整五seed统计判定。共同S0仍为一份真实测量复用。repeat退役plan/complete原字节相同c47fc480，严格绑定原paired5899及原clean early eb0fa7d4/退役bc927710；仅自身S32八个model/optim共46,302,164,640bytes退役，9小成员、raw和全部指纹保留，该点不可直接恢复。475退役证据中362small及最终共921unique small SHA通过，普通full restore实际拒绝退役目录，old6+failed工程S8/source12/commonS7/corrupt11&12保持。本范围无未闭P0/P1。15小文件1,045,195bytes已本机核对server SHA，prefix13于00:16:32.119789UTC实际启动；其S0/S7真实重建、剩余三块及全量统计/图表/成本/科研报告仍待。

prefix13起点S0已实际独审：cf35deff的16永久证据及69去重small SHA通过，seed/owner/current learning与原历史reference绑定正确，9个小捕获成员161,729bytes及五项完整小指纹一致，capture与历史reference均被普通full restore实际拒绝。本范围无未闭P0/P1，cal64起点评测、U1–7、S7及前缀终态继续验收。

2026-10-14第三个固定种子前缀已完整验收：seed20261013重放原U1–7后正常stop7/return0，acceptedb86c2073覆盖7单位/121证据，历史S7重建证明e3fa2b28通过。新S0和保护S7的完整worker、driver RNG、cursor及position小指纹与历史认证状态精确相等；0/7是工程cal64及固定128/768探针，正式P0另绑定冻结历史P256。独审核准完整前缀176small SHA、accepted中的117small及原生77单元证据。S0退役plan/complete同字节b7995ef9，142证据中的128small、生命周期共250unique small SHA通过，仅自身8大成员16,264,940,296bytes退役，9初始小成员161,729bytes/全部raw/完整S7保留，元数据与历史reference普通full restore实际拒绝，删除后cf35的16永久证据继续有效。本范围无未闭P0/P1。123小文件3,821,764bytes和追加7文件163,901bytes已本机核对server SHA。clean13按原4–7从该完整S7实际恢复，44f755d2的11证据核验原生恢复flags和解码RNG/cursor内容指纹通过；首次辅助错比文件SHA在写passed前被拦，纠正并保留失败记录，训练/原始数据保持。P256起点评测和第三paired尚待，完整paired仍2/5。
