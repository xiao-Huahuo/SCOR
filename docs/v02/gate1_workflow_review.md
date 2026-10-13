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
