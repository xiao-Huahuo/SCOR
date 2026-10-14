# SCOR

研究方案见 [idea.md](docs/RSI-Agent/idea.md)。开发和服务器使用分别遵守 [科研开发规范.md](科研开发规范.md) 与本机私有的 `服务器.txt`。

服务器统一工作目录为 `/data1/yangbin/SCOR/`，本项目创建或下载的资产均存放在其内部。此前 `/data1/yangbin/SCOR_v01` 已整体迁入，旧路径作为兼容链接保留；迁移与占用核对见 [server_migration.md](docs/v01/server_migration.md)。

科研开发规范已同步用户提供的新版本，加入强制代码正确性深度审查，并保留服务器使用条款；后续开发需完成设计/实现双向核查、参数核查与独立反例验证，未闭环的P0/P1实现问题不能推进正式实验。

v01 机制 baseline 已完成，当前推进 `v02 / Gate1` 的实现与验收，方案见 [Gate1 协议](docs/v02/gate1_protocol.md)，真实工程状态见 [实施验收记录](docs/v02/implementation.md)；后续所有训练与修复实验最多使用四张 RTX 4090，分支串行。深度设计见 [baseline_design.md](docs/v01/baseline_design.md)，六个 Gate 的协议见 [gate_protocol.md](docs/v01/gate_protocol.md)，需求与风险审查见 [design_review.md](docs/v01/design_review.md)。v01 为 baseline，v02–v07 对应 Gate1–6；原论文模型与绝对分数复现不作为前置。

固定Qwen2.5-VL-3B-Instruct/data08已通过cal11的14项健康规则、三个真实GPU恢复/重放/重复评测对照，以及六项冻结证据和累计磁盘预算。[正式冻结记录](docs/v01/baseline_freeze.md)固定源码、模型、数据、环境与三个seed各32步。三个seed的全部32步、五次评测和完整S32均正常完成，D已验收；同S7/U8的E单奖励实际更新与独立配对审计也通过。当前结果见 [baseline_report.md](docs/v01/baseline_report.md)，逐句实现与反例审查见 [baseline_code_review.md](docs/v01/baseline_code_review.md)。全部D完成才称正常baseline验收，六个Gate另行检验科研现象。

以下为2026-10-10历史pilot证据：Agent0-VL 的 3B 核心链路已在四张 RTX 4090 上执行两次原生 GRPO update，并从第一步的完整 checkpoint 恢复继续训练。已观察到真实工具、Verifier、Repair，恢复后的第二次update观察到参数抽样变化。2026-10-11复核原生constant scheduler：warmup=0时首轮使用LR=0，两次update不能表述为两次权重变化。真实 batch 的离线 reward 修改影响了同组四条轨迹的 advantage；配对 GPU 更新对照因空闲卡不足尚未执行。这些结果属于小规模工程验证，未完成新的机制 baseline、论文 benchmark 复现或六个验证 Gate。

当前源码实现 Agent0-VL 基线与实验记录；SCOR 核心算法尚待接入。

本轮文档核对已补明[数据选型与覆盖范围](docs/v01/baseline_report.md#当前数据与选型范围)、[六份训练状态的各自用途](docs/v01/storage_cleanup.md#实际清理与保留)。v02 已完成正式冻结和工程验收，五组固定种子T32实验正在执行：两个完整paired block及合法退役已独审，seed20261013前缀正在重建。当前进度2/5，五seed主统计与科研判定尚未完成；后续Gate保持实验提案。下方早期进展保留为历史，以末尾最新收据为准。

历史pilot入口的[间隔式检查点策略](docs/v01/checkpoint_policy.md)默认每2步保存；当前独立baseline按冻结条件每8步及保护/末步保存，三次完整32步已实际执行滚动替换。只有新完整状态提交后才清理本run非保护旧快照，原始单元收据保留；模型恢复以完整快照为边界，间隔内中断可能需要重算未提交段。2026-10-11按用户授权清理旧3B试跑的权重和优化器分片，释放43.1GiB；旧恢复点已退役，后续baseline的关键完整状态仍需保存。[清理记录](docs/v01/checkpoint_retirement.md)

- `solutions/v01/src/`：实验入口、配置、数据准备与验证代码。
- `solutions/v01/inputs/`：输入数据，Git 忽略。
- `solutions/v01/outputs/`：按运行编号保存原始收据、checkpoint 和分析结果，Git 忽略。
- `docs/v01/`：baseline 方案与审查记录。
- `solutions/v02/`：Gate1 源码、版本内输入和运行结果，入口见 [v02 README](solutions/v02/README.md)。
- `docs/v02/`：Gate1 协议、验收与结果文档。
- `.runtime/`：隔离的依赖环境与上游源码，Git 忽略。
- `CURRENT_STATE.json`：当前进度与证据状态。

已有工程试跑方案见 [pilot_plan.md](docs/v01/pilot_plan.md)。运行前重查物理4–7的GPU空闲状况及累计存储，只在本项目目录写入。3B完整初始状态约15.15GiB、含Adam状态约43.12GiB；当前引擎S0/S7完整恢复、U1/U8更新等价已验收。冻结时余盘420.5494GiB，累计新资产写入峰值与raw/安全预算418.6016GiB；这是实测预算，运行期间仍有空间检查。2026-10-12按用户明确要求清理旧校准/验收大权重、失败半写、重复初始/中间状态、旧压缩备份和下载残片；当前保留三个正式终态与首seed S7/S8、E S8，共六份完整状态，以及3B/环境/输入和全部原始记录。本地项目约0.8GiB，服务器约278.5GiB；历史证书与小元数据保留，但已退役旧状态不能恢复。[清理与同步记录](docs/v01/storage_cleanup.md)

结果与证据边界见 [pilot_report.md](docs/v01/pilot_report.md)，当前运行入口见 [v01 README](solutions/v01/README.md)。JSON/CSV/log 收据已同步本机；大型轨迹tensor仍在服务器 `/data1/yangbin/SCOR/`，旧完整checkpoint的权重/优化器已退役。第一步运行在保存 checkpoint 后因指标汇总报错，失败状态保留；修复后恢复运行正常完成，这些历史结果不因后续清理而改写。四道验证题始终为 3/4 正确，未观察到准确率提升。

## Git 管理

远程仓库为 [xiao-Huahuo/SCOR](https://github.com/xiao-Huahuo/SCOR)，`origin` 使用 `git@github.com:xiao-Huahuo/SCOR.git`，默认开发分支为 `main`。

Git 跟踪源码、配置、开发规范、研究文档、论文来源清单与下载脚本。`.gitignore` 排除服务器凭据、个人交接记录、运行环境、训练输入输出、checkpoint、模型、缓存与论文 PDF 下载副本。`.gitattributes` 固定源码与配置使用 LF，避免跨平台换行改变实验代码哈希。实验报告中的 `outputs/` 收据链接需要本地运行文件；这些文件和服务器凭据不随 Git 克隆分发。

```bash
git clone git@github.com:xiao-Huahuo/SCOR.git
cd SCOR
git status
# 选择本次修改文件后提交
git add <本次修改的文件>
git commit -m "说明本次修改"
git push origin main
```

论文下载来源保存在各文献目录的 `papers_manifest.tsv`；SCOR 分算法文献可用 `docs/RSI-Agent/SCOR_related_papers_by_algorithm/download_papers.py` 重新下载。服务器配置由使用者在本机单独准备，使用前遵守开发规范中的服务器规则。

Gate1最新进展（2026-10-13）：原实际空间预算未通过。已实现提前认证clean/source完整末态后退役重复clean权重的工作流，保持五种子/T32/原评测及clean→corrupt→repeat顺序；服务器隔离native09共133项检查通过。新codeReview、生产CLI及累计13槽实际预算仍待验收，尚无正式Gate1结果。

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

2026-10-14第二个种子corrupt T32已完整验收：seed20261012正常return0，25个原生更新及7/8/16/24/32五点128/768/P256保护通过，stage accepted29aec5b3。仅U8原row7/certf5fb终局outcome0→1，后24步固定12tensor/19metadata且无再次注入。完整S32证书5f05f8fa/state858d5960正确，46,302,326,466bytes、保存724.3529秒，长期保留。独审324条accepted中的311份去重small SHA、最终身份/finite小状态/full成员尺寸通过，物理认证引用actualnative/parent。全部滚退后，1414/da09/fdec/3435的23/36/14/15条持久证据及c087的9成员/mapping仍有效，ordinaryfull restore拒绝metadata-only；old6+failed8/newS7/source32/corrupt11保全。本范围无未闭P0/P1。240小文件10,624,119bytes已本机逐SHA同步。clean-repeat12已自动启动并实际恢复同完整S7（29a3ed04），floor、第二paired及全五seed统计尚待验收。完整paired进度仍1/5。

2026-10-14第二种子clean-repeat U8已独立验收：共同fullS7恢复、实际7/8点128/768/P256与RNG/cursor保护通过，两role评分floor在这两点实测均0。U8无注入，12固定tensor/19metadata匹配；完整S8认证后9小成员167,796bytes持久保存（markere83c13a1），605b2470的23条证据不依赖原checkpoint8目录。独审核准fresh11单元证据及31份去重small SHA、9成员mapping/当前S8小指纹一致，并实际确认普通full restore拒绝metadata-only；大模型与cache认证引用真实native，不重复哈希。本范围无未闭P0/P1。62小文件4,032,987bytes已本机逐SHA同步。T32后续floor、实际滚退后证据复核、第二paired和五seed统计仍待，完整paired进度仍1/5。

2026-10-14 clean-repeat12第16阶段已独审：U9–16八个无注入原生单元、点16的128/768/P256与RNG/cursor保护通过，双role floor16实测均0；88单元证据及118去重small SHA核准。完整S16证书d42cd5a8/statec7fd2f8f提交，保存707.4981秒、46,302,326,498bytes，原S8已真实滚退。ec18d987的36永久证据、原605b的23证据及e83c的9成员/mapping继续有效，普通完整恢复实际拒绝元数据目录。本范围无未闭P0/P1。136小文件6,317,308bytes已下载并核对服务器SHA，原62个U8成员亦重新核准。下载辅助脚本首次TarFile迭代错误发生在本机写入前，修正后通过，失败记录保留。后续T32、第二paired及五seed结果仍待。

2026-10-14 clean-repeat12第24阶段已独审：U17–24八个无注入原生单元、88证据及131去重small SHA核准，点24的128/768/P256与RNG/cursor保护通过，双role floor24实测均0。完整S24证书d6b21eb3/statead6fd888提交，保存667.4974秒、46,302,326,610bytes，原S16确已滚退。db81a3b9的14永久证据、原U8的23条、S16的36条及e83c的9成员/mapping持续有效，普通full restore拒绝metadata-only。本范围无未闭P0/P1，后续T32及第二paired仍待。

2026-10-14 clean-repeat12 T32已完整独审：25个无注入原生更新及7/8/16/24/32五点128/768/P256、RNG/cursor保护通过，正常return0，actual accepted9b4abda2/324证据。逐token比较五点双role floor均为实测0。完整S32证书9f1c8ee1/state9af38c0e提交，46,302,326,466bytes、保存660.8018秒，完整worker、driver RNG、cursor和position指纹与保留source12及retiredclean12精确相等。独审核准整链394去重small SHA及accepted中的311small；原23/36/14条永久证据和9成员/mapping经全部滚退保持有效，full restore拒绝metadata-only。物理认证引用native/parent，本范围无未闭P0/P1。231小文件10,539,279bytes已本机核对server SHA。第二paired及repeat权重退役仍待，当前完整paired进度仍1/5。

2026-10-14第二个完整paired block已验收（2/5）：seed20261012配对proof5899a6a9覆盖75个原生更新及15个分支评测点、1343条证据，762去重small SHA经独审。逐token→候选均方→case/role RMS→双role平均的独立重算最大差2.77556×10⁻¹⁷。T32 D_score=0.05540106903、floor=0，扣冻结ε_D=.01后的单seed Q=0.04540106903；ε保持AI研究提案身份，这两个已完成种子的描述数值尚待完整五seed统计判定。共同S0仍为一份真实测量复用。repeat退役plan/complete原字节相同c47fc480，严格绑定原paired5899及原clean early eb0fa7d4/退役bc927710；仅自身S32八个model/optim共46,302,164,640bytes退役，9小成员、raw和全部指纹保留，该点不可直接恢复。475退役证据中362small及最终共921unique small SHA通过，普通full restore实际拒绝退役目录，old6+failed工程S8/source12/commonS7/corrupt11&12保持。本范围无未闭P0/P1。15小文件1,045,195bytes已本机核对server SHA，prefix13于00:16:32.119789UTC实际启动，S0小状态与历史指纹已独审；其S7真实重建、剩余三块及全量统计/图表/成本/科研报告仍待。
