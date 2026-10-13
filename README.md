# SCOR

研究方案见 [idea.md](docs/RSI-Agent/idea.md)。开发和服务器使用分别遵守 [科研开发规范.md](科研开发规范.md) 与本机私有的 `服务器.txt`。

服务器统一工作目录为 `/data1/yangbin/SCOR/`，本项目创建或下载的资产均存放在其内部。此前 `/data1/yangbin/SCOR_v01` 已整体迁入，旧路径作为兼容链接保留；迁移与占用核对见 [server_migration.md](docs/v01/server_migration.md)。

科研开发规范已同步用户提供的新版本，加入强制代码正确性深度审查，并保留服务器使用条款；后续开发需完成设计/实现双向核查、参数核查与独立反例验证，未闭环的P0/P1实现问题不能推进正式实验。

v01 机制 baseline 已完成，当前推进 `v02 / Gate1` 的实现与验收，方案见 [Gate1 协议](docs/v02/gate1_protocol.md)，真实工程状态见 [实施验收记录](docs/v02/implementation.md)；后续所有训练与修复实验最多使用四张 RTX 4090，分支串行。深度设计见 [baseline_design.md](docs/v01/baseline_design.md)，六个 Gate 的协议见 [gate_protocol.md](docs/v01/gate_protocol.md)，需求与风险审查见 [design_review.md](docs/v01/design_review.md)。v01 为 baseline，v02–v07 对应 Gate1–6；原论文模型与绝对分数复现不作为前置。

固定Qwen2.5-VL-3B-Instruct/data08已通过cal11的14项健康规则、三个真实GPU恢复/重放/重复评测对照，以及六项冻结证据和累计磁盘预算。[正式冻结记录](docs/v01/baseline_freeze.md)固定源码、模型、数据、环境与三个seed各32步。三个seed的全部32步、五次评测和完整S32均正常完成，D已验收；同S7/U8的E单奖励实际更新与独立配对审计也通过。当前结果见 [baseline_report.md](docs/v01/baseline_report.md)，逐句实现与反例审查见 [baseline_code_review.md](docs/v01/baseline_code_review.md)。全部D完成才称正常baseline验收，六个Gate另行检验科研现象。

以下为2026-10-10历史pilot证据：Agent0-VL 的 3B 核心链路已在四张 RTX 4090 上执行两次原生 GRPO update，并从第一步的完整 checkpoint 恢复继续训练。已观察到真实工具、Verifier、Repair，恢复后的第二次update观察到参数抽样变化。2026-10-11复核原生constant scheduler：warmup=0时首轮使用LR=0，两次update不能表述为两次权重变化。真实 batch 的离线 reward 修改影响了同组四条轨迹的 advantage；配对 GPU 更新对照因空闲卡不足尚未执行。这些结果属于小规模工程验证，未完成新的机制 baseline、论文 benchmark 复现或六个验证 Gate。

当前源码实现 Agent0-VL 基线与实验记录；SCOR 核心算法尚待接入。

本轮文档核对已补明[数据选型与覆盖范围](docs/v01/baseline_report.md#当前数据与选型范围)、[六份训练状态的各自用途](docs/v01/storage_cleanup.md#实际清理与保留)，并修正验收状态及清理后的旧备份表述。设计、冻结、实施验收、报告、审查、清理同步和变更记录均已具备；v02 已开始实现固定数据持续性实验；当前117项原生检查、新源码clean U8及真实GPU中断恢复至S9的完整状态/评分等价已通过，初始S0/S1、probe128/P256控制亦已验收通过，正在部署最终代码并核验冻结预算；尚未冻结或获得Gate1科研结果，后续Gate保持实验提案。

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
