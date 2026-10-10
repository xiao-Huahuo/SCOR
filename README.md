# SCOR

研究方案见 [idea.md](docs/RSI-Agent/idea.md)。开发和服务器使用分别遵守 [科研开发规范.md](科研开发规范.md) 与本机私有的 `服务器.txt`。

科研开发规范已同步用户提供的新版本，加入强制代码正确性深度审查，并保留服务器使用条款；后续开发需完成设计/实现双向核查、参数核查与独立反例验证，未闭环的P0/P1实现问题不能推进正式实验。

当前版本为 `v01`，目标是建立自己的机制 baseline；后续所有训练与修复实验最多使用四张 RTX 4090，分支串行。深度设计见 [baseline_design.md](docs/v01/baseline_design.md)，六个 Gate 的协议见 [gate_protocol.md](docs/v01/gate_protocol.md)，需求与风险审查见 [design_review.md](docs/v01/design_review.md)。v01 为 baseline，v02–v07 对应 Gate1–6；原论文模型与绝对分数复现不作为前置。

设计建议优先验证 Qwen2-VL-2B-Instruct，保留共享 Solver/Verifier、工具、inner Repair、原生奖励与 GRPO。候选模型、规模和阈值尚未冻结；新的 baseline 入口与数据尚未实现。正常对照、完整恢复和评测通过才算 baseline 立起来，六个 Gate 另行检验研究现象。续接审查已修正 Gate5 双分支替换试验的更新预算；文档验收、当前发布状态和下一步按监管记录维护。

已有工程证据：Agent0-VL 的 3B 核心链路已在四张 RTX 4090 上完成两次真实 GRPO 更新，并从第一步的完整 checkpoint 恢复继续训练。已观察到真实工具、Verifier、Repair 和参数变化。真实 batch 的离线 reward 修改影响了同组四条轨迹的 advantage；配对 GPU 更新对照因空闲卡不足尚未执行。这些结果属于小规模工程验证，未完成新的机制 baseline、论文 benchmark 复现或六个验证 Gate。

当前源码实现 Agent0-VL 基线与实验记录；SCOR 核心算法尚待接入。

v01 已增加[间隔式检查点策略](docs/v01/checkpoint_policy.md)：默认每2步保存完整状态，末步/主动停止/关键点额外保存；可选保留数量仅在新快照完整后清理本轮旧快照。模型状态以完整快照为恢复边界，间隔内中断可能需要重算未提交段。减少份数仍不能解决当前余盘放不下一份完整快照的问题。

- `solutions/v01/src/`：实验入口、配置、数据准备与验证代码。
- `solutions/v01/inputs/`：输入数据，Git 忽略。
- `solutions/v01/outputs/`：按运行编号保存原始收据、checkpoint 和分析结果，Git 忽略。
- `docs/v01/`：本版方案与审查记录。
- `.runtime/`：隔离的依赖环境与上游源码，Git 忽略。
- `CURRENT_STATE.json`：当前进度与证据状态。

已有工程试跑方案见 [pilot_plan.md](docs/v01/pilot_plan.md)，下一步开发依据 [baseline_design.md](docs/v01/baseline_design.md)。运行前先执行服务器资源检查；服务器账号为共享账号，只在本项目新建的个人工作目录内写入，不改变已有任务和环境。当前约32GiB余盘不能预先认定足够新模型实验，需实测完整快照、raw增量和保护状态的共存预算。

结果与证据边界见 [pilot_report.md](docs/v01/pilot_report.md)，当前运行入口见 [v01 README](solutions/v01/README.md)。JSON/CSV/log 收据已同步本机；完整 checkpoint 和大型 tensor 保留在服务器 `/data1/yangbin/SCOR_v01`。第一步运行在保存 checkpoint 后因指标汇总报错，失败状态保留；修复后恢复运行正常完成。四道验证题始终为 3/4 正确，未观察到准确率提升。

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
