# SCOR

研究方案见 [idea.md](docs/RSI-Agent/idea.md)。开发和服务器使用分别遵守 [科研开发规范.md](科研开发规范.md) 与本机私有的 `服务器.txt`。

当前版本为 `v01`：Agent0-VL 的 3B 核心链路已在四张 RTX 4090 上完成两次真实 GRPO 更新，并从第一步的完整 checkpoint 恢复继续训练。已观察到真实工具、Verifier、Repair 和参数变化。真实 batch 的离线 reward 修改影响了同组四条轨迹的 advantage；配对 GPU 更新对照因空闲卡不足尚未执行。这些结果属于小规模工程验证，未完成论文 benchmark 复现或六个验证 Gate。

当前源码实现 Agent0-VL 基线与实验记录；SCOR 核心算法尚待接入。

- `solutions/v01/src/`：实验入口、配置、数据准备与验证代码。
- `solutions/v01/inputs/`：输入数据，Git 忽略。
- `solutions/v01/outputs/`：按运行编号保存原始收据、checkpoint 和分析结果，Git 忽略。
- `docs/v01/`：本版方案与审查记录。
- `.runtime/`：隔离的依赖环境与上游源码，Git 忽略。
- `CURRENT_STATE.json`：当前进度与证据状态。

本版方案见 [pilot_plan.md](docs/v01/pilot_plan.md)。运行前先执行服务器资源检查；服务器账号为共享账号，只在本项目新建的个人工作目录内写入，不改变已有任务和环境。

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
