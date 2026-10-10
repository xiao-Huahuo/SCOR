# v01 baseline 实施与验收清单

日期：2026-10-10（Asia/Shanghai）。依据 [baseline_design.md](baseline_design.md) 和 [design_review.md](design_review.md)，用户已要求开始工作。以下是实施清单，未勾选项目不算通过；六Gate仍按自己的版本执行。

| 用户要求/设计阶段 | 实现位置与工作 | 实际验收 |
|---|---|---|
| 统一服务器目录、补规范 | 科研开发规范首要10；server_migration.md；CURRENT_STATE.server | 整体目录迁移、文件身份/关键hash及环境检查已通过 |
| A：自己的模型与原生闭环 | 独立候选配置/入口；固定模型全文件身份 | CPU配置检查；真实四卡视觉/工具/Verifier/Repair/GRPO与容量实测待执行 |
| B：完整状态与恢复 | 完整分片、driver/worker RNG、数据位置、step0/保护点、原子提交 | 部分写入/错配置/损坏拒绝；实际中断、固定下一unit与晚期分叉待执行 |
| C：数据和记录 | train/calibration/probe/evaluation独立scene；稳定case/group/slot；原始收据 | 独立split、图像身份、参数生效值和逐角色记录；不以prompt文本配对 |
| C：重放与修订 | native reanchor；终局false-positive；原生整组advantage重算 | 同unit无修订更新等价；只改一项reward而全组重算；实际GPU对照待执行 |
| C：评测 | Solver显式答案、fallback分列、按case聚合；独立评测随机流 | 相同文本不同图像不合并；Verifier数字/工具输出不替代Solver答案 |
| D：正常baseline | 独立校准后冻结，正常多seed训练、固定评测与终态 | 新模型/数据/seed/阈值尚未冻结；全部正常结果可复核后才算立起来 |
| E：Gate准备 | 同prefix/同U[k]的正确与错误reward真实更新 | 完整状态分叉、干预证书与差异范围验收后进入Gate1 |

当前共享GPU0–5忙，仅6/7空闲；约33GiB余盘。继续开发可由CPU验收的契约与接口，四卡任务须重新预检。2B候选与32轮等数值为设计提案，校准不得访问正式held-out选配置。

2026-10-11资源更新：用户授权清理旧3B权重/优化器，实测可用空间增至75.87GiB，旧恢复点已退役。已开始编写records、四split数据准备和显式Solver评测模块；原生adapter、独立训练入口、完整状态/重放及这些新模块的测试尚未完成，不将其视为已验收baseline。

实际代码修改完成后，从设计到代码、代码到设计核查，再用独立反例、手算或参考原生函数验证。未闭环P0/P1实现问题阻止依赖它的正式实验。旧15份pilot源码及历史产物保留；此任务没有产品界面，界面冒烟不适用，CLI和原生链路需实际验收。
