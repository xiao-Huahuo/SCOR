# 2026-10-10：启动 Agent0-VL 小规模复现与验证

### 现状

已有 SCOR 研究方案及文献，尚无训练实现或实验结果。用户授权在当前项目开发，并尝试四张 4090 上的小规模复现和验证。已读取开发规范与服务器规定。

### 实施方案

按 v01 组织代码与记录，固定官方代码版本，在个人服务器目录和隔离环境内运行。先验收原生链路与 checkpoint 恢复，再尝试受控 reward 错误注入。

### 完成与未完成状态

已建立本版方案、根目录说明、交接状态与敏感文件忽略规则。连接、资源检查、上游源码准备、GPU 训练及验证尚未完成。

# 2026-10-10：四卡原生 GRPO、checkpoint 恢复与实际 batch 验证

### 现状

服务器实际有八张RTX4090，本任务选择GPU4–7并保持最多四卡。模型采用已有Qwen2.5-VL-3B-Instruct，数据在服务器生成16train+4val合成视觉题。个人盘初始余约95GiB，完整step1 checkpoint写完后余约32.8GiB。

### 实施方案

在独立目录/venv固定Agent0上游commit与依赖，保留工具、Verifier、Repair、原生奖励与GRPO/FSDP。按实际报错修复跨DP宽度和无critic指标条件，调整24GB推理内存；原始评分和更新规则不改。保存真实generation/updater batch、不可覆盖收据和完整checkpoint；验证恢复后的下一批更新，再检查单条reward修改的组内影响。

### 完成与未完成状态

- `native_stop1_10` 已执行第一步update并保存完整checkpoint，随后指标代码报错；失败状态与缺失GPU指标事实保留。
- 修复指标后，`native_resume2_01` 从step1完整模型/optimizer/scheduler/RNG/dataloader恢复，使用样本4–7继续到step2，正常退出；16轨迹、3315非零advantage token、参数抽样发生变化且有限，11次工具成功、23个已验证步骤、5次Repair。
- 服务器10项测试通过；CUDA/BF16 kernel通过。实际batch的原生advantage重算一致；单条return减2使同组4条advantage变化，其中未改reward的一条发生符号翻转。
- 四题验证前后均3/4，未观察到提升。3B工程适配不作为论文数值复现或六个Gate通过。
- 配对GPU reward试验 `native_corrupt1_01` 因GPU4/5被其他任务占用，在预检停止。step2最终模型未保存；vLLM RNG逐位恢复未验收。
- JSON/CSV/log与checkpoint完成证明已同步本机；完整快照与大型tensor在服务器。已更新README、版本说明、方案、审查、结果报告与CURRENT_STATE/TASK_STATE。

# 2026-10-10：初始化 Git 与配置仓库发布

### 现状

用户指定将当前项目推送到 `git@github.com:xiao-Huahuo/SCOR.git`。本地原先未建立 Git；已有代码为 Agent0-VL 基线和实验支架，SCOR 核心算法尚未接入。工作目录同时包含本机凭据、运行环境、训练产物和约124MiB文献PDF等本地资料。

### 实施方案

建立 `main` 分支并配置 `origin`；完善 `.gitignore`，跟踪源码、配置、规范和研究文档，排除凭据、个人交接记录、运行环境、输入输出、模型及文献PDF下载副本。提交前检查实际忽略规则、暂存文件和凭据特征；使用普通推送，随后核对本地与远端分支的commit，不强制覆盖历史。

### 完成与未完成状态

已建立main及origin，65个UTF-8文件纳入Git；实际凭据、运行环境、输入输出、模型和PDF路径均被忽略，暂存凭据特征扫描未发现匹配，源码/配置blob与本机字节一致。新增Git文件与说明的空白检查通过，既有文献笔记中的空白警告保持原样。按用户提供的终端脚本使用本机HTTP代理，并为本仓库SSH设置显式CONNECT代理。初始提交 `3064bf33b7281bb1f1788e3d22597f8cd819911c` 已成功推送，随后核对远端main与该SHA一致。本次只发布Agent0-VL基线和研究文档，SCOR核心算法仍未接入。
