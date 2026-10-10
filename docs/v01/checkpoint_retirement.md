# 旧3B工程快照退役

日期：2026-10-11（Asia/Shanghai）。用户确认新baseline从小模型重新开始，并明确要求清理旧3B试跑的权重和优化器状态。

## 实际清理范围

已删除 `/data1/yangbin/SCOR/solutions/v01/outputs/native_stop1_10/checkpoints/global_step_1/actor/` 下的4个 `model_world_size_4_rank_*.pt` 和4个 `optim_world_size_4_rank_*.pt`。操作使用核实过的8个明确文件路径；删除前确认项目标记、原完成证书和训练签名、文件身份、没有符号链接或其它硬链接，且无本项目训练/Ray进程。没有清理其它运行或共享基座模型。

删除文件逻辑大小共46,302,164,640 bytes（43.122GiB），其中权重15.1GiB、优化器28.0GiB。操作前后服务器可用空间从32.751GiB增加到75.873GiB。

## 保存与验收

实际核对49个保留文件的SHA-256未变，包括原始轨迹、updater batch、日志、配置、manifest、运行状态、分析结果及历史完成证书。共享3B基座配置hash未变，隔离环境保留。已逐个确认8个目标文件不存在，并实际调用旧 `complete_checkpoint`，结果为false。

原始运行仍保持其历史状态：step1更新及保存曾完成、之后指标汇总报错；step2曾恢复成功。清理不改写这些结果。原 `scor_complete.json` 继续作为当时保存成功的证据，不能当作当前可恢复状态；CURRENT_STATE登记资产已退役。旧checkpoint剩余约75KB的随机/调度/数据状态和证书，不具备完整续训能力。

小收据已同步本机：`solutions/v01/outputs/retire_3b_checkpoint_20261011_01/raw/plan.json` 和 `retirement.json`。这两份文件与其它运行产物一样被Git忽略；共享目录规则和可分享状态记录在项目文档。

后续小模型baseline和Gate仍需完整的关键状态与实际恢复验收；此次仅退役已结束的旧3B工程资产。存储预算根据新模型实测大小、保护点、滚动替换峰值及原始记录重新计算。
