# 2026-10-12：本地与服务器资产清理、源码同步

用户在本对话明确要求清理本地与服务器垃圾，然后同步服务器、本地和GitHub源码。本次保留已完成实验结果与原始记录，退役不再需要的大模型状态及旧备份。此前冻结、校准、失败和恢复试验事实保持；旧证书只证明当时保存/验收成功，不能当当前仍有完整权重。

## 实际清理与保留

| 位置 | 清理前GiB | 清理后GiB | 实际释放GiB |
|---|---:|---:|---:|
| 本地SCOR项目 | 84.675 | 0.796 | 83.879 |
| 服务器SCOR项目 | 952.964 | 278.487 | 674.477 |

本地清理5个旧档案文件：cal01 S0、cal05 S1两个完整旧校准备份，以及cal01 S1/S2、cal06 S1三个下载残片。服务器按155项明确文件清单清理：非保留状态的model/optim分片、旧压缩恢复档案、SCOR自有2B模型。校准/三个一致性对照的大权重、失败半写、重复S0和另外两个seed的S7均退役；对应原始source/profile/响应/tensor/日志/指标/worker和driver/data元数据、证书保留。

服务器保留3B原始模型、GPU环境、固定上游代码、输入数据、全部raw/processed及以下六份完整训练状态，约258.7GiB：

| run | 保留完整状态 | 用途 |
|---|---|---|
| baseline_formal_20261011_seed_20261011 | S32 | 第一次正常训练的最终模型，保存baseline结果 |
| baseline_formal_20261011_seed_20261012 | S32 | 换随机种子的第二次最终模型，检查训练结果的稳定性 |
| baseline_formal_20261011_seed_20261013 | S32 | 第三次最终模型，避免结论只依赖一次训练 |
| baseline_formal_20261011_seed_20261011 | S7 | 错误发生前的共同起点，可恢复后分别执行正常与错误奖励更新 |
| baseline_formal_20261011_seed_20261011 | S8 | 正常奖励更新结果，作为同起点、同训练单元的参照 |
| baseline_formal_20261011_E_delta1 | S8 | 单次错误奖励更新结果，与正常S8比较及后续研究影响/修复 |

每份约43.122GiB，包含模型和继续训练所需的Adam、scheduler、RNG、数据游标等完整状态。S7表示完成第7次更新，S8表示第8次，S32表示32次正式更新结束。六份用途清单不表示后续科研Gate已经完成；当前只验收一次错误奖励真实进入更新。其余约19.8GiB包括3B原始模型7.0GiB、GPU环境8.5GiB，以及输入、原始记录、结果、上游代码等。

清理计划拒绝根目录外、symlink或保留状态中的目标，并确认无运行中的本项目训练。先核对6份状态完整文件尺寸/证书、所有删除目标的路径/尺寸/mtime/inode；删除后6份状态文件信息未改变，服务器22186个source/doc/input/raw/processed/3B文件及本地864个记录/源码/文档/凭据文件SHA全部相同。6份保留状态全部16个必要文件的内容SHA均已实际核验通过（并行3个CPU任务，364.977秒），结果在`retained_full_hash_validation.json`记录；这不重跑训练。

## 当前恢复与备份范围

只有上表六份训练状态可在当前服务器上完整恢复。旧校准、GPUcontrols、重复初始/中间状态的历史证书与worker/driver/data元数据仍在，但model/optim已退役。旧本机完整备份、下载残片和同服务器压缩恢复包都已按用户本次要求清理；cal06下载任务已退役，不能继续写成“96/145等待完成”。正式六状态目前没有本机或GitHub完整模型备份。3B基座可用，2B模型可在需要时按原固定revision重新准备。

## 源码同步验收

本地和GitHub清理前main均为`b9cb0766bc9a852c55f60aceb1a7ebeba62b3d35`。服务器原先是文件副本，无项目根Git仓库；104个tracked文件里44个文献/规范/Git配置等缺失，所有已有tracked文件内容与本地相同。同步以新提交的本地/GitHub同一快照为准，完整补齐tracked文件；通过Gitbundle建立服务器main索引与提交身份，不覆盖输入/输出/runtime，不传输凭据/模型/实验数据。最终核对三处HEAD一致、全部tracked文件SHA一致、工作区干净和Git忽略规则有效；最终SHA与过程记录保存在`.runtime/git/storage_cleanup_sync_20261012.json`。

来源记录：本地`.runtime/cleanup_local_20261012/plan.json`、`complete.json`；服务器`.runtime/cleanup_20261012/plan.json`、`deletion_events.jsonl`、`complete.json`、`retained_full_hash_validation.json`。历史冻结JSON与六witness的原字节不改；当前退役范围由本记录和CURRENT_STATE更新。学习引擎、profile、data08、3B模型与环境没有改动。

最终三处源码同步已实际通过：服务器、本地、GitHub实施提交均为`cdb86b3873a2575dfa3356d94cc40cc5dc7d7b71`，105tracked文件SHA全部一致，服务器Git工作区干净、inputs/outputs/runtime/凭据/marker忽略规则有效。发现一份旧TSV本机CRLF与仓库LF不同，按仓库字节统一后通过；Windows批处理按已有Git属性保留CRLF。服务器旧Git不支持init --initial-branch，改用标准init+设置main索引后实际成功。所有失败都在严格SHA/命令前置检查处停止，完整数据与六状态未覆盖。后续只补发布状态记录，最终三处HEAD记录在同步收据，学习引擎/profile/data/3B环境无改动。
