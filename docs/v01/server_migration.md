# 服务器目录归并

日期：2026-10-10（Asia/Shanghai）。用户要求将本项目服务器内容统一放入 `/data1/yangbin/SCOR/`，并写入科研开发规范。

## 范围与操作

已将带 `.scor_workspace` 项目标记的 `/data1/yangbin/SCOR_v01` 整体重命名为 `/data1/yangbin/SCOR`。迁移前没有本项目训练/Ray进程，目标目录原先不存在，源与目标位于同一文件系统。整个代码、隔离环境、输入、原始结果、完整快照和文档随目录迁入，原始文件内容不改写。

旧路径保留为指向 `SCOR` 的相对符号链接，用于已有环境脚本和历史收据；没有复制一份训练产物。共享账号中的现有模型和基础Python是只读借用资源，非本项目创建的资产，保留原位。新下载或创建的资产按规范存入统一目录。

## 实际验收

- 迁移前后77,159个条目的device、inode、大小、mtime、mode及链接目标一致，核对清单指纹为 `09c962008dd8cf54fcdde2d2d7382bb7f51c57783604bcb4b666f0f9c07db2e4`。
- 51个源码、配置、manifest、状态和完成证书的SHA-256一致；旧快照12个worker分片实际存在，训练签名仍为 `0a0fd00185a624189484f29955052fafb645c4671cfdbe2f15d0320b8048a720`。
- 新目录下Python的prefix实际为 `/data1/yangbin/SCOR/.runtime/gpu-venv`；`pip check`通过，torch2.6.0+cu124与vLLM0.8.2身份一致。
- 这是同文件系统目录迁移；大型tensor没有做两遍内容hash。其文件身份和元数据未变，不将该检查描述为新的GPU恢复测试。

收据：`solutions/v01/outputs/server_migration_20261010/raw/migration.json`。旧报告、配置和运行manifest保留当时的路径事实；今后代码从自身src推导根目录，当前监管记录使用新位置。

## 占用与仍待验证项

约52GiB的文件逻辑大小主要为：step1完整快照43.1GiB（模型分片15.1GiB、AdamW状态28.0GiB）、独立依赖环境约8.5GiB，另有源码、轮子、原始轨迹、日志和缓存。磁盘分配占用约52.2GiB。迁移前后可用空间都约32.75GiB，整理目录没有扩大存储。

迁移当时旧快照、原始记录和共享资源未删除。新baseline的容量、四卡空闲情况、实际保存/恢复/晚期分叉/重放仍需验收。旧pilot的签名包含绝对输入路径，迁移后的新运行不能仅凭路径链接宣称与旧签名相同；重用旧资产需按原manifest或明确的路径映射检查。

2026-10-11用户随后授权退役旧3B试跑的权重和优化器状态；释放43.1GiB，当前旧快照已不可恢复。迁移时的完整性核对仍为历史事实，后续清理见 [checkpoint_retirement.md](checkpoint_retirement.md)。
