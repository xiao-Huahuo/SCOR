# v01 小规模复现审查记录

日期：2026-10-10（Asia/Shanghai）。依据实际工程证据审查，完整结果见 [pilot_report.md](pilot_report.md)。

## 用户要求与验收对应

| 要求 | 实现与验收 | 状态 |
|---|---|---|
| 在当前项目开发 | solutions/v01/src、版本说明、监管状态与变更记录 | 已完成 |
| 服务器使用四张4090 | 实际资源检查、隔离工作目录/venv、物理GPU4–7 | 已实际执行，最多四卡 |
| 在Agent0-VL基础上保留核心链路 | 官方视觉rollout、Solver/Verifier、真实工具、Repair、奖励、GRPO/FSDP | 原生短程运行与恢复续训通过 |
| 小规模复现 | 3B、每步16条轨迹、两次更新、完整step1 checkpoint | 工程适配完成；论文数值复现未完成 |
| 小规模验证 | 10项测试、真实batch reward对照、四题直接acc、实际参数变化 | 已执行；GPU错误分支因资源不足未启动 |
| 服务器规范纳入开发规范 | 科研开发规范.md 首要规范第9条 | 已加入 |

## 已确认的证据

- CUDA和BF16 FlashAttention实际kernel通过；依赖pip check通过。
- 服务器native_tests_05.log为10 passed。本机为6项通过、原生模块跳过，不把跳过当通过。
- step1原生update后写完完整model/optimizer/extra/dataloader/RNG checkpoint；随后指标报错的failed状态保留。
- native_resume2_01恢复同签名快照，继续使用样本4–7，完成第二次更新并正常退出；观察到11次工具成功、23个已验证步骤与5次Repair。
- 原生GRPO重算与真实updater输入一致。一条return减2改变同组四条advantage，使另一条未改reward的轨迹由负变正；其他组不变。
- 上游改动为入口hook、DP宽度与无critic指标条件，共三份源码；评分、GRPO loss与optimizer规则未改。
- 3B合成数据、ground-truth终局奖励和内循环Repair均明示；四题acc始终3/4，未宣称性能提升。

## 失败与修订

环境依赖、Ray路径、vLLM显存、多模态重复图像、跨rank宽度、记录器文件写入、探针索引与GRPO指标错误均保留独立现场，见报告失败表。配置随诊断修订，不把最终配置说成首次预注册配置。两项项目记录器错误已实际回归检查。

native_corrupt1_01发现GPU4/5属于其他任务，资源预检停止；GPU错误分支没有执行，未改为两卡不同配置冒充配对结果。

## 挖洞式审查与边界

- **P1，研究范围**：指令3B和合成视觉题不等于论文SFT、7B/8B规模或benchmark复现；六个Gate未执行。不能推导SCOR优越性或长期污染。
- **P1，对照缺口**：离线reward影响已证实，相同前置状态的GPU梯度与后续policy影响未对照，需要四张空闲卡。
- **P1，恢复范围**：完整checkpoint恢复与后续训练通过；vLLM采样逐位RNG恢复未验证，不承诺独立重采样得到相同轨迹。
- **P1，存储**：step1完整checkpoint约43.1GiB，余盘约32.8GiB。step2未新增checkpoint，最终模型未保存；全量运行前必须解决存储。
- **P2，指标单位**：不同图像的相同文本prompt可被上游mean@k合并，本版报告逐样本acc，仅4个验证单位。部分final_answers为空，未验收严格boxed输出稳定性。
- **P2，计算指标**：RTX4090不在原生FLOPs表内，MFU不可作效率证据；缓存/预填充适配不等于论文配置。
- **P2，收据位置**：本机有JSON/CSV/log与完成证明，完整tensor和checkpoint分片在服务器。本机证明不能代替恢复资产。

核心链路观察在声明范围内通过；上述缺口未写成已完成研究结论。
