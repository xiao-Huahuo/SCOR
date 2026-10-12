# v01 正式 baseline 冻结记录

冻结在三次正式训练启动前完成；本记录于2026-10-12依据服务器原始收据整理。权威文件是 `solutions/v01/outputs/baseline_calibration_11/processed/baseline_freeze_20261011.json`，SHA256为 `7208b1b1821ff3eb1b5fd238de4abda016f3770f964f56a885ec5dd510e11d12`。正式实验不得按held-out结果修改条件。

## 固定条件

| 条件 | 生效值 |
|---|---|
| 模型 | Qwen2.5-VL-3B-Instruct，revision66285546d2b821cf421d4f5eb2576359d3770cd3；自有14文件copy，逐SHA与只读源一致 |
| 数据 | visual_baseline_08；三位整数图像；train1024/cal64/probe128/eval256，scene隔离；固定Verifier候选cal128/probe256 |
| 原生代码 | Agent0 commit f775b5101e62fe92976831adf4a21a38fcc0a767；独立baseline适配；12个引擎文件的完整SHA见下表 |
| 拓扑 | 物理GPU4–7，FSDP4、rollout TP2；分支串行、全参数更新、固定reference |
| 视觉 | native slow processor；min_pixels3136/max_pixels32768；实际grid[1,12,12]、pixel_values(144,1176)、36 merged tokens；Torch/vLLM typed bytes相同 |
| 推理 | 最多4轮，每次256token、response buffer1536、context6144；实际Python工具、共享参数Verifier、自然Repair及再生成 |
| 更新 | 原生mixed reward alpha_out1、GRPO/sample std/epsilon1e-6、AdamW；batch4×n4、micro1、一次global optimizer step；cosine/fullT32/warmup0/初始LR5e-7 |
| 正常训练D | 预列seed20261011、20261012、20261013，各32步；greedy evaluation在0/8/16/24/32，各256独立case；不据结果追增seed |
| 快照 | 每8步及保护/末步提交完整状态；首seed保护S0/S7/S8，另两seed保护S0/S7；保留滚动终态，只有新完整证书提交后才退役本run非保护旧状态 |
| 配对E | D之后复用首seed真实native U8/S8为正确分支；错误分支恢复相同完整S7、重放同一U8；第一stable ID的显式错误/native acc0/非截断轨迹，唯一terminal outcome+1，全组重算adv，再做一次原生更新；没有eligible如实记录，不重新挑未来结果 |
| 停止 | 资源不足、非有限状态、输入/身份/完整group错误立即停止并保留现场；连续3个zero-advantage单元停止；所有attempt新run_id |

校准仅用于calibration与工程选择：初始每题型准确率10%–90%、Solver/Verifier解析≥90%、截断≤5%、至少20%group非零adv、有真实工具/自然Repair及有限真实参数变化。cal11的14项checks全部通过。候选2B→3B、任务三位整数、slow像素预算等变化均在校准期记录，不能描述为2026-10-10已预注册的条件。

## 身份与实际证据

- profile_sha256：`35c489d89d16767283fbd09bb9f3d0130b11f27aadcf710ba05011a06f18f655`。
- model_signature：`8c77666273a27a297fce710738349d3f0fb3ebbf8abfc1e361501c9dfacc0710`。
- dataset_signature：`8b38d437b78fe97caa74a47ba3584eebcd0c89e3b55d5ea9a09f1963d46cb8a1`。
- environment_signature：`d594f4e67d235c7490079e4decfc6e7bac90ff2ed8f960a6663ef40891b97e76`。
- calibration_learning_signature：`aceb5549ad018d19b3d7482f9ff58a64341c08171919fffb62f799f25a46d4b0`。

各正常seed的learning identity包含seed，不能把校准learning signature当所有run同值；正式launcher按freeze核对源/profile/model/data/ENV及允许的seeds。协议与父状态身份另行记录。

| 冻结证据 | SHA256 |
|---|---|
| actual_state_restore：实际通过 | `7239ae2cee86ed9a1bdac10aea29a3063c231d62658288b0e302b112dc677d17` |
| checkpoint_storage_budget：实际通过 | `570a80bdc23eaf364f0275ef84df332bc256269bfc08fecea8cf01017a3a0123` |
| evaluation_rng_isolation：实际通过 | `2e23cb3e4218a78ec1623e123a8e4c8cba274047eb586fd7dff2bc1bf56cf7eb` |
| independent_code_review：实际通过 | `3f7061fbb6f159752ad8b6acc932e299ccd83c41e8bc087c16fd07c62425f9de` |
| native_loop_health：实际通过 | `d39cd8818de68d10da493c14fccf20a59d566f1e8c001558f1ce276251260e92` |
| same_unit_replay：实际通过 | `4e60a25d5be00217fb5584ce536b0104366da6e5695d0eccb540a51ffb18f72d` |

健康、完整S0/S7恢复、同U1/U8重放及评测RNG隔离都来自已完成的真实四卡对照。完整model/persistent buffers/Adam/groups/scheduler、worker RNG和driver/data cursor，以及全部13训练tensor/视觉metadata精确一致。通用sampler逐位恢复仍未验证。代码独立深审及9配对/3成本反例补充审查无未解P0/P1；补充review_03引用当前源码，CPU04实际12passed/3.17秒。

| 训练引擎文件 | SHA256 |
|---|---|
| apply_baseline_instrumentation.py | `264348cb5893626a8e5809ea375cda5a8377858ce73df4f0078ae31d961753c2` |
| apply_instrumentation.py | `e2e9e11521f564babc9af18ffdd5680fcb9b768bad78fa6933f1aae7c05c41ed` |
| baseline_acceptance.py | `7579b43552ce2373c726363ae6991b75a3f81ff036460957191e7c1fee1d326a` |
| checkpoint_store.py | `e0735ab1340f3e3f6df9cbc368c09405b5d480d3e59d1c848c1f865c113ee89b` |
| evaluate.py | `aaeac089ec88c34996a479fca33945dbf5575e1f3a783316e7fac07662e05346` |
| interventions.py | `388ac1bcc532780485935990114432ed38ea217fd26956868c8eac23e3308259` |
| native_adapter.py | `213c4fa1a6e7a6dacb2f556d3628a8777737532929613fb35131ff2549b520fa` |
| pilot_utils.py | `06cd6ca7ecf987f5966ed01c7cc1e7a68be4ca44b941be19de42220741c56120` |
| probe.py | `c9b5ef20adc28cee0a3be1d748df0d7273a780a51d096b9606e5bddeda328772` |
| records.py | `16540d8a2743e986b42011072160f762c704ef6fdf379d0a90842870ea02f733` |
| replay.py | `d41252b7201ea84b48dd00f2798655ffb9bf0667a9280d69c4a819ab187fc9c8` |
| run_baseline.py | `e457eb1e3830071fc3aa0c7ddc255f3a38708564c50b29f26a9f5c41be611d2a` |

## 累计磁盘预算与恢复资产

全部既有模型、失败现场、校准、三个consistency分支及压缩资产均已包含在实测余盘中。预算使用实际完整S0/Adam状态大小，向上按MiB取整，并给每state16MiB元数据余量；另保留四个未来run各7GiB的raw与安全余量。冻结时实测余盘 **420.549374GiB**，累计未来写入峰值需求 **418.601562GiB**，预算后余量 **1.947811GiB**。这不是文件系统空间预留；每次启动/保存仍实查空间。

cal05 S1有本机完整lossless归档（38799427217bytes；SHA ac1e7373440898c378dd3b26eb34b474f708473ff9d846aed4767c7ac343831d），整包及17个member逐字节SHA全部通过，服务器复核后才退役展开副本。cal06 S1及cal05/06/08/09/10 S0另有六份**同一服务器的压缩恢复副本**，整包/所有member及源再次核验通过后才退役展开副本。cal06 S1 archive保留于`.runtime/checkpoint_archives/cal06_step1/state.tar.gz`；五份S0保留于`.runtime/checkpoint_archives/server_lossless_20261011/<run>_S0/state.tar.gz`，verification/retirement收据保持。

这六份服务器副本不能称独立异地备份；cal06本机下载仍只有96/145已验证整块，完整本机归档未完成。旧校准结果/raw/source和完整证书不变；恢复压缩state时须先验证archive、解压，并按原certificate重新核验全部成员。当前cal11及三个实际GPUcontrols未退役。

## 交付边界

此freeze只批准上述D/E，不批准六个科研Gate或SCOR模块效果主张。D必须实际完成三seed×32且原始收据/完整S32齐全；E必须完成真实配对GPU更新与逐字段审计。cal11固定Verifier全拒绝是校准阴性能力观察；mixed reward含GT outcome；native reanchor是固定历史干预，未声称on-policy或无偏。后续Gate1–6在v02–v07另做预算、冻结、审查与结果报告。
