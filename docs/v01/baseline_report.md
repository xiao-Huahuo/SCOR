# v01 baseline 实测报告

更新：2026-10-12。六项冻结证据和累计预算已通过，正式D三seed各32步、五次256case评测和认证完整S32均正常完成，正常baseline已验收。E同S7/U8实际GPU配对与独立审计也通过，A–E均已验收。最终CSV已导出并逐SHA同步；源码与报告已发布，实施提交为`5573536d17c741612f8a13c3d1a29d8523477cdc`，远端SHA已核对。[冻结记录](baseline_freeze.md)固定训练条件与判定；下面正式held-out结果只用于报告，不用于继续选配置。

## 正式正常训练D

| seed | 初始0 | 步8 | 步16 | 步24 | 步32 | 相对初始 |
|---|---:|---:|---:|---:|---:|---:|
| 20261011 | 179/256 | 181/256 | 177/256 | 179/256 | 178/256 | -0.3906个百分点 |
| 20261012 | 179/256 | 175/256 | 178/256 | 179/256 | 181/256 | +0.7812个百分点 |
| 20261013 | 179/256 | 181/256 | 185/256 | 184/256 | 185/256 | +2.3438个百分点 |

每个终态均对应32个不可改写unit完成标记、实际optimizer位置1–32、used LR>0、五次完整256case评测和认证完整S32，三个进程均return0。终局格式解析分别94.5313%/94.9219%/95.7031%，截断0/256、1/256、0/256；显式答案与native fallback分列。

三个终局准确率均值70.8333%，共同初始为69.9219%；每seed终局减初始分别−0.3906、+0.7813、+2.3438个百分点，均值+0.9115个百分点，范围[−0.3906,+2.3438]。三次训练重复仅作描述性统计，不宣称显著或可泛化提升；同组轨迹及重复评测的同一case不能当独立seed。

三个报告都在`baseline_formal_20261011_seed_<seed>/processed/normal_baseline_report.json`。各run另有`processed/updates.csv`及五个`processed/evaluation_step_<step>/cases.csv`，共15份逐case表/3840条评测记录（同256case重复评测），原始JSON/typed tensor、角色/工具、固定Verifier诊断与完整训练状态保留在同run的raw/checkpoints。本机small receipts已逐SHA核验同步；完整model/Adam和大型tensor仍在服务器，本机cert copy不能称本机完整checkpoint。


## 已完成正常run的成本

| seed | run wall秒 | 四卡分配GPU小时 | 原生step秒 | checkpoint秒 | evaluation秒 | snapshot写入bytes |
|---|---:|---:|---:|---:|---:|---:|
| 20261011 | 13347.905 | 14.8310 | 11275.482 | 3474.736 | 5591.586 | 247776716232 |
| 20261012 | 13169.148 | 14.6324 | 10945.207 | 3746.792 | 5253.088 | 247776717000 |
| 20261013 | 12673.425 | 14.0816 | 10528.831 | 3467.199 | 5142.820 | 247776717384 |

三normal设备采样峰值分别24213/24215/24213MiB，E为24117MiB；每2秒设备总量采样的下界，不能把它当瞬时精确峰值。

E wall1970.108秒/选择四卡2.1890GPU小时，一次native unit1168.661秒、checkpoint686.619秒、两次calibration评测807.107秒，snapshot写入46302326480bytes。D合计43.5450分配GPU小时，加E合计45.7340；这些是运行账单，不含此前校准/三个consistency controls、CPU分析、归档和Git操作。

这些组件有重叠，不能相加为wall；GPU小时按选择四卡×run wall计算，不是利用率。wall由服务器manifest UTC至终态status原mtime，包含原生启动、评测、IO/恢复，不含此前preflight/模型hash。native角色成本是per-worker记录，TP2含重复请求；报告额外给出全部request/response与rank齐全核验后去重的logical calls/requests/tokens。物理GPU CSV每2秒采样，峰值是采样下界，设备总使用量可能包含其它进程；PyTorch睡眠pool 30–35GiB计数不能当4090物理峰值。

## 同S7/U8的实际配对E

E正常intentional stop8/return0，一次实际optimizer step8；使用与正常U8相同的LR4.432526133406842e-7，grad norm2.084620714187622，完整S8提交。实际恢复的父state为`ddcfa9fe1fe944ac77e04ceb746efc63a49f059c154415a1f0006a8f711d3bf4`；model/Adam/groups/scheduler/worker RNG完整指纹、driver RNG和data cursor全部匹配，learning identity与首正常seed一致。正确分支复用其已完成的native U8/S8。

目标是第一stable ID的显式语义错误/native acc0/非截断轨迹：row5、Solver答案956、checker -158。实际唯一奖励改动在`[row5, token260]`，terminal outcome0→1，整条trajectory总return1→2；其它过程奖励保留。固定的九个训练tensor（token、mask、position、old_log_probs和ref_log_prob）typed bytes及全部non-tensor视觉/组/角色metadata相同；另外四个reward/advantage/return tensor只按声明修订并独立审计。

| 目标组row | 正常return | 错误分支return | 正常归一化advantage | 错误分支归一化advantage |
|---|---:|---:|---:|---:|
| 4 | 1.000000 | 1.000000 | -0.654241 | -0.982764 |
| 5 | 1.000000 | 2.000000 | -0.654241 | 0.041228 |
| 6 | 3.290000 | 3.290000 | 1.456696 | 1.362178 |
| 7 | 1.548951 | 1.548951 | -0.148214 | -0.420642 |

Python sample-std/epsilon1e-6独立oracle与实际全目标组adv/returns匹配（E前确定atol1e-6/rtol1e-5）；其余三个prompt groups的四类mutable tensor字节保持相同。错误目标的adv由负转正，组内其它三条也随归一化变化。更新后四rank模型指纹均与正常S8不同，两个分支参数/optimizer均有限。这验证一次声明的错误奖励确实进入原生更新；没有执行后续24步污染传播或能力损害Gate。

正常/错误S8的state signatures分别为`57a71716a580de78507e1d59c30795a8f107d6b73414e372a90e8f2d636387f2`、`7de8b5bb8a1409d77241f57663889cb9a049d3785f5b9d0a5123cd617a989592`。原始两个U8 tensor的SHA和完整审计在[paired_reward_audit.json](../../solutions/v01/outputs/baseline_formal_20261011_E_delta1/processed/paired_reward_audit.json)。没有重算已完成的正确U8，也没有按未来结果重选目标。

E的64题calibration诊断50/64→53/64，显式格式57/64→60/64，截断均0；初始格式57/64低于候选初始筛选参考.9。这是冻结后的单分支诊断，与正确S8的256题held-out评测不同，不能据此做干预能力比较或调配置；E验收是同状态/同unit/单分量/实际更新/完整状态，不是新一轮候选筛选。

| 冻结验收 | 实际结果 |
|---|---|
| 原生健康 | cal11完整8updates/正常stop8/rc0，14checks全True；201真实工具、22自然Repair、32/32非零groups，训练截断6/128 |
| 完整恢复 | 实际S0两次、非零Adam S7一次；参数/persistent buffers/Adam/groups/scheduler/worker RNG、driver/data cursor精确相同 |
| 原生/同unit历史重放 | S0/U1和S7/U8全部13typed训练tensor及visual/nonTensor输入、完整updated state精确一致 |
| 评测RNG隔离/无操作 | 两次initial评测、delta0之后未来native U1和全S1与cal11精确相同 |
| 累计预算 | six-witness freeze前余盘420.549374GiB、未来写入峰值/raw/安全418.601563GiB；按实测完整状态与每state16MiB、四run各7GiB余量；运行中空间guard继续 |
| 独立深审 | 无未解P0/P1；core CPU21实际82passed/1GPU诊断skip14.15s；独立entropy GPU1passed6.88s；后处理CPU04的9配对+3成本反例12passed3.17s、review_03实际hash匹配 |

最终模型为固定官方3B revision66285546d2b821cf421d4f5eb2576359d3770cd3；自有14文件逐SHA与只读原基座一致，本次未冒称新的HF LFS fetch验证。data08 signature8b38d437b78fe97caa74a47ba3584eebcd0c89e3b55d5ea9a09f1963d46cb8a1。slow processor min3136/max32768，实际grid[1,12,12]、144patch/36merged；四轮×256/总buffer1536/context6144、batch4×n4/micro1、full32 cosine/warmup0/lr5e-7、FSDP4/TP2、原生mixed reward alpha_out1。12个engine/profile/model/data/ENV与freeze一致，原15pilot文件未改。

六份不再依赖的旧state已在同服务器完整lossless压缩、整包/全部member及源再次核验后退役展开副本；压缩恢复资产继续保留。cal05 S1另有本机整包和17member完整验证备份。cal06本机下载仍只有96/145已验证整块，完整异地备份未完成。同服务器副本不能称独立异地备份，恢复时需解压并按原cert重核全部成员。

解释边界：cal11固定Verifier正确/错误候选接受率均0，是校准能力阴性观察；正式probe256的初始正确接受2/128、错误接受0/128，三seed终局正确接受3/128、1/128、0/128，错误接受均0/128。高JSON解析率不能当有效判断能力；mixed reward含GT outcome；native reanchor是固定历史干预，未声称on-policy/无偏。特定U1/U8精确控制不能推广为任意sampler逐位恢复。人工terminal误判不直接代表自然Verifier错误。六Gate及HED/CSS/TRA/RR尚未执行。

## 可复核产物与未完成边界

序列终态[summary.json](../../solutions/v01/outputs/baseline_formal_20261011/processed/summary.json)的SHA为`68cc81338fafde7718ae00e1730fd27ae64fc805ec93129eceac94e925224055`，明确D_complete/E_passed=True、six_gates_passed=False。服务器按原status时间戳运行`export_baseline_summary.py`，完整CLI导出成功；本机文件逐SHA校验。

- [results.csv](../../solutions/v01/outputs/baseline_formal_20261011/processed/results.csv)：三seed×五时点，15行；各run另有逐case CSV和32行updates.csv。
- [costs.csv](../../solutions/v01/outputs/baseline_formal_20261011/processed/costs.csv)：3D+1E，4行，含真实wall/分配GPU小时/组件/写入bytes/物理采样峰值。
- [role_costs.csv](../../solutions/v01/outputs/baseline_formal_20261011/processed/role_costs.csv)：32个run/phase/role行，per-worker与完整TP请求核验后去重的logical calls/requests/tokens分列。
- [paired_advantages.csv](../../solutions/v01/outputs/baseline_formal_20261011/processed/paired_advantages.csv)：正常/错误两branch×16trajectory，32行，含目标组标记及独立oracle数值。
- [result_summary.json](../../solutions/v01/outputs/baseline_formal_20261011/processed/result_summary.json)：三seed描述性均值/范围、输入device CSV SHA、导出源SHA与成本解释范围。

所有实验原始响应/tensor/source/profile和保护/终态仍在服务器SCOR目录；本机完整模型/optimizer只对明确归档资产成立，small cert复制不冒充完整checkpoint。产品UI不适用于用户已豁免的CLI/GPU任务；真实训练、恢复、无操作、重放、配对更新及最终导出CLI冒烟已完成。

v01 A–E没有未完成的必需验收项。后续v02–v07六Gate、HED/CSS/TRA/RR、自然Verifier误判与通用sampler逐位恢复/采样概率等价均未实现或验证，不能称SCOR有效。旧cal06完整本机异地备份仍只有96/145已验证整块，完整同服务器lossless恢复副本已验证并保留；这是存储范围限制，不能称异地备份完成。

## 原始校准与实现修订时序（历史事实，下面的“当前/最新/待执行”只对应当时）


更新日期：2026-10-11（Asia/Shanghai）。独立入口和完整记录/状态/评测/重放接口已实现，当前进行3B候选的真实四卡校准。正式baseline未冻结，三seed×32步尚未完成，六Gate与SCOR效果未验收。

| 运行 | 真实结果 | 判定 |
|---|---|---|
| baseline_calibration_01 | 2B/constant，正常stop2；完整S0/S1/S2；initial64显式答案0、native fallback7/64；首轮实际LR0 | 健康拒绝，保留工程链路证据 |
| baseline_calibration_02 | 2B/cosine；完整S0，初始评测中发现额外Qwen包装重复原生chat template | 自主中止，原始日志与旧engine保留 |
| baseline_calibration_03 | 移除重复包装；完整S0，成功工具后仍重复程序；未完成完整64题评测 | 自主中止，不编造准确率/格式率 |
| baseline_calibration_04 | 2B/强Tool后终局提示；initial64显式答案0/64、准确率0、fallback21/64；Verifier128解析39/128，正确/错误接受率均6/64；真实更新1–3，step4反向传播OOM；完整S0/S1 | 模型协议健康与容量拒绝，不算正式baseline |
| baseline_calibration_05 | 正常stop1/return0；完整S0/S1；initial64准确49/64、显式50/64；final准确40/64、显式41/64；Verifier解析61/128→59/128；真实Tool17、自然Repair4、正LR更新 | 格式/Verifier健康拒绝；不隐瞒更新后下降 |
| baseline_calibration_06 | 正常stop1；完整S0/S1；initial/final准确54/64，显式61/64→62/64，Verifier127/128→128/128；正确/错误候选接受都0/64；Tool13/自然Repair1/正LR有限真更新 | 仅初始求和20/22和差值19/21超过.9 headroom而拒绝 |
| baseline_calibration_07 | 原生启动和部分S0写盘，独立CPU懒游标恢复暴露driver RNG顺序P1；无完整证书/评测/单元 | 自主中止，保留全部现场；新engine另起cal08 |
| baseline_calibration_08 | initial64准确56/64、显式62/64、Verifier127/128；sum22/22/diff20/21；S0完整，S1部分保留 | headroom拒绝，自主中止 |
| baseline_calibration_09 | data07、max_pixels32768、统一slow processor、显式native vision deterministic，fullT32/stop8/pin1,pin7；已启动 | 等待完整GPU健康、状态恢复与同unit验收 |

cal04前三轮used LR为5e-7、4.98796181668e-7、4.95196320101e-7，grad norm分别2.0772、2.2393、2.0205，参数探针改变且有限。step4申请2.01GiB时物理仅剩1.42GiB，失败现场保留；S1以后已算但未持久化模型的区间不能称完整恢复点。PyTorch睡眠pool峰值计数30–35GiB超过4090物理容量，不能用该计数宣称真实显存峰值，后续采样选定GPU物理内存。

2B裸数字/自然句可能正确，但不能降终局解析门槛或把工具输出代替Solver答案。模型变更依据用户已授权的模型/配置选择和独立校准证据，未访问正式held-out。3B固定revision为66285546d2b821cf421d4f5eb2576359d3770cd3；从既有共享原始基座只读复制到本项目.runtime/models，14文件、7520919614bytes逐文件SHA一致，原件不改。本次证明与复制源一致，没有冒称重新核验HF LFS。旧训练后的3B快照仍退役，不用它恢复新baseline。

3B候选保留全参数共享Solver/Verifier、Python工具、自然Repair、原生混合reward alpha_out1、原生GRPO/AdamW。使用原生cosine、warmup0、完整32步调度，每轮4题×4轨迹/一个global optimizer step、micro1、四卡统一拓扑。候选4轮推理、单次256token、总buffer1536/context6144；原生torch.compile entropy启用。原生compiled entropy在GPU4与独立float64解析数值和导数对照1passed6.88s，证明数学一致，不代替完整训练容量验收。

图像processor预算min3136/max65536通过原hf_processor/AutoProcessor和vLLM mm_processor_kwargs统一传入，初始化记录expected/actual并拒绝None/参数失效。512×512图像的真实fast处理输出为pixel_values(324,1176)、grid[1,18,18]，对应252×252/81合并vision token；不是臆造不存在的RLHFDataset max_pixels接口。输入原图、checker和scene定义不改，分辨率条件在学习身份中固定。物理GPU采样每2秒，记录峰值是采样下界；所有对照必须相同预算。

四split按scene隔离：train1024/cal64/probe128/evaluation256；额外cal_verifier128/probe_verifier256为固定正确/错误候选配对，expected_correct只在extra_info。data05签名fcfdc236b1df56b06b9c64ff96ff09f70ae4e3c159305c30c80500ac7b18d705。正式held-out只在冻结后运行，不能据其选择候选。

独立实现审查已修复非整数/空boxed解析、缺worker指纹证书、重复原生chat模板、空/失败/陈旧/异配置freeze、视觉history token/mask/typed tensor/组成员或必要字段缺失。角色mask从语义角色及真实token位置重建，不能只比较镜像实现；源unit完成收据、group SHA、全部payloadHash与原生generation schema共同校验。CPU15实际71passed/1GPU诊断skip12.50s；本机native模块因无Torch跳过，未计成功。

cal01状态字节实测S0=9771147607，S1/S2各27443238391（不含certificate），约9.10/25.56GiB；3B完整S0为16265099131bytes（15.148GiB）、含Adam moments的S1为46302323603bytes（43.122GiB），cal06实测保存分别229.935s/803.349s。cal01 S0本机lossless归档已全member SHA验证且server副本退役，S1/S2传输中断，完整server原件保留；cal02/03 S0、cal04 S0/S1均保留。旧3B退休记录和当时的两次update事实见pilot_report，首轮constant LR0不能写成两次权重变化。

冻结要求原生健康、完整状态实际恢复、同状态同unit重放、评测RNG/数据游标隔离、实测完整状态预算、独立代码深审六项同身份实际证据。健康规则采用设计建议：初始各题型10%–90%、显式/Verifier解析至少90%、训练/评测截断至多5%、全校准非零group至少20%、自然完整Repair与真实有限参数更新；不因当前失败降格。后续完整S0/S7恢复和U1/U8等价、三seed32步及E单事件GPU对照尚未完成，不以CPU/小GPU检查宣称完成。

成本报告区分native timer、保存/评测、重叠worker role计时和选定GPU wall计费。wall以服务器原manifest UTC到结束status mtime计算，包含native启动/恢复/评测/IO但不含此前preflight/model hash；物理GPU轨迹用于资源预算。当前任务是CLI/GPU工程，产品界面冒烟不适用；实际CLI与原生闭环、恢复验收仍须执行。

2026-10-11 cal07校准决断：cal06 initial64格式61/64、Verifier解析127/128，但求和20/22与range19/21超过事先采用的.9上界。下一候选统一使用三值之和、三值乘积、最大值减去其余两值之和；图像、原生训练算法和验收阈值保持，所有split问题/checker/收据一起更新并新建不可改写dataset07。只用calibration选择，未查看正式held-out结果。固定Verifier全部不接受是负面能力观察，不能由解析率高推断判断准确。

cal06完整正常stop1/return0，health重建仅task_headroom失败，其余13规则通过。final64准确54/64、显式62/64、Verifier128/128；Tool13/自然Repair1、非零group4/4、训练/评测截断0、usedLR5e-7、grad2.471236、全参数4shards变化且有限；物理采样最大22479MiB。完整wall2086.555s/四卡2.3184GPU小时，native unit119.357s、保存1033.284s、评测722.863s，重叠项不相加。

cal07启动data07后，真实torchdata懒游标恢复probe发现state_dict会物化iterator并消耗driver随机流；原load先restore RNG再检查游标，存在P1。已只中止本run标签进程，raw/source/16265099208bytes部分状态保留，无完整证书/评测。首次中止脚本误用旧系统Python，在任何signal/写入前因Path API报错；改隔离3.11后实际中止，事实存abort收据。

修订先物化并核对cursor，再最后restore driver RNG；实际S0/S7 CPU loader的下一批0–3/28–31与下一次随机数通过。CPU18完整78passed/1GPUskip11.68s；重放实际counter仅允许num_steps/num_repairs的同值int32→int64原生转换，改值/浮点、mask/图像/slot仍拒绝。持久buffer指纹/NaN检查、driver/data实际hash、acceptance源身份加入；delta0与repeat-initial-evaluation仅consistency，用于设计第9节无操作对照。

cal08新engine新run于2026-10-11 07:55:06UTC启动，PID657013/资源657014，data07 signature23986c40e196e274457ed02376b82cf5ce3557c7a245e53e099cf8bf27bf32f2，fullT32/stop8/pin1,pin7，预检734.57GiB和GPU4–7空闲。正式未验收，实际S0/S7恢复/U1/U8等价、重复评测下一rollout、六证据冻结、3seed32/E仍待完成。

cal08初始真实64：准确56/64、显式62/64、Verifier127/128、无截断；sum22/22和差值20/21超过.9，仍健康拒绝，已仅中止本run进程。保留完整S0和S1部分写盘（共45172999971bytes），无S1完成证书，不把部分写入称完整恢复点。

cal09候选在相同data07上将native processor max_pixels65536→32768，min3136不变。实际grid[1,12,12]/pixel(144,1176)/36视觉token。独立检查揭示原HF driver fast与vLLM固定slow在同网格下pixel最大差.015008，vLLM传use_fast参数仍未替换显式提供的slow image processor；现在Torch driver/actor/ref都显式use_fast=False，与vLLM原生slow完全复用，实际pixel与grid类型/形状/字节全一致，maxdiff0（processor_probe_02）。

HF原视觉flash_varlen调用不经过环境deterministic flag，现明确给同一原生函数传deterministic=True，没替换模型/梯度公式。独立GPU4 BF16(144,16,80)原函数前向一致、两次确定性反向逐位一致，且前向/梯度与独立float64密集attention参照一致；rtol.02/atol.005是读取输出前选定的BF16小kernel诊断，完整更新仍要求typed exact，没有把该诊断当baseline验收。

无操作独立反例表明torch.equal接受bool/int同值，且加0会改变negative-zero符号bit；现在paired compare以dtype/shape/bytes校验，delta0直接保持clone字节。完整pip/Python hash纳入learning和formal freeze，避免同代码异依赖恢复。CPU20真实81passed/1GPUskip12.77s，local49passed/3native skip0.63s，32text source/tests/profile逐SHA相同。cal09新run fullT32/stop8/pin1,pin7已启动；所有旧raw/source与失败完整保留，实际健康/恢复/重放仍待GPU。

D到E资产控制：首个normal seed保护S8，直接复用已完成的正常native U8作为正确reward GPU分支，错误分支从同S7重放这一个U8，只改目标terminal并执行一次native update；校验原始batch所有其它字段/组不变。不重算已完成的正确unit，预算不再重复计一份等价clean更新状态。完整累计存储见证必须在freeze前按实际余量通过。

cal09最新：initial64正确57/64、显式61/64、Verifier127/128、截断1/64，仍sum22/22和差值19/21超过headroom上界。真实完成更新1–3，S0/S1完整；step1grad2.441516/usedLR5e-7/参数探针真变有限。已于09:21:36UTC只中止本run，全部raw/source/fullS0S1保留，不能把未保存模型区间当可恢复state。

cal10候选数据独立scene为三位整数100–999，512×512图、字号24、bar高度floor(.28*value)保证柱顶/标签在画布内，三值和/三值乘积/最大减其它二值和公式不变。Legacy15 source未改；baseline自己的render/checker/收据一致，新immutable visual_baseline_08 signature8b38d437b78fe97caa74a47ba3584eebcd0c89e3b55d5ea9a09f1963d46cb8a1，正式held-out效果未读。统一slow processor预算min3136/max49152（196px/49vision tokens待实际probe）；Tool成功但No output时要求真实print(result)，不把裸Python表达式当打印结果；终局用无逗号的十进制整数，防大数格式失败。CPU21实际82passed/1skip14.15s，本机50passed/3skip0.69s。

服务器剩634.063GiB；累计正式/consistency/恢复资产预算必须重新核算，先为不再依赖的旧校准快照做lossless完整backup并逐member SHA验证，验证后才能退役server副本。本机可用289GiB，旧partial archives不能假称complete。所有正式冻结、GPU恢复/replay和三个32步正常baseline/E仍待完成。

cal10完整initial64为56/64（.875），显式61/64（.953125），Verifier117/128（.9140625），截断0。family0 19/22与family2 18/21满足规则，family1 19/21仍超过.9上界；已自主仅停本run，certifiedS0/raw/source与partialS1保留，没有已完成unit。cal11同source、同data08、同seed/full32/四卡，只max_pixels49152→32768，所有门槛保持。backup cal05 S1的全source SHA已通过，durable gzip正在生成，尚未本机完整member验证，也未退役源。

cal11完整initial64：正确51/64（.796875）、显式58/64（.90625）、Verifier122/128（.953125）、截断0；family0 18/22（.81818）、family1 18/21（.85714）、family2 15/21（.71429）全部满足初始headroom/格式/解析规则。仍须真实8unit、自然Repair/非零group/有限实际更新、final格式/解析及完整S1/S7/S8；初始过线不等于六证据或正式baseline通过。learningaceb5549ad018d19b3d7482f9ff58a64341c08171919fffb62f799f25a46d4b0，当前源/profile/data冻结于这轮，继续监督。cal05S1 lossless backup压缩约29.1GB尚未完、本机未验证、未retire源。


2026-10-11 23时存储决断：cal06本机传输在96/145整块后多次网络超时，完整archive/member验收仍未完成。为继续正式任务，在获准SCOR服务器目录创建same-server lossless recovery archive：cal06 S1已有staged gzip保留；另串行完整压缩不再依赖的cal05/06/08/09/10初始S0。每份必须整源SHA、压缩包及全部member对应原证书核验、源再次核验，proof不可改写后才能退役展开副本。压缩资产始终保留，不称异地备份；本机传输继续。当前archive进程PID1202422，尚未完成，累计预算/freeze仍待实际核验。cal11及三个GPUcontrols不动。

成本核验补充：cal11原始wall5462.384秒/选择四卡6.0693GPU小时，native step2467.744秒、snapshot3279.147秒、eval792.790秒有重叠，不能相加。原native成本phase_role是per-worker记录，TP2含重复requests/tokens；新增baseline_costs.py在全部TP request/response一致且rank齐全后去重，额外评测call occurrence保持独立。实际cal11所有记录通过，12个配对/成本反例CPU通过2.94秒；报告同时给出worker总量与logical请求/生成token，均不等同于sampler概率或独立wall。

## 2026-10-12用户授权清理后的资产范围

本文件前述校准/controls/备份状态描述属于清理前的执行事实；实际原始证据与冻结JSON未改。用户明确清理后，仅保留三个正式S32、首seed S7/S8和E S8共六份完整训练状态；旧校准/对照/重复初始及中间状态大权重、失败半写、2B和旧压缩/本机备份与下载残片已退役，证书及small metadata保留。当前旧点不能完整恢复，cal06本机下载不再是待完成项。3B/环境/data08/所有raw与processed保留，结果与验收结论不变；详见[storage_cleanup.md](storage_cleanup.md)。新的校准/恢复复核若需要旧完整状态须另立新运行，不能用历史证书冒充现存模型。
