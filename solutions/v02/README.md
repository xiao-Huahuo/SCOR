# v02 / Gate1

检验一次终局 false-positive 在完全相同的后续历史数据下是否持续到32步。[实施协议](../../docs/v02/gate1_protocol.md) 固定五组 seed、主指标、对照和停止规则。

当前正在实现与验证；尚无正式 Gate1 结果。`src/` 包含 v01 原生链路的版本内副本和 Gate1 审计/测量入口；旧版本保持原貌。`inputs/` 保存经核验的跨版本资产与固定探针，`outputs/<run_id>/` 保存 immutable raw、完整 checkpoints 和可重建 processed。

当前工程状态：646文件导入、新boxed固定输入、S7/U8实际clean状态等价和117项服务器原生检查已通过。队列04已完整复核45份clean证据，真实污染中断恢复与连续运行的S9完整状态/评分等价已通过，S0/S1、probe128/P256控制已正常结束并通过44份证据复核；预算和八项正式冻结证据待闭合。正式sequence/export的原生轨迹导入路径P1已在隔离候选版通过真实输入验证，待当前队列结束后部署。详见[实施与验收记录](../../docs/v02/implementation.md)。

入口为 `src/run_gate1.py`（单attempt）和 `src/run_gate1_sequence.py`（五组序列）；`export_gate1.py` 重建统计与辅助CSV，`export_gate1_costs.py` 重建全量费用，`plot_gate1.py` 使用独立绘图环境导出PNG/SVG。所有科研结果必须来自完整原始证据；当前合成排版测试不是正式结果。

服务器Python环境不全局安装patched `verl`。单独运行 `import_gate1_assets.py` 或 `gate1_probes.py` 的准备入口，须从项目根目录显式设置 `PYTHONPATH=solutions/v02/src:.runtime/upstream/Agent0_baseline_264348cb5893/Agent0-VL`；本次固定补丁身份来源于 `apply_baseline_instrumentation.py` 的SHA前12位。正式sequence/export候选入口已在父进程设置这一精确路径，实际收据见[独立审查](../../docs/v02/gate1_workflow_review.md)。

2026-10-13：因原累计空间实际不足，工作流保持clean→corrupt→repeat并将clean/source完整末态独立验收及去重提前。纯early proof不包含后续退役收据，paired/export重建时严格核对early与clean complete，最终只退役repeat。成功终态清理中断从原proof/plan续做；clean续做前再次认证保留source S32，plan/complete须完全相等并绑定本sequence/seed/commonS7。ENGINE15未变；native09全部133检查通过，新增16个生命周期/guard病例明确是小状态夹具，不是正式科研结果。原native08夹具导入别名失败收据保留。新累计13槽预算保留48GiB及原raw/eval额度并另核验各原launcher guard，实际budget/新codeReview/正式freeze仍待验收。

2026-10-13最新验收：新05独立代码审查（611a596d）与存储证据（baa5a621）已实际通过，八项证据齐备；预算实际free677054451712bytes高于严格累计需求666741017387bytes。CPU正在完整重核所有唯一证据SHA（包括探针及24大分片），随后写入新冻结文件并进行独立实际复核；正式五组T32尚未启动。冻结预审还闭合了历史P0混配漏洞：三组映射严格绑定原草案27d811，引用项和导入SHA同源，重签反例拒绝。早期草案由本机原字节补齐部署，不伪装成既有服务器freeze或用户预注册。

2026-10-13正式启动：冻结文件b36195b5129d8fe33fb26d808434bfd7db8e9b8e0de21f69c139af19fd164d3a经实际1021份唯一证据全SHA与validator通过，独立实际复核无P0/P1（独审另核995small，26large引用主代理全核验）。fresh GPU4–7/累计空间预检通过。五种子T32序列gate1_20261013_01的父进程2886355于2026-10-12T22:32:30.807780UTC启动，先完整核验导入资产，再串行GPU分支；启动不代表T32验收或科研结论。源码15+6锁定，失败保留现场并重新核查资源后才恢复。
