# 痛点
在VLM强化学习训练中, 模型解题时看错图形、选错公式后，仍然能够生成连贯的解答, 纯文本评分程序(LLM 纯文本 Verifier)则能认可这类错误并反馈高分给训练过程.
论文集中处理的是：**怎样让模型检查自己的视觉推理过程，并产生更可靠的训练奖励。**
论文目标: **VLM在没有人工持续给reward的情况下,自己产生训练信号,为自己提供自进化(强化学习)**.
# 解决方案
### 内层
> 由Solver和Verifier两个Agent解决问题并产出Trajectory和Reward.
```
问题 -> Solver(进行Reasoning) -> 产出Trajectory -> Verifier(进行Verification) -> 产出Score/Confidence/Critique -> Repair -> Reward
```
##### Solver与Verifier
Agent0-VL引入"自回归模型检查",同一个VLM同时承担:
- Solver: 生成视觉推理轨迹
- Verifier: 检查轨迹 + 产生 reward
	Verifier除了会对Solver给出的答案给出评价, 还会对Solver的"思考过程"做出评价: 检查每一步的中间推理,给出对应的 score + confidence + critique(评价), 这样就可以得到 Process reward(过程级奖励).
两者都是agent,共享同一套参数,使用不同的提示词切换.
两者均可以使用工具,Solver可以使用解题工具,**Verifier可以使用检查工具**(Tool-Grounded Verification).
##### Repair
Verifier检查给出的critique将通过Repair机制反馈给Solver,实现单次推理过程中多个自我纠错:
```
Reason 
↓ 
Verify 
↓ 
Repair 
↓ 
Reason Again
↓ 
...
↓ 
Reward
```
Repair保留已经验证正确的步骤,从被标记为错误的地方重新开始执行后续推理.
##### Trajectory
Solver产出Trajectory,其内容包括:
- 状态 $s_t$：当前图像、历史推理、之前的工具输入输出等；
- 动作 $a_t$：既可以是一段 reasoning，也可以是一次 tool call；
- 观察 $o_t$：工具执行后返回的结果。
Solver每思考一步,就添加一个Trajectory元组. 最终一条 trajectory 严格写成：

$$ \tau=\{(s_1,a_1,o_1),\ldots,(s_T,a_T,o_T)\} $$
##### Reward
Verifier产出的Reward计算如下：

$$ r_{\text{proc}}^{(t)} = \lambda_{\text{tool}}r(\text{tool}_t) + \text{score}_t\cdot \text{conf}_t - \beta_{\text{div}}D_{\mathrm{KL}} $$
其中考虑了三部分:
- Tool correctness: 工具执行有没有得到可靠、正确的 grounding。
- Semantic reliability: 评分Score与可信度Confidence的乘积.
- Distribution regularization: 作者的意思是避免 Solver / Verifier 两个角色在训练过程中分布漂得过远，维持所谓的 **dual-role distributional alignment**。

最终$r_t$计算式如下, 如果触发 Repair ,则中间reward会较低, 扣除掉Repair的成本, 以达到**更偏向未Repair的路径**的学习效果：

$$ r_t=r_{\text{proc}}^{(t)}-C_{\text{repair}} $$
最终给 GRPO 的总 reward：

$$ g(\tau) = \alpha_{\text{out}}r_{\text{out}} + \sum_t \gamma^{t-1} r_t $$
### 外层
> 收集内层的Trajectory和Reward,通过[GRPO](papers/06_DeepSeekMath.pdf)机制更新VLM参数,形成自进化(Self-Evolving Reasoning Cycle，SERC).
```
Trajectory/Rewards -> GRPO -> 更新VLM参数
```
更新后的模型继续执行下一轮解题和验证:
$$  
\boxed{  
\text{Reason}  
\rightarrow  
\text{Verify}  
\rightarrow  
\text{Repair}  
\rightarrow  
\text{Reward}  
\rightarrow  
\text{Update}  
\rightarrow  
\text{Reason Again}  
}  
$$
外层收集多次内循环的Trajectory和Reward,更新VLM参数的过程:
```
当前模型 M₀（参数 θ₀）
        │
        ├── Task 1 → Solver → Verifier → trajectory + reward
        ├── Task 2 → Solver → Verifier → trajectory + reward
        ├── Task 3 → Solver → Verifier → trajectory + reward
        ├── ...
        └── Task N → Solver → Verifier → trajectory + reward
                         ↑
                 全部使用同一个 θ₀
                         │
                         ▼
                收集一批训练数据
                         │
                         ▼
                       GRPO
                         │
                    更新参数
                         ▼
                   得到 θ₁ / M₁
```
# 实验
1. Tool-Grounded Verification 有效
2. Self-Evolution 能持续提高模型能力
3. Verifier 本身具有一定泛化能力
