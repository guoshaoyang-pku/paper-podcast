# RL Token — 论文重要图

> 配套 `card.json` / `card.md`。论文共 9 张 Figure,全部列出。每张图给出:所在页、原文 caption(精简)、我们写的"这张图到底在讲什么"。
> 页图存在 `extracted/rl-token-vla/pages/pXX.png`,可对照查看。

## Figure 1 — RLT 系统总览(page 1)

![Figure 1](../../extracted/rl-token-vla/pages/p01.png)

**原文 caption**:Our method introduces an "RL token" into the VLA by training an encoder and decoder to produce a compact and meaningful representation from a VLA's internal features. The extracted representation is then used to train lightweight actor-critic networks with sample-efficient online RL, enabling very precise tasks to be fine-tuned with a few hours or even minutes of robot experience.

**这张图讲什么**:门面图。左:冻结 VLA + RL token encoder/decoder(产生 RL token)。中:小 actor-critic + critic + value Q(s,a) 在 RL token 上做 online RL。右:四个真实任务(螺丝/扎带/Ethernet/充电器)。核心信息:RLT 把"十亿参数 VLA 的 RL 微调"拆成"冻结 VLA + 几层 MLP actor-critic",所以能几小时真实数据训完。是论文分工哲学的视觉表达——VLA 当感知+行为先验,小 actor-critic 做 local refinement。

## Figure 2 — RL Token Extraction(page 4)★ 最重要

![Figure 2](../../extracted/rl-token-vla/pages/p04.png)

**原文 caption**:Details on RL token extraction. RLT adds an encoder-decoder transformer to a pretrained VLA. It produces a compressed embedding of the VLA representation (the RL token). This representation then enables data and parameter efficient fine-tuning during online RL.

**这张图讲什么**:全文最该看的图。π0.6 内部:SigLIP(400M)+Gemma(4B)VLM backbone 处理图像 token embeddings(N×2048),860M action expert。RLT 在 VLA token 序列末尾 append 一个 e_rl,过 encoder transformer g_φ,取该位置输出作为 RL token(1×2048)。decoder d_φ 训练时从 RL token 自回归重建原 embeddings(Lro loss),逼 RL token 成瓶颈。这张图同时回答"RL token 怎么提取、为什么是瓶颈、和 VLA 怎么接口"。

## Figure 3 — 四个评测任务(page 6)

![Figure 3](../../extracted/rl-token-vla/pages/p06.png)

**原文 caption**:The tasks in our experiments: each task contains a critical phase that requires high precision: (top) using a screwdriver to install a screw, (middle) fastening a zip tie, (bottom) plugging in an Ethernet cable and plugging in a charger.

**这张图讲什么**:三个真实任务的可视化:螺丝安装(电动螺丝刀+M3 螺丝,亚毫米对齐)、扎带紧固(双臂协调穿可变形扎带)、Ethernet+充电器插入。每个任务都有 critical phase(插入/紧固/旋转段)需高精度,5~20s/250~1000 步。是 RLT 设计动机的视觉证据——VLA 在 critical phase 最容易慢/失败,正是 RL 该精修的地方。

## Figure 4 — Throughput 提升(主结果)(page 8)

![Figure 4](../../extracted/rl-token-vla/pages/p08.png)

**原文 caption**:RLT increases throughput significantly over the base VLA policy, improving both the speed and consistency of the critical phase of each task. The improvement is especially pronounced for the harder tasks where the VLA policy is prone to making mistakes.

**这张图讲什么**:四任务(screwdriver/zip tie/Ethernet/charger)× 两设置(critical-phase / full-task)的 throughput(successes/10min)柱状图。RLT 全面大幅超过 Base Policy。难点任务(screw/zip tie)提升尤其显著——这正是 VLA 容易失败的地方。关键论点:RLT 不仅提成功率,更提速度(throughput 同时反映两者)。

## Figure 5 — Success Rate 提升(page 8)

![Figure 5](../../extracted/rl-token-vla/pages/p08.png)

**原文 caption**:RLT can boost success rates across multiple tasks. Where the VLA is already competent (e.g., the Ethernet task) it maintains success rate and increases throughput. For tasks that are challenging for the base VLA policy (screwdriver and zip tie) RLT leads to a significant improvement in success.

**这张图讲什么**:四任务 × 两设置的 success rate 柱状图。Ethernet/charger 上 VLA 已不错,RLT 维持成功率先(主要提速);screw/zip tie 上 VLA 弱,RLT 大幅提成功——screw full-task 从 base 提 40%,zip tie 提 60%。这张图区分"RLT 在 VLA 已强任务上的角色(提速不损精度)"vs"在 VLA 弱任务上的角色(根本性修复)"。

## Figure 6 — vs 其它 RL 算法(Ethernet)(page 8)

![Figure 6](../../extracted/rl-token-vla/pages/p08.png)

**原文 caption**:Comparison to other RL algorithms. Methods that consider only single actions, rather than action chunks, (HIL-SERL, PLD) perform poorly. DSRL leads to high success but significantly lags behind in throughput.

**这张图讲什么**:Ethernet 任务上 RLT vs DAgger/HIL-SERL/PLD/DSRL 的 success rate + throughput。HIL-SERL/PLD(单步方法)学不出来——稀疏奖励下 horizon 太长 credit assignment 失败。DAgger/DSRL 成功率追平 RLT 但 throughput 明显低(DAgger 受限于人类示教速度,DSRL 强约束 VLA 改进空间小)。RLT 既高成功又高吞吐。是"chunking + RL token + 参考 anchor"三件套必要性的对照证据。

## Figure 7 — 消融:throughput vs 训练分钟(page 9)

![Figure 7](../../extracted/rl-token-vla/pages/p09.png)

**原文 caption**:Throughput at different points in training for Ethernet task. The ablation study shows that each part of our method is important for good performance. RLT outperforms the alternative policy after consuming only 5 minutes of data on the critical part of the task.

**这张图讲什么**:Ethernet throughput vs 训练分钟曲线,五条线(RLT/w-o Pass-Through/w-o RL Token/w-o Chunk/w-o BC Regularizer)+ Base Policy。全 RLT 5 分钟数据就超 Base Policy,15 分钟达最终最佳。w-o BC Regularizer(β=0)跌最狠——无 anchor 让 actor 在全动作空间裸搜。w-o RL Token(换 ResNet-10)throughput 降 50%——证明 VLA token 编码了 ResNet 没有的 manipulation-relevant 结构。w-o Chunk(单步)credit assignment 失败。w-o Pass-Through 最终能追平但学得慢+训练中失败多。

## Figure 8 — 消融:success rate vs 训练分钟(page 9)

![Figure 8](../../extracted/rl-token-vla/pages/p09.png)

**原文 caption**:Success rate evaluation during training for the Ethernet task. RLT quickly matches the success rate of the VLA policy on the Ethernet insertion task, while boosting throughput. Not using the reference-action pass-through or not using the RL token leads to slower learning.

**这张图讲什么**:Ethernet success rate vs 训练分钟曲线。RLT 快速追平 VLA 的成功率同时提吞吐。w-o Pass-Through / w-o RL Token 学习更慢。和 Figure 7 配合:Figure 7 讲 throughput,Figure 8 讲 success rate,共同证明每个组件都对"快速收敛 + 高最终性能"有贡献。

## Figure 9 — 速度超过人类 teleop(关键 insight)(page 9)

![Figure 9](../../extracted/rl-token-vla/pages/p09.png)

**原文 caption**:Speed on the Ethernet task. RLT significantly improves the speed of the Ethernet task. The final policy is faster even than the demonstrations produced by expert teleoperators. Half of the RL episodes during the critical insertion phase are faster than all of the teleoperated demonstrations.

**这张图讲什么**:Ethernet critical phase episode length 分布:Teleop 中位数 146 步,Base Policy 228 步(更慢,有 probing 行为),RLT 66 步。RLT 中位数不到 teleop 一半!且 RLT 一半 episode 比 teleop 全部都快。关键观察:RLT 不只是模仿,它学出 teleop 数据里没有的策略——直接接近端口、压入并轻微 wiggle 利用 compliance,流畅一次插入。这是"RL 能超越示教数据"的最有力证据。

---

## 用法

- 想看系统全貌:Figure 1(总览)、Figure 2(RL token 提取细节)
- 想看主结果:Figure 4(throughput)、Figure 5(success rate)、Figure 6(vs 其它 RL 算法)
- 想看每个组件必要性:Figure 7(throughput 消融)、Figure 8(success rate 消融)
- 想看 RL 超越示教的证据:Figure 9(速度超过 teleop)
- 想看任务难度:Figure 3(四任务)
