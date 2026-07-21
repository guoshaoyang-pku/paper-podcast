# RL Token — 结构化卡片

> 论文:**RL Token: Bootstrapping Online RL with Vision-Language-Action Models**(Physical Intelligence, arXiv 2604.23073, 2026-04-30)
> 配套:`card.json`(机器可读) · `architecture.md`(架构图) · `podcast.md`(口播稿)

## 一句话定位

给冻结的 π0.6 VLA 训一个"RL token"瓶颈表示,在它上面挂一个几层 MLP 的 actor-critic 做 online RL 微调,几小时甚至几分钟真实机器人数据就能把毫米级精密任务的 critical phase 提速 3× 并显著提升成功率。

## 它要解决的问题

**问题**:VLA(π0.6)开箱能做很多操作,但在"最后一毫米"上失败——动作慢、要重试、critical phase 小误差累积成失败。要靠 RL 精修。但全量 RL 微调一个 billion 参数 VLA 既算力贵又样本低效,真实机器人几小时预算内做不完;而 SERL/RL100 这类小模型 online RL 虽然几小时能学,但放弃了 VLA 的泛化先验。

**为什么 prior work 不够**:现有 VLA-RL 路线分两极端。一端 RECAP/PPO 直接更新整个 VLA,需大规模数据、算力贵。另一端轻量辅助模块:ConRFT 冻结 VLA encoder 训单步 action head(无 chunking,credit assignment 长 horizon 失败);Policy Decorator 学残差(只 sim 验证,百万步);PLD 单步残差;DSRL 在 diffusion noise 空间学 latent policy(强约束 VLA、改进空间小)。共同问题:要么没 chunking 导致稀疏奖励 credit assignment 失败,要么强约束 VLA 限制改进空间,要么单步无法匹配 VLA 原生 chunk 接口。

## 输入 / 输出

| 方向 | 名称 | 类型 | 说明 |
|---|---|---|---|
| 输入 | camera images | image | 2 腕部 + 1 base,喂冻结 π0.6 产 z1:M |
| 输入 | language | text | 每任务固定一条指令(实验) |
| 输入 | proprio | vector | 关节位置+速度(screw);末端位姿(zip/Ethernet/charger) |
| 输入 | VLA 参考 chunk | continuous | 冻结 π0.6 采样 H=50 步;RL 用前 C=10 步 ã1:C |
| 输出 | RL action chunk | continuous | C=10 步 @50Hz = 0.2s,14 维/步 = 140 维 |
| 输出 | Q value | scalar | critic 估的 chunk-level C-step return |

控制频率:50Hz;VLA chunk H=50 (1s);RL chunk C=10 (0.2s);replay stride 2,1 秒数据≈25 样本。

### 输入拼接 protocol

```
<bos_vla>[img_wrist1, img_wrist2, img_base, lang, proprio] → π0.6(冻结)→ z1:M token embeddings<eos_vla>
<bos_rltok>[appended e_rl=<rl>] → encoder g_φ → z_rl (1×2048)<eos_rltok>
<bos_rlstate>[z_rl ⊕ proprio]<eos_rlstate>
<bos_ref>[ã1:H ∼ π_vla(冻结采样)]<eos_ref>
<bos_actor>[π_θ(· | x, ã) → N(μ_θ(x,ã), σ²I) → a1:C]<eos_actor>
```

逐段解释:

- `<bos_vla>...<eos_vla>`:冻结 π0.6 处理图像+语言+proprio,产出 final-layer token embeddings z1:M(每 token 2048 维)。RL 训练时全程冻结,只前向。
- `<bos_rltok>...<eos_rltok>`:在 z1:M 末尾 append 学习的特殊 embedding e_rl=<rl>,过轻量 encoder transformer g_φ,取 <rl> 位置输出 z_rl(1×2048)。z_rl 是瓶颈——decoder d_φ 训练时要从它自回归重建原 z1:M,逼 z_rl 保留 task-relevant 信息。训练完 φ 后冻结。
- `<bos_rlstate>...<eos_rlstate>`:RL 状态 x = (z_rl, proprio)。z_rl 提供 VLA 压缩感知+行为先验,proprio 补充即时本体感受。
- `<bos_ref>...<eos_ref>`:VLA 参考动作 chunk ã1:H(实际取前 C=10 步)。RLT 关键设计:actor 条件化于 VLA 动作建议,而非从零生成。
- `<bos_actor>...<eos_actor>`:小 MLP actor 输出高斯分布,采样 a1:C ∈ R^10×14。训练时 loss = -Q(x,a) + β·‖a-ã‖²(KL 正则 anchor 到 VLA)。

**三个坑**:①language 在 RL token 提取阶段被实验性 drop(任务固定指令),不是序列拼接进 RL;②多视角在 VLA 内部已被处理成 token,RL 侧只见 z_rl;③训练时 reference action 50% 概率 drop 成零(防 actor 抄作业),推理时必给。chunk 内开环执行 C=10 步,然后重新观测+重新规划。

## 数据集

| 数据 | 规模 | 备注 |
|---|---|---|
| π0.6 预训练 corpus | 数万小时(继承) | VLA 主干已预训练,RLT 不重训 |
| Screw installation demos | 1-10h teleop | M3 螺丝+电动螺丝刀,亚毫米对齐,10cm screwdriver 杠杆放大旋转误差 |
| Zip tie fastening demos | 1-10h | 双臂协调穿扎带尾进窄锁孔,可变形物体毫米级容差 |
| Ethernet insertion demos | 1-10h | Ethernet 接头插凹陷端口,需角度+力度果断插入 |
| Charger insertion demos | 1-10h | 充电器插排插,厘米级对齐,prongs/socket 观测不全 |

## 架构(详见 architecture.md)

- **主干**:π0.6(SigLIP 400M + Gemma 4B VLM + 860M action expert),~5.3B,**冻结**
- **RL token**:encoder/decoder φ 轻量 transformer;输出 1×2048 bottleneck 向量
- **actor-critic**:2 层 MLP hidden 256(zip/Ethernet/charger)或 3 层 MLP hidden 512(screw);参数量 << 1M
- **结构**:frozen VLA + 瓶颈 RL token + 轻量 off-policy actor-critic(TD3 风格,chunk-level)

**为什么这样设计**:核心思想是分工——冻结 VLA 提供感知+行为先验,小 actor-critic 在 critical phase 做 local refinement。RL token 是关键桥梁,把 VLA 2048×M 高维 token 压成 1×2048 瓶颈向量,既保 task-relevant 信息又小到能挂轻量 MLP。三个设计选择互相强化:①chunk-level action (C=10 < H=50) 缩短 TD horizon;②actor 条件化 VLA 参考 + KL 正则,把 RL 变成 local refinement;③reference dropout 防抄作业。

### 数值 sense

| 项 | 值 |
|---|---|
| 主干 | π0.6 = SigLIP 400M + Gemma 4B + 860M action expert;总 ~5.3B(冻结) |
| RL token | 1×2048 维 bottleneck 向量;encoder/decoder φ 轻量 transformer(精确层数论文未给) |
| actor-critic | 2 层 MLP hidden 256(zip/Ethernet/charger);3 层 MLP hidden 512(screw);<< 1M 参数 |
| RL chunk | C=10 步 @50Hz = 0.2s;每步 14 维 → 140 维 chunked action |
| VLA chunk | H=50 步 @50Hz = 1s |
| 控制频率 | 50Hz;chunk 内开环执行前 10 步后重新规划 |
| RL token 训练 | φ 训 2000~10000 gradient steps on 单任务 demo;之后 VLA+φ 全冻结 |
| online RL | 400~1000 episodes/task;15min~5h 真实数据;U2D G=5;2 critic 更新 per 1 actor 更新;reference 50% dropout |
| replay subsampling | stride 2 存中间步;1 秒≈25 样本;off-policy 复用 VLA+RL+人类干预 |
| reward | 稀疏 +1(人标 success)/0(failure);episode 30~120s(1500~6000 步);critical phase 5~20s(250~1000 步) |
| critic | twin Q 集成(TD3),取 min 算 target;target network ψ' |

**给听众的标尺**:5.3B 冻结 VLA + 1×2048 RL token + 几层 MLP(256/512 hidden)+ C=10 chunk + 50Hz + U2D=5 + 15min~5h 真实数据。最该记住的是"小":actor-critic << 1M 参数,所以几小时能学。

## 关键技术

1. **RL token 瓶颈表示** — 在 VLA token 末尾 append e_rl,过 encoder 取该位置输出 z_rl(1×2048);decoder 自回归重建原 embedding 逼瓶颈。换 ResNet-10 throughput 降 50%,证明 RL token 编码了通用编码器没有的 manipulation 结构。
2. **Chunk-level actor-critic** — C=10 < H=50,把 TD horizon 从 1500~6000 步缩到 10 步级,稀疏 reward 才能传播。HIL-SERL/PLD 单步方法在 Ethernet 完全失败是直接对照。
3. **Reference-action conditioning + KL anchor** — actor 条件化 VLA 参考 ã + β·‖a-ã‖² 正则,把 RL 变成 local refinement 而非无约束搜索。reference 50% dropout 防抄作业。w-o BC Regularizer 跌最狠。
4. **Critical-phase targeting + 两阶段训练** — 只在 critical phase 用 RL,几小时数据集中在这段。先 critical-phase only 再扩 full-task,让策略对 VLA 诱导状态分布鲁棒。
5. **Off-policy + 高 U2D + 异步 rollout** — TD3 twin Q + replay 复用 VLA warmup + RL rollout + 人类干预;U2D=5 榨干每条 transition;异步 rollout 和学习。继承 SERL/RL100 的 sample-efficient 配方。

## 关键结果

| 指标 | RLT | 最强 baseline | setup |
|---|---|---|---|
| Critical phase speedup | up to 3× faster | base π0.6 VLA | 四任务 critical phase,15min~5h 数据 |
| Screw success rate (full-task) | +40% over base | base π0.6 VLA | M3 螺丝亚毫米对齐,5h 数据 |
| Zip tie success rate (full-task) | +60% over base | base π0.6 VLA | 双臂穿扎带,5h 数据 |
| Ethernet speed (median) | 66 steps | 146 steps (teleop) / 228 (base VLA) | RLT 比 teleop 快 2.2×,一半 episode 比所有 teleop 快 |
| Ethernet vs RL baselines | RLT 高成功 | DAgger/DSRL 追平成功但 throughput 低;HIL-SERL/PLD 失败 | 同等数据量 |
| Sample efficiency (Ethernet) | 5min 数据即超 base | base π0.6 VLA | total experiment ~40min |
| Ablation: w/o RL token | throughput 降 50% | full RLT | 证明 RL token 编码 manipulation 结构 |
| Ablation: w/o BC regularizer | 最大单点跌幅 | full RLT | 无 anchor 让 actor 全空间裸搜 |
| Robot data budget | 15min~5h per task | — | 400~1000 episodes |

## Insights

1. 分工哲学:VLA 该当感知+行为先验,fine-tune 该靠小 actor-critic 在瓶颈表示上做 local refinement——"全量 RL 微调 VLA"和"小模型从零 RL"都是错的两极端(p1, p4)。
2. RL token 是 VLA → 小 RL 的关键桥梁:它把 2048×M 高维 token 压成 1×2048 瓶颈向量,靠 decoder 重建 loss 保证不丢 task-relevant 信息,又小到能挂几层 MLP。换 ResNet-10 throughput 降 50%(p9 Figure 7)。
3. chunk-level action 是稀疏奖励 RL 的必需品:50Hz 任务 episode 1500~6000 步,单步 RL credit assignment 失败;C=10 把 TD horizon 缩到 10 步级。HIL-SERL/PLD 单步方法在 Ethernet 完全失败就是证据(p8 Figure 6)。
4. **RL 能学到示教数据里没有的策略**:RLT Ethernet 中位数 66 步,比 expert teleop 的 146 步快 2.2 倍,一半 episode 比所有 teleop 都快。RLT 学会直接压入+wiggle 利用 compliance,teleop 数据里没有这个策略(p9 Figure 9)。
5. reference action 50% dropout 是反直觉但必要的 trick:条件化 ã 会让 actor 倾向抄作业;dropout 逼 actor 保独立 action 通路,critic 有信号后自然学会偏离 ã(p5)。
6. critical-phase targeting 是 practicality 的关键:承认 VLA 不该全推翻,只在它最弱的 critical phase 用 RL 补。几小时数据集中在一个 5~20s 段,才有可能学会(p5-6)。

## vs 同类工作

- **vs RECAP / 全量 VLA RL**:RECAP 训整个 π*0.6,需大规模数据、算力贵、不适合几小时 adaptation;RLT 冻结 VLA,只训小 actor-critic,15min~5h 真实数据就能学。
- **vs ConRFT / PLD(单步 residual)**:ConRFT 无 chunking,稀疏奖励长 horizon credit assignment 失败;RLT chunk-level (C=10) 匹配 VLA 原生接口。PLD 同样单步限制。
- **vs DSRL(diffusion noise space)**:DSRL 强约束 VLA 动作生成、改进空间小,throughput 明显输 RLT;RLT actor 直接条件化参考动作并 KL 正则,允许更自由的 local refinement。
- **vs Policy Decorator**:残差+超参缩放,只 sim 验证、百万步;RLT 真实机器人几小时数据验证四任务。
- **vs GR-RL**:多阶段针对长horizon 鞋带任务;RLT 通用 critical-phase targeting + chunk-level actor-critic,更直接。
- **vs HIL-SERL/SERL/RL100**:用 ResNet encoder,放弃 VLA 泛化先验;RLT 用冻结 VLA + RL token,既保 VLA 知识又拿到小模型样本效率——HIL-SERL 在 Ethernet 完全失败是直接对照。

## 局限

**论文自承**:
- 仍需人在环——reward 信号、干预纠正、RL/base policy 切换都靠人。全自主 RL pipeline(用 reward model + progress prediction)是 future work(p7)。
- critical-phase 隔离评测降低方差但也简化问题;full-task 评测显示 compounding error 仍让整体成功率低于 critical-phase only(p7)。
- 只在固定语言指令任务验证(每任务一条指令);RL token 提取时实验性 drop 语言 embedding——多指令/语言条件 RL 未验证(p4 footnote)。

**我们读出**:
- 四任务都是 50Hz 双臂/单臂精密插入类,形态接近;RLT 在更动态任务(接球、长horizon 多阶段装配)是否仍几小时收敛未证。
- actor-critic 是 Gaussian 单模态,靠 reference ã 暴露多模态;如果 VLA 本身多模态弱(参考 chunk 单一模式),RLT 多模态恢复能力受限,论文未测。
- RL token encoder/decoder 的精确参数量、层数论文未给;β 的具体值也没明示——复现需从 code release 补。

## 可复现性

- 代码:https://pi.website/research/rlt(PI 代码 release 声明)
- 权重:依赖 π0.6 base VLA(PI release);RL token + actor-critic checkpoint 论文未明示是否开源
- 仿真基准:无(全真实机器人)
- 真机评测:四真实任务(screw/zip tie/Ethernet/charger),critical-phase + full-task 两设置,50 episodes/task

## 论文重要图(详见 `figures.md`)

| 图 | 页 | 重要性 | 一句话 |
|---|---|---|---|
| [Figure 2](../../extracted/rl-token-vla/pages/p04.png) | 4 | key | RL token 提取细节,全文最该看,瓶颈表示怎么来 |
| [Figure 1](../../extracted/rl-token-vla/pages/p01.png) | 1 | key | RLT 系统总览,分工哲学 |
| [Figure 4](../../extracted/rl-token-vla/pages/p08.png) | 8 | key | throughput 全面超 base,难任务提升大 |
| [Figure 5](../../extracted/rl-token-vla/pages/p08.png) | 8 | key | success rate,screw +40%/zip tie +60% |
| [Figure 6](../../extracted/rl-token-vla/pages/p08.png) | 8 | key | vs 其它 RL 算法,单步方法失败 |
| [Figure 7](../../extracted/rl-token-vla/pages/p09.png) | 9 | key | 消融 throughput,每个组件都重要 |
| [Figure 9](../../extracted/rl-token-vla/pages/p09.png) | 9 | key | 速度超过人类 teleop,RL 学到示教没有的策略 |
| [Figure 8](../../extracted/rl-token-vla/pages/p09.png) | 9 | supportive | 消融 success rate,配合 Figure 7 |
| [Figure 3](../../extracted/rl-token-vla/pages/p06.png) | 6 | supportive | 四评测任务,critical phase 可视化 |

## 标签

`VLA` `online RL` `fine-tuning` `actor-critic` `RL token` `bottleneck representation` `chunked actions` `π0.6` `Physical Intelligence` `precision manipulation`
