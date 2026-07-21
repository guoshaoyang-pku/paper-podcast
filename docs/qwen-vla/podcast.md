# Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments

2026-05-28，Alibaba Qwen Team 的 Qwen Team发布的论文，标题是 Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments。

## 开场:这篇论文真正要解决什么

这篇论文要解决一个非常根本的碎片化问题:**现在的具身模型是分裂的——操控模型只管桌面或灵巧,导航模型只管 waypoint,各自专属本体。你想让一个机器人既会操控又会导航,得训两个模型,还都绑死在一个本体上。**

为什么这件事难。因为这些任务表面看异质得要命。操控要预测末端位姿、关节位置、夹爪状态、灵巧手构型;导航要预测 waypoint 或离散移动决策;人类第一视角视频给的是腕和手轨迹,不是机器人控制信号。它们的观测格式、控制频率、预测 horizon、动作维度、评测协议全都不一样。

但 Qwen-VLA 的核心观察是:**这些任务在计算结构上是共享的**——都是 agent 条件于视觉观测、语言指令、本体约束,然后预测未来动作或轨迹。所以可以统一进单个 VLA。

它的解法分三块:用 embodiment-aware prompt——一个文本模板描述当前机器人是什么、几臂、控制频率多少、预测几步——作为唯一的本体特定接口;用统一的 action-and-trajectory 表示——所有动作塞进一个 H×K 张量,实际用的通道放前面,余下零填充加 mask;用一个 DiT flow-matching action expert 专门处理连续动作的多模态和高频动态。

它走通了,一个通用模型同时做到 LIBERO 97.9%、RoboTwin-Easy/Hard 86.1%/87.2%、ALOHA 真实 OOD 76.9%、VLN RxR 59.6% SR、DOMINO 动态操控零样本 26.6%——在多个 benchmark 上反超专门 finetune 的 specialist。

## 它的输入和输出到底是什么

你需要在脑子里先建一个模块图。

输入有四路:**视觉观测**,单帧或多帧,多视角用一对 view tag 包裹,比如 ego、cam_left_wrist、cam_right_wrist,这让 VLM 形成视角感知的表征;**语言指令**,粗的如 "pick up the red cup",细的是 13 维 fine-grained caption;**embodiment prompt**,文本模板描述当前机器人;**任务标识符**,可选,标识是操控还是导航还是轨迹。

输出就一路:**action 或 trajectory chunk**,一个 H×K 的连续张量。H 是固定预测 horizon,K 是固定通道数。实际用的通道数 c 小于等于 K,放在前 c 维,剩下的零填充,配一个 per-channel mask 告诉模型哪些位置有效。

### 输入到底怎么拼成一条序列

光说四路输入还不够具体,我把它拆成一条显式的拼接序列,你在脑子里能建出来。

序列大致长这样:开头是 embodiment prompt,一个文本模板,告诉模型"这个机器人是某某型号、单臂还是双臂、有没有腰或移动底盘、控制频率多少赫兹、请预测接下来多少步动作,执行这个任务"。然后是多视角观测,每个图像用 view tag 包裹,排成一串。接着是语言指令。然后是 VLM 处理完这些之后的 hidden states,经一个 linear 层投影到 DiT 的 channel 维度。最后是 action expert 要处理的 noisy action chunk,它和 VLM hidden 拼接成一条序列进 DiT。

这里有几个坑要特别说清,不然容易误解。

第一,language 和 embodiment 是 VLM 的 causal 序列 token,而 action 是 DiT 的连续张量,不是 token——它通过和 VLM hidden 拼接进 joint self-attention 来耦合。所以这套设计相比统一自回归,区别在于是 VLM 和 DiT 两个解耦模块通过 hidden state 拼接加 cross 注意协作。

第二,多视角不在序列层拼接。它在 VLM 内部用 view tag 标识,让 VLM 形成 view-aware 表征,action expert 可以选择性 attend 每个视角。不需要改架构或加输入通道。

第三,训练阶段不同,这条序列长得不一样。T2A 阶段——也就是第一阶段——完全不给图像,只给文本加 embodiment prompt,让 DiT 从语言重建动作。CPT、SFT、RL 才加图像。而且每个 dataset 保留原生动作格式,靠 embodiment prompt 告诉模型当前是什么控制约定,不强制统一物理语义。

记住这条序列,后面听别的统一 VLA 论文你会反复遇到。Qwen-VLA 的答案是:context 是 embodiment prompt 加多视角加语言,language 是 VLM causal token,action 是 DiT 连续张量,本体差异全部靠 prompt 解决。

## 架构:4B VLM 加 1.15B DiT action expert

主干是 Qwen3.5-4B,一个原生多模态 VLM。它的特点是早期视觉语言融合——ViT 空间合并后的视觉 token 直接 interleave 进文本流,图像视频语言在单个 transformer 里统一处理。它的 hybrid attention 设计,多数层用 gated linear attention 做高效长序列编码,周期性插一层 grouped-query softmax attention 保全局精度推理。

在 VLM 之上接一个 DiT-based flow-matching action expert,专门做细粒度连续动作生成。这个 expert 大约 1.15B 参数,16 个 DiT block 每个 70.8M,占大头 1.13B。剩下的是 action projection MLP、VLM 到 DiT 的 linear 层、timestep embedding、output AdaLN 调制。

action expert 把 VLM hidden states 和 noisy action chunk 拼接成一条序列,过 16 个 block 的 joint self-attention,配 AdaLN 注 timestep,用 multi-section RoPE 对齐 backbone。flow matching 目标预测一个 velocity field,推理时少步 Euler 积分出 action chunk,低延迟实时控制。

这个解耦设计是关键。VLM backbone 已经从大规模预训练得到了通用视觉语言表征,DiT action decoder 是随机初始化的。如果 naive 联合训练,decoder 要同时学动作分布形状、语言条件、本体条件、flow-matching 动力学、视觉 grounding——这太多太乱,而且噪声梯度会扰动 VLM 的预训练表示。所以解耦:action expert 专注细粒度连续动作多模态高频动态,VLM 保预训练能力。

## 给你一个数值 sense:这套模型到底多大

光说 4B 加 1.15B 可能没感觉,我把维度念细一点。

VLM 是 Qwen3.5-4B,原生多模态,hybrid attention——多数层 gated linear,周期性 grouped-query softmax。ViT 空间合并后的视觉 token 直接进文本流。action expert 1.15B,16 个 DiT block 每个 70.8M 加起来 1.13B,再加 action projection MLP 4.9M、VLM 到 DiT 的 linear 3.9M、timestep embedding 2.8M、output AdaLN 4.7M。

注意 Qwen-VLA 没有独立 VAE,不是 latent diffusion。action expert 直接在 raw action 空间做 flow matching。每个 dataset 用 1% 和 99% 分位数做 quantile 归一化,把动作压到 -1 到 1,去掉跨本体尺度差但保留每个 dataset 内的相对运动结构。

action chunk 是 H×K 张量。SFT 操控任务 H=16 步每 chunk,导航 H=8 个 waypoint 每 chunk。合成数据 50Hz。RL 阶段 H=16,采样温度 τ=1.0 训练、0.6 评测来锐化动作分布。

动作维度看任务。操控是 delta 末端位姿加 Euler 或 quaternion,或绝对关节位置,加夹爪,加灵巧手关节。导航是每个 waypoint 的 delta 平移加朝向变化,三个自由度。人类第一视角数据更特殊——每只手腕用 SE(3) 变换表示 6 维,手部 articulation 对 45 维 axis-angle 关节做 PCA 取前 10 主成分叫 eigengrasps,所以每只手 16 维,双手 32 维每步。这些不同物理语义都塞进同一个 H×K 张量,靠 embodiment prompt 和 mask 区分。

训练规模。T2A 阶段冻结 VLM 只训 DiT,2000 步,用 Sigmoid-Normal timestep 分布。CPT 解冻两模块,异质混合含 sim 加 real。SFT 双轨:一路多任务(VL 加操控加导航),一路真实机器人 in-house teleop;VL loss 权重 0.1,action loss 权重 1.0。RL 用 PPO 加 GAE,折扣 γ=0.99,trace-decay λ=0.95,clip ε=0.2,只在 SimplerEnv 单环境做稀疏二值奖励。128 个并行环境实例,每次迭代 8192 个 transition chunk。value head 学习率 1e-4,actor 5e-6,差 20 倍,让 value 快收敛同时 policy 保守更新。

这套训练量不小,但相比 π0.7 那种三模型系统,Qwen-VLA 是单一端到端模型,部署更轻。

## 真正让 Qwen-VLA 成立的:T2A 阶段

这是论文最核心的设计,我必须单独讲。

核心视角是:动作学习是结构化解压缩。一条语言指令比如 "pick up the red cup" 加一个 embodiment prompt,几个 token 就紧凑编码了任务意图。但对应的动作轨迹可能跨数百个高维关节位置值。这里有个巨大的维度差距,桥接它是一个结构化解压缩问题。

T2A 阶段就是学这个解压缩映射。冻结 VLM,只训 DiT,条件是文本加 embodiment prompt,故意不给图像。DiT 必须从紧凑的语言编码重建高维动作分布,没有任何视觉捷径。这样 DiT 学到的是:不同语言描述选不同动作空间区域,embodiment prompt 怎么把同样任务意图调制成平台特定运动程序,完整动作轨迹的时序相干性和组合性。

Fig.6 给了决定性证据。第一,数据组成——纯真实数据 51%,纯合成 64%,但 20% 合成加 80% 真实最佳 71%。合成广覆盖语言动作对应,真实锚定物理动态。第二,序列预测模式——full-sequence 比 chunk 一致更好,因为完整轨迹让 decoder 学语言怎么映射到完整动作序列,chunk 只给局部片段。第三,也是最关键,T2A 加图反而 -2.87 个点。因为图像一加,decoder 就走视觉捷径,不去建可靠的语言-动作映射,稀释的正是 T2A 要建的那个先验。所以 T2A 必须完全无视觉,视觉 grounding 推迟到 CPT,那时 decoder 已经有好的动作先验了。第四,timestep 分布——Sigmoid-Normal 在 T2A 最佳(中间 timestep 信噪比最 informative,因无视觉引导),Beta 在 SFT 最佳(有 VLM 条件后更 sample-efficient)。第五,训练时长——2000 步最佳,40000 步过拟合掉到 60.42%。因为语言-动作映射结构上比完整 vision-language-action 空间低维,少量步就够捕获,训太久开始记具体轨迹实例而不是精炼结构先验。

这套消融决定性地证明:T2A 不是 warm-start,是建一个语言索引的结构化动作先验。而且 Fig.7(b)证明这个先验可迁移——预训练的 DiT 接一个新的 Qwen3.5 VLM,比 from-scratch 的 DiT 全程更快收敛更高峰值。说明 T2A 学的不绑定特定 backbone。

## embodiment prompt:把多本体从架构问题变成 prompt 问题

这是 Qwen-VLA 第二个核心设计。

多本体的难点是动作维度和语义不同——7 自由度臂 vs 29 自由度全身,末端位姿 vs 关节 vs 灵巧手。传统做法是每本体一个专属 head 或 encoder。Qwen-VLA 的解法是 embodiment-aware prompt 加统一 action 表示。

每个训练样本前缀一个文本 prompt,模板化描述:这个机器人是什么型号、单臂还是双臂、有没有腰或移动底盘、控制频率多少赫兹、预测接下来多少步动作。这是模型知道当前控制什么机器人的唯一接口。action 用统一张量,实际用的通道放前面,余下零填充,per-channel mask 排除填充进梯度。每 dataset 保留原生动作格式,靠 prompt 告知,不强制统一物理语义。

Table 10 消融给决定性证据。Multi-MLP(每本体私有 encoder/decoder)、Concatenation(所有本体动作 concat)、Zero-Padding(共享 MLP 加零填充)三种 projection 设计,在 Bridge 和 Robocasa 上差异都小于 1.2 个点。Zero-Padding 参数最少,所以选默认。这说明什么?说明一旦建立了共享 latent 空间,projection 设计的影响很有限——真正关键的是 embodiment prompt。

这把多本体从架构问题变成了 prompt 问题。加一个新本体不需要改架构不需要新 head,只要写一段 embodiment prompt 描述它。这是 Qwen-VLA 能跨那么多本体的根本原因。

## 关键结果:一个通用模型反超多个 specialist

我挑几个有对照的。

**操控仿真四 benchmark**:LIBERO 97.9%、RoboCasa-GR1 56.7%、Simpler-WidowX 73.7%、RoboTwin-Easy/Hard 86.1%/87.2%。注意这些都是 specialist 在各自 benchmark 上单独 finetune 的,而 Qwen-VLA 是一个通用模型在所有 benchmark 上联合训练后直接评测。在 RoboCasa 上 56.7% 超 π0.5 的 37.0%、GR00T N1.6 的 49.9%;在 Simpler 上 73.7% 超 StarVLA-OFT 的 64.6%。证明联合多本体训练不牺牲任务性能,反而常超专门模型。

**ALOHA 真实**:in-domain 平均 83.6%,OOD 平均 76.9%。关键是 ablation——同架构从零训只有 48.5%/36.2%,从 Qwen-VLA-Base finetune 涨到 83.6%/76.9%。这证明性能不来自架构,主要来自预训练。OOD 76.9% 比 π0.5 的 41.5% 高 35.4 个点,在 background 和 instruction generalization 上尤其强(80.8%/84.6%)。

**VLN 导航**:R2R Val-Unseen SR 57.5%、OSR 69.0%,RxR SR 59.6%、SPL 47.8%,都超 specialist 导航模型 StreamVLN。而且 Qwen-VLA 同时是操控模型——操控和导航共训练互益。

**DOMINO 动态操控零样本**:SR 26.6%、MS 39.5%。这个最惊人。Qwen-VLA 没用任何动态操控数据训练,只用当前帧观测,在 35 个动态 suite 上零样本超所有 baseline——包括专门 finetune 的 PUMA(17.2%)和 WAM 风格的 LingBot-VA(24.1%)。论文解释是 flow-matching 出的 action chunk 减少 hesitation 帮策略在窄时间窗精确行动,加上大规模联合预训练给的 spatial-to-kinematic 先验。

**RL 后训练累积增益**:Table 11。CPT→SFT 大幅提升(LIBERO 90.8→97.8,RoboTwin-E 64.3→86.3),SFT→RL 进一步(SimplerEnv 70.8→73.7)。关键是 RL 增益不局限于训练环境——RL 只在 SimplerEnv 做,但 RoboCasa +0.7pp、DOMINO 零样本动态 SR 25.7→26.6。证明 task-success 优化的"decisive 执行加误差恢复"泛化到未见环境。

## 一个最值得记住的 insight

论文里有一个观察我建议你重点记:**VL 和 action 共训练是互益的,不是互斥的。**

Fig.7(a)消融:VLA-Only vs VL+VLA。在简单 benchmark(Libero/Simpler)两者持平,说明 VL 共训练无干扰。但在需要细粒度物体识别和组合指令解析的难任务上,VL+VLA 明显赢——RoboCasa +4.9pp,RoboTwin +4.6pp。

这个 insight 分量很重。它推翻了"action 训练会污染 VLM 所以要隔离"的直觉。事实上,VL 数据的物体词汇和视觉多样性正好补 action 数据的窄分布,而 action 的细粒度控制信号又强化了 VL 的空间推理。这是"统一 VLA"主张的直接证据——VL 和 action 在共享 backbone 上互益,这是 Qwen-VLA 能同时做好操控、导航、VL 三件事的根本。

## 局限:别替它过度外推

论文自己承认的:具身动作数据规模和多样性远小于 VL 预训练数据,限制长尾物体/环境/本体/接触密集交互的鲁棒性;VL 加导航加动作的联合训练引入优化 trade-off,action 训练会让部分纯 VL 和导航评测小幅回退,需要更好的目标平衡和数据课程;当前评测仍主要短程 benchmark 驱动,长时程易失败的真实部署仍是开放挑战。

我们读出的:T2A 阶段完全无视觉,建的先验是语言索引的——对视觉歧义大(同语言多视觉场景)的任务,这个先验可能不够,需要 CPT 和 SFT 补,但论文没给 T2A 先验在视觉歧义任务上的消融;state conditioning 消融显示 proprio 收益 marginal(≤1.3pp)故默认不用,但这是在多视角观测充分的前提下,对腕部不可见或遮挡场景,放弃 proprio 可能损失,论文未给这个边界分析;RL 只在 SimplerEnv 单环境做,泛化增益虽在但幅度小(多数小于 1pp),且依赖稀疏二值奖励语义,对奖励难以二值定义的任务(比如操控质量)RL 路线未证;统一 action 用零填充加 mask,但固定通道 K 和固定 horizon H 的选择论文没详述——对本体 horizon 差异大的(导航 2FPS vs 操控 50Hz),固定 H 可能造成某些本体 chunk 过长或过短。

所以,Qwen-VLA 是统一 VLA 路线的一个强证据,但它支持的是"embodiment prompt 加四阶段训练能让单模型跨多本体多任务"这个方法论,不等于所有具身任务都已解决——它依赖大规模异质预训练数据和精心分阶段训练,这是它的隐形门槛。

## 一句话收束

Qwen-VLA 把 Qwen3.5-4B VLM 接一个 1.15B DiT flow-matching action expert,用 embodiment prompt 把多本体从架构问题变成 prompt 问题,用四阶段训练(T2A 纯文本→动作解压缩→CPT 视觉 grounding→SFT→RL 闭环)解决 VLM 已预训练而 DiT 随机初始化的不对称,把操控、导航、轨迹预测、人类第一视角动作统一进同一 action-and-trajectory 预测问题。它最有力的论证是 Fig.6,而非某个跑分——T2A 必须无视觉且短训,加图反而变差。这把"动作学习是结构化解压缩"这个视角工程化成了可复现的训练 recipe。
