# DreamZero — 论文重要图

> 配套 `card.json` / `card.md`。论文共 16 张 Figure，全部列出。每张图给出：所在页、原文 caption（精简）、我们写的"这张图到底在讲什么"。
> 页图存在 `extracted/dreamzero/pages/pXX.png`，可对照查看。

## Figure 1 — Overview（page 1）

![Figure 1](../../extracted/dreamzero/pages/p01.png)

**原文 caption**：By jointly predicting video and action, World Action Models (WAMs) inherit world physics priors that enable 1) effective learning from diverse, non-repetitive data, 2) open-world generalization, 3) cross-embodiment learning from video-only data, and 4) few-shot adaptation to new robots.

**这张图讲什么**：这是论文的"门面图"，一眼给出 DreamZero 的四大卖点。最上面是数据来源——多种机器人（双臂 AgiBot、单臂、人类第一视角、纯视频、未来视频、语言、proprio），汇入中间的 World Action Model（DreamZero）。下方分两块：左边展示"少量 play data（30 分钟）就能 few-shot 适配新本体"，右边展示 zero-shot 泛化到未见任务/未见环境（如解鞋带、按电梯、放橙子进南瓜）。这张图的核心信息是：**WAM 已经能直接当 policy 用，做零样本策略**，而且能从异质、无动作标签的数据里学。

## Figure 2 — Joint Video and Action Prediction（page 3）

![Figure 2](../../extracted/dreamzero/pages/p03.png)

**原文 caption**：DreamZero jointly generates video and action. We observe that the predicted actions closely align with the generated video. The examples are from totally unseen tasks.

**这张图讲什么**：展示 video 和 action 是真的"对齐"的。上面一行是模型生成的未来视频帧，下面一行是机器人按 action chunk 实际执行的画面。两者帧对帧吻合——模型"想象的未来"和"实际执行的未来"一致。这支撑了论文的核心论断："失败主要来自 video 预测错，而不是 action 提取错"。任务"用锅铲炒蔬菜"是训练里没见过的。

## Figure 3 — Free-form Evaluation（page 4）

![Figure 3](../../extracted/dreamzero/pages/p04.png)

**原文 caption**：DreamZero performs a diverse range of tasks when conditioned on natural language instructions, including object manipulation, tool use, and human-robot interaction.

**这张图讲什么**：18 个自由测试任务的可视化清单（叠绿碗、扔垃圾、抓果汁盒插进杯子、推椅子、抽牌、双手交接铲子、拿酱瓶倒进杯子、按箭头方向移方块、拉车、递工具给人、滑盒子、叠短裤、放袋子进篮子、抓橡皮擦擦白板、抽湿巾擦桌、伸手进箱抓梨、拔电缆、和人碰拳、抓芒果放托盘、关灯开关）。这些是 100+ 自由测试任务里挑出来的，强调 DreamZero 能跟随自然语言指令做没见过的操作，没有停留在固定任务集上过拟合。

## Figure 4 — Model Architecture（page 6）★ 最重要

![Figure 4](../../extracted/dreamzero/pages/p06.png)

**原文 caption**：The model takes three inputs: visual context (encoded via a VAE), language instructions (via a text encoder), and proprioceptive state (via a state encoder). These are processed by an autoregressive DiT backbone using flow matching, which jointly predicts future video frames and actions through separate decoders. During training (left), for each chunk, the model denoises noisy video and action latents conditioned on clean video context. During inference (right), predictions are executed asynchronously in the real world, and ground-truth observations are fed back into the KV cache to prevent error accumulation.

**这张图讲什么**：这是全文最该看的一张图。左边训练：三路输入（视觉经 VAE、语言经 text encoder、proprio 经 state encoder）→ AR DiT 主干（flow matching）→ 两路输出（video decoder 出未来帧、action decoder 出动作 chunk）。训练用 teacher forcing，前 chunk 的 clean latent 作为上下文，对当前 chunk 的 noisy [z_t, a_t] 联合去噪。右边推理：生成一个 chunk 后，动作异步执行，真实环境的观测被回填到 KV-cache 替换预测帧，下一个 chunk 基于真实上下文生成。这张图同时回答了三个问题：**输入怎么拼、video 和 action 怎么联合、闭环怎么消误差**。对应我们 `architecture.md` 的总体数据流图。

## Figure 5 — Decoupled Noise Schedules（page 9）

![Figure 5](../../extracted/dreamzero/pages/p09.png)

**原文 caption**：DreamZero (blue) uses coupled noise for video and action (both uniform). DreamZero-Flash (red) biases video toward high-noise states via a Beta distribution while keeping action noise uniform, training the model to predict clean actions from noisy visual context.

**这张图讲什么**：两个子图，分别是 video timestep 和 action timestep 的采样分布。蓝色（标准 DreamZero）：video 和 action 都从 U(0,1) 均匀采样，两者噪声级别一致。红色（DreamZero-Flash）：video 的 timestep 偏向高噪声（Beta(7,1)，E[t]=0.125，即主要在 t 接近 1 的噪声大区域采样），action 仍均匀。为什么这样设计？因为 1-step 推理时 video 还没去噪干净，但 action 要从"干净的 video"读条件——训练时让模型见惯"video 很脏时也要输出干净 action"，推理时才不崩。这是 Flash 把 1 步推理从 52% 拉到 74% 的关键 trick。

## Figure 6 — AgiBot Pretraining Corpus Stats（page 11）

![Figure 6](../../extracted/dreamzero/pages/p11.png)

**原文 caption**：Distribution statistics for the AgiBot pretraining corpus: episode durations, subtask density, and skill coverage across 7.2K episodes (~500 hours).

**这张图讲什么**：三个子图。(a) episode 时长分布：7193 个 episode，平均 4.4 分钟，长尾到 10 分钟+。(b) 每个 episode 的子任务数分布：平均 42.4 个子任务，有的甚至 100+。(c) 技能分布：导航、躯干调整、抓放、工具使用等的占比。这张图支撑"数据多样性 > 重复性"的论断——这些 episode 是长程、多子任务、跨环境的真实部署式采集，和"一个任务重复 100 次"完全不同。对比传统 VLA 数据集，这是 DreamZero 能用异质数据学好的前提。

## Figure 7 — AgiBot Evaluation Set-up（page 12）

![Figure 7](../../extracted/dreamzero/pages/p12.png)

**原文 caption**：We are first-citizens of generalization evals, where the default setting is unseen environment and unseen objects.

**这张图讲什么**：评测协议示意。训练数据收集地与评测地不在同一地理位置，所以**默认评测就是 out-of-distribution**（unseen 环境 + unseen 物体）。三档任务：seen tasks（训练里有类似动作的，如叠衣服换颜色尺寸）、unseen tasks（动作完全没见过的，如解鞋带、熨衣服）、post-training 后的下游任务（叠衣服、水果打包、收餐桌）。这张图定义了论文所有结果的评测口径——"unseen env+obj"是底线，不是加分项。

## Figure 8 — Seen Task Evaluation（page 13）

![Figure 8](../../extracted/dreamzero/pages/p13.png)

**原文 caption**：DreamZero effectively learns from diverse data and generalizes to new environments, outperforming VLAs across all task categories. VLAs trained from scratch achieve near-zero success, while pretrained VLAs show modest performance—likely benefiting from embodiment-specific knowledge acquired through repetitive demonstrations during pretraining.

**这张图讲什么**：分组柱状图，三档任务（PnP Easy / PnP Hard / Contact-Rich）× 5 个 checkpoint（GR00T N1.6 scratch/pretrained、π0.5 scratch/pretrained、DreamZero scratch）。AgiBot G1 上 DreamZero 平均 task progress 62.2%，最好的 pretrained VLA 是 27.4%；from-scratch VLA 基本是 0。DROID-Franka 上同样规律。这张图回答 Q1：WAM 能从异质非重复数据学到东西，而 VLA 不能——即使 VLA 在数千小时跨本体数据上预训练过。

## Figure 9 — Zero-shot Generalization to Unseen Tasks（page 14）

![Figure 9](../../extracted/dreamzero/pages/p14.png)

**原文 caption**：DreamZero achieves non-trivial task progress on 10 tasks absent from training, while VLAs struggle across both embodiments.

**这张图讲什么**：10 个训练里完全没出现的任务（解鞋带、从假人头上摘帽子、用笔画、抽吸管、堆方块、画画、熨衣服、握手、折地图、拉车）。AgiBot 上 DreamZero 平均 39.5%，pretrained VLA 16.3%；DROID 上 success rate DreamZero 22.5%，GR00T 12.5%，π0.5 7.5%。关键观察：pretrained VLA 不管指令是什么，都倾向于伸手抓物体——说明它过拟合到训练里的主导行为（pick-and-place），不理解新任务语义；DreamZero 能做视觉规划并执行。这是"VLA 只有语义先验、缺物理动态先验"的最直接证据。

## Figure 10 — Posttraining Results（page 15）

![Figure 10](../../extracted/dreamzero/pages/p15.png)

**原文 caption**：WAMs enable stronger post-training results across three tasks, indicating that environment generalization of DreamZero is retained after post-training.

**这张图讲什么**：三个下游任务（叠衣服 33h、水果打包 12h、收餐桌 40h）post-training 后的结果。DreamZero 在叠衣服和收餐桌与 pretrained VLA 持平或略胜，在水果打包上明显胜出（76% vs 56%）。关键点：post-training 仍然在 unseen 环境评测，说明 DreamZero 的环境泛化能力在 fine-tune 后没丢——这是 WAM 比 VLA 的一个隐性优势，VLA post-training 后通常会过拟合到训练环境。

## Figure 11 — Cross-Embodiment Transfer（page 15）

![Figure 11](../../extracted/dreamzero/pages/p15.png)

**原文 caption**：We explore robot-to-robot (YAM → AgiBot) and human-to-robot embodiment transfer to unseen tasks.

**这张图讲什么**：跨本体迁移的两种设置。Robot2Robot：用 YAM 双臂机器人的 video-only 数据（20 分钟，无动作标签）共训。Human2Robot：用人类第一视角视频（12 分钟，无动作标签）共训。两条路都让 unseen tasks 从 38.3% 涨到 54%+。这张图的关键信息是：**WAM 能用没有动作标签的视频数据增益 policy**——因为 video prediction objective 本身就在强化 world model 对任务动态的理解，不需要 action。这打开了"人类视频规模远大于机器人数据"的 scaling 路径。

## Figure 12 — Few-shot Embodiment Adaptation（page 16）

![Figure 12](../../extracted/dreamzero/pages/p16.png)

**原文 caption**：We explore few-shot embodiment adaptation by post-training on 30 minutes of new embodiment play data and evaluating on pick-and-place variants requiring strong language following.

**这张图讲什么**：把 AgiBot 上训好的 DreamZero checkpoint，用 55 轨迹（约 30 分钟）YAM 机器人的 play data post-train，就能迁移到全新本体 YAM。而且迁移后仍保留 zero-shot 泛化——能处理没见过的物体（南瓜、泰迪熊、笔、杯面、纸袋）。论文说这是数据效率的新标杆。机制假设：学的是 implicit IDM（从视觉未来反推动作），比直接 policy learning 更 sample-efficient。

## Figure 13 — Bidirectional vs. Autoregressive WAMs（page 20）

![Figure 13](../../extracted/dreamzero/pages/p20.png)

**原文 caption**：When the sampling point falls mid-task (T=20), bidirectional WAMs must subsample video to align with the language caption, distorting native FPS and degrading video-action alignment. Autoregressive WAMs avoid this trade-off by conditioning on video context, preserving both language-video correspondence and native frame rate.

**这张图讲什么**：这是论证"为什么选 AR 不选 bidirectional"的关键图。双向模型要求固定长度，长程示教里语言指令只对应任务片段。如果不 subsample，生成的视频只覆盖任务片段，语言和视频错位（图里上半部分 mismatch 1/2）；如果 subsample 到任务区间，从任务中段（T=20）采样会扭曲原生帧率，video-action 对齐就崩（下半部分 mismatch）。AR 模型用视频上下文做条件，既保住语言-视频对应，又保住原生帧率，三模态对齐才稳。这是 DreamZero 选 AR 的根本理由。

## Figure 14 — Attention Strategy（page 21）

![Figure 14](../../extracted/dreamzero/pages/p21.png)

**原文 caption**：(a) QKV Self-Attention mask for DreamZero training. Y-axis shows the Query (Q) and X-axis shows the Key/Value (KV). Given conditioning frames (C0, C1, C2), we train the model to predict the velocities of next frames (Z1, Z2, Z3) and actions (Y1, Y2, Y3). (b) During inference, we compute the KV-cache of conditional frames and concatenate them to predict the action and frames. For example, Y3 (action) is able to attend to C0, C1, and C2, taking into account previous visual observations as history to predict the current actions during both training and inference. Note that the C0, C1, and C2 during inference is replaced with the GT observations.

**这张图讲什么**：Attention mask 的细节，对应 chunk-wise teacher forcing 和 KV-cache 推理。训练（a）：C0/C1/C2 是 conditioning 帧（clean），Z1/Z2/Z3 是要预测的 noisy video latent，Y1/Y2/Y3 是要预测的 action。下三角 mask 保证当前 chunk 能 attend 前面所有 clean context，但不能看未来。推理（b）：C0/C1/C2 换成 GT 观测，KV-cache 复用，Y3 能 attend C0/C1/C2——这就是"闭环用真实观测替换预测帧"的实现。这张图是 Figure 4 推理部分的细节展开。

## Figure 15 — Data Collection Environments（page 25）

![Figure 15](../../extracted/dreamzero/pages/p25.png)

**原文 caption**：We collect teleoperation data across 22 diverse real-world environments, including offices, laboratories, restaurants, supermarkets, coffee shops, warehouses, homes, hotels, and more.

**这张图讲什么**：22 个真实数据采集环境的照片网格——办公室、实验室、餐厅、超市、咖啡店、仓库、家、酒店等。这张图是"数据多样性"的视觉证据。和 Figure 6 配合看：Figure 6 给统计数字（7193 episode、4.4min/42subtask），Figure 15 给环境多样性证据。这是 DreamZero 反直觉结论"多样性 > 重复性"的数据基础。

## Figure 16 — Generated and Executed Pair（page 29）

![Figure 16](../../extracted/dreamzero/pages/p29.png)

**原文 caption**：We illustrate the generated video and action execution pair. These two examples show scenarios where the video prediction failed and the robot followed the failed prediction faithfully.

**这张图讲什么**：两个失败案例。上面一行是模型生成的未来视频，下面一行是机器人实际执行。关键观察：**模型生成的视频错了（比如把动作方向预测反了），机器人忠实执行了这个错误的轨迹**。这反向证明 action 和 video 对齐得很紧——失败主要来自 video 预测错，action 提取本身没问题。这是论文最有力的论断之一："改进 policy 等价于改进 video backbone"，因为 action 已经紧紧贴住 video。

---

## 用法

- 想看模型长什么样：Figure 4（架构）、Figure 14（attention 细节）
- 想看结果：Figure 8/9/10/11/12
- 想理解关键设计：Figure 5（Flash）、Figure 13（为什么 AR）
- 想理解数据：Figure 6（统计）、Figure 15（环境）
- 想理解对齐质量：Figure 2（成功对齐）、Figure 16（失败也对齐）
