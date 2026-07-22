# Motus — 论文重要图

> 配套 `card.json` / `card.md`。论文共 12 张 Figure，全部列出。每张图给出：所在页、原文 caption（精简）、我们写的"这张图到底在讲什么"。
> 页图存在 `extracted/motus/pages/pXX.png`，可对照查看。

## Figure 1 — Motus 架构（Tri-model Joint Attention）（page 2）★ 最重要

![Figure 1](../../extracted/motus/pages/p02.png)

**原文 caption**：Motus Architecture. Here, at...at+k are actions, zt...zt+k are latent actions, and τv and τa are the rectified flow timesteps for the video generation model and the action expert, respectively.

**这张图讲什么**：全文核心架构图。三个专家（Video Gen. Model / Action Expert / Und. Expert）各自有 LayerNorm+AdaLN+QKV+FFN 独立分支，但在 Tri-model Joint Attention 处把三路 QKV 拼起来做共享自注意力——这是 MoT 的关键：**不拼 token，而是拼 attention**。右侧画出 Video Encoder/Decoder、Action Encoder/Decoder 两个 VAE。底部标 τv、τa 两个独立 timestep——UniDiffuser 式调度的视觉证据。看清这张图就理解了 Motus 为什么能用一个模型切换五种模式。

## Figure 2 — Action-Dense Video-Sparse 预测（page 4）

![Figure 2](../../extracted/motus/pages/p04.png)

**原文 caption**：Action-Dense Video-Sparse Prediction. The sampling rates for video frames and actions differ.

**这张图讲什么**：一条时间轴上，下方采样的 action 密（48 步），上方采样的 video frame 稀（8 帧）。直观说明 video token 数远少于 action token 数的设置——这是为了纠正"video token 多导致 attention 偏向 video 预测、弱化 action"的问题。把视频率压到动作率的 1/6，让两边 token 平衡。

## Figure 3 — Latent Action VAE（page 5）

![Figure 3](../../extracted/motus/pages/p05.png)

**原文 caption**：The Latent Action VAE.

**这张图讲什么**：光流→latent action 的管道：DPFlow 算光流→转 RGB 图→DC-AE 编码成 4×512 token→轻量 encoder 压到 14 维 latent action（对齐机器人动作尺度）。同时 DC-AE Decoder 能从 latent action 重建光流做自监督。这是 Motus 能用无动作标签视频预训练 action 专家的根基。

## Figure 4 — 六层数据金字塔（page 6）

![Figure 4](../../extracted/motus/pages/p06.png)

**原文 caption**：The Embodied Data Pyramid categorizes data into six levels, from Level 1 at the base to Level 6 at the top. Data quantity decreases from bottom to top, while data quality increases. The order of Levels 3 and 4 may sometimes vary.

**这张图讲什么**：金字塔从底到顶：Level1 Web Data（规模最大质量最低）→ Level2 Egocentric Human → Level3 Synthetic Data → Level4 Task-Agnostic → Level5 Multi-Robot Trajectory → Level6 Target-Robot（规模最小质量最高）。底层数量大但无动作标签（靠 latent action 学），顶层小但精（直接对齐目标本体）。回答了"怎么用异构数据"。

## Figure 5 — 真机任务定义与可视化（page 8）

![Figure 5](../../extracted/motus/pages/p08.png)

**原文 caption**：Task Definitions and Visualizations. For each task, we describe its language instruction and definitions of each sub-task.

**这张图讲什么**：三个长程任务的子任务分解：Touch Instructed Keyboard（按屏幕字母按键）、Brew Coffee using Coffee Maker（抓杯倒豆→放杯→按键）、Put Bread into Oven（开门→放面包→关门→按键）。强调 Motus 评测的是长程多子任务能力，用 partial success rate 量化。

## Figure 6 — RoboTwin 消融（三阶段贡献）（page 8）★ 关键

![Figure 6](../../extracted/motus/pages/p08.png)

**原文 caption**：Ablation in RoboTwin 2.0 Randomized Multi-task Setting. The figure presents the total success rates (%) of the original Motus (Stage 2 Pretrain) and its two variants: Without Pretrain and Stage 1 Pretrain.

**这张图讲什么**：三档预训练对比柱状图（Clean / Randomized 两场景）：w/o pretrain → stage1 pretrain → stage2 pretrain。Randomized 场景从 77.00% → 81.86% → 87.02%（+10.02%）；Clean 从 77.56% → 82.26% → 88.66%（+11.10%）。证明 **Stage2（unified training with latent actions）的增量贡献最大**，光流 latent action 预训练是真涨点源头。

## Figure 7 — VGM 模式可视化（Agilex-Aloha-2）（page 15）

![Figure 7](../../extracted/motus/pages/p15.png)

**原文 caption**：Visualization of Motus's VGM mode on Agilex-Aloha-2.

**这张图讲什么**：Motus 在 VGM 模式（只生成视频，给定 o_t + ℓ）下，在 Agilex-Aloha-2 真机数据上的生成帧序列。证明统一模型在"只做视频生成"这一模式上质量也没退化——这是 UniDiffuser 调度的好处，五种模式共享一个权重。

## Figure 8 — 真机任务执行（2 机器人 9 任务）（page 16）

![Figure 8](../../extracted/motus/pages/p16.png)

**原文 caption**：Demonstrations of Motus for real-world tasks execution featuring 2 robots and 9 tasks.

**这张图讲什么**：真机执行可视化合集：AC-One 上 5 个任务（磨咖啡豆/按键盘/咖啡机冲咖啡/放方块入盘/水壶浇花）+ Agilex-Aloha-2 上 4 个任务（取水/放黄方块/放面包入烤箱/叠熊纹毛巾）。是 Table 3 真机结果的视觉佐证，跨本体泛化的证据。

## Figure 9 — VGM 模式可视化（AC-One）（page 17）

![Figure 9](../../extracted/motus/pages/p17.png)

**原文 caption**：Visualization of Motus's VGM mode on AC-One.

**这张图讲什么**：和 Figure 7 对应，VGM 模式在另一台机器人 AC-One 上的视频生成结果。两张图共同证明：统一模型在两种不同本体的真机数据上，VGM（纯视频生成）模式都工作良好。

## Figure 10 — World Model 模式可视化（Agilex）（page 20）★ 关键

![Figure 10](../../extracted/motus/pages/p20.png)

**原文 caption**：Visualization of Motus's World Model Mode on Agilex-Aloha-2 Dataset.

**这张图讲什么**：World Model 模式（给定 o_t + a_{t+1:t+k}，生成未来视频）的 Pred. vs GT 帧对齐。5 组 Pred./GT 对照，验证给定真实动作时模型能准确预测未来视频——这是 IDM/VLA 反过来的方向，证明统一模型在"动作→视频"这一条件方向也成立。配合 Table 6 的 FID 9.46 / FVD 49.28 量化。

## Figure 11 — World Model 模式可视化（AC-One）（page 21）

![Figure 11](../../extracted/motus/pages/p21.png)

**原文 caption**：Visualization of Motus's World Model Mode on AC-One Dataset.

**这张图讲什么**：Figure 10 在 AC-One 平台的对应版。World Model 模式在两台机器人上都给 Pred./GT 对照。Table 6：AC-One FID 12.96 / FVD 73.13 / SSIM 0.846，弱于 Agilex 但仍可用。说明统一模型跨本体视频预测能力稳健。

## Figure 12 — Video-Action 联合预测模式（真机推理）（page 22）★ 关键

![Figure 12](../../extracted/motus/pages/p22.png)

**原文 caption**：Visualization of Motus's Video-Action Joint Prediction Model mode during Real-World Inference.

**这张图讲什么**：Joint 模式（同时生成未来视频 + 动作）在真机推理时的可视化。这是 DreamZero 式的 WAM 用法，但 Motus 是把 video 和 action 用各自独立 timestep 去噪（都从 Tτ 降到 0），区别于 DreamZero 的共享 timestep。是 Motus 五模式里最接近 DreamZero 的一个，但调度机制不同。

---

## 用法

- 想看模型长什么样：Figure 1（MoT 架构）、Figure 3（latent action VAE）
- 想理解核心机制：Figure 2（action-dense video-sparse）、Figure 4（数据金字塔）、Figure 6（消融）
- 想看五模式效果：Figure 7/9（VGM）、Figure 10/11（World Model）、Figure 12（Joint）
- 想看真机结果：Figure 5（任务定义）、Figure 8（执行可视化）
