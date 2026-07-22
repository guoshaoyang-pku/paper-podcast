# WAM Survey — 论文重要图

> 配套 `card.json` / `card.md`。综述共 7 张 Figure(主体是分类图和路线图),全部列出。每张图给出:所在页、原文 caption(精简)、我们写的"这张图到底在讲什么"。
> 页图存在 `extracted/wam-survey/pages/pXX.png`,可对照查看。

## Figure 1 — WAM 时间演化与架构 taxonomy(page 2)★ 最重要

![Figure 1](../../extracted/wam-survey/pages/p02.png)

**原文 caption**:Temporal evolution and taxonomy of representative works on World Action Models (WAMs). The left branch illustrates the progression of Joint WAM architectures, which tightly couple world prediction and action generation, showing a divergence into Autoregressive and Diffusion-based representation schemes, with the continuous approach further bifurcating into Unified Stream and Multi-Stream backbones. The right branch summarizes the development of Cascaded WAM pipelines, where world modeling and action execution are primarily decoupled, evolving along Explicit and Implicit representation alignment trajectories.

**这张图讲什么**:全文最该看的图。左分支 Joint WAM 时间线:从 GR-1(2024)开始,2025-2026 爆发,分 Autoregressive(GR-2/CoTVLA/WorldVLA/VLA-JEPA/ℱ1)和 Diffusion-based(VideoVLA/Cosmos-Policy/DreamZero/LingBot-VA/Fast-WAM/GigaWorld-Policy),后者再分 Unified Stream 和 Multi-Stream。右分支 Cascaded WAM 时间线:UniPi(2023)起,沿 Explicit(UniPi/VLP/AVDC/Gen2Act/Veo-Act)和 Implicit(VPP/LAPA/mimic-video/S-VAM)发展。这张图是整个综述的导航地图——一眼看清 WAM 全景和两种架构范式的演化路径。任何新 WAM 论文都能在这张图上找到位置。

## Figure 2 — WAM 路线图(四大维度)(page 4)

![Figure 2](../../extracted/wam-survey/pages/p04.png)

**原文 caption**:The comprehensive roadmap and taxonomy of World Action Models (WAMs) reviewed in this survey. The literature is systematically categorized into four core dimensions: background (Sec. 3), architecture (Sec. 4), training data (Sec. 5), and evaluation protocols (Sec. 6).

**这张图讲什么**:综述的四大维度路线图:Background(VLA/WM 脉络)/ Architecture(Cascaded Explicit+Implicit / Joint Autoregressive+Diffusion-based)/ Training Data(Robot-centric Teleop / UMI Human / Simulation / Human Data)/ Evaluation(World Model: Visual Fidelity+Physical Commonsense+Action Plausibility / Action Policy: General+Bimanual+Mobile+Contact+Real-Device)。每个分支下列具体方法/数据集/benchmark。是综述结构的可视化,告诉你"想找 X 该去哪节"。

## Figure 3 — WAM 概念定义与对比(page 5)

![Figure 3](../../extracted/wam-survey/pages/p05.png)

**原文 caption**:Conceptual definition and comparison of World Action Models (WAMs). The left panel contrasts the input-output formulations of VLA models, WAMs, and standard World Models (WMs), highlighting WAM's capability to jointly predict actions and future observations. The right panel illustrates the conceptual scope of WAMs relative to other paradigms such as Video Action Models (VAMs) and Video Policies.

**这张图讲什么**:两张子图。左:VLA(输入 o,l → 输出 a)/ WM(输入 o,a → 输出 o')/ WAM(输入 o,l → 输出 a + o')的输入输出对比,WAM 同时出动作和未来观测。右:WAM 与 VAM/Video Policy/AWM 的概念包含关系——WAM 是 modality-agnostic 超集(不只视频,含点云/触觉/力),Video Policy 限于视频 backbone 且不强制预测承诺。这张图把 WAM 的概念边界钉死,是 Section 2 定义的可视化。关键判断:WAM 的定义性特征不是"用视频 backbone"而是"预测承诺"——必须显式或隐式预测未来状态作为推理输出。

## Figure 4 — World Model 如何辅助 VLA(page 12)

![Figure 4](../../extracted/wam-survey/pages/p12.png)

**原文 caption**:Schematic overview of world models for VLA learning and evaluation. World models can support (a) imitation learning by generating or filtering training trajectories, (b) reinforcement learning by enabling imagined interaction and reward-guided policy optimization, (c) reward modeling by producing reward signals from learned dynamics or future outcomes, and (d) policy evaluation.

**这张图讲什么**:WM 作为 VLA 的外生工具(非集成进 policy)的四种用法:①imitation learning(生成/过滤训练轨迹);②reinforcement learning(想象交互 + reward-guided 优化);③reward modeling(从动态/未来产 reward);④policy evaluation(作 benchmark)。这四条是 WAM 出现前的 WM-in-robotics 主流,和 Section 4 的"WM 集成进 policy 成 WAM"形成对照——把 WM 从离线监督变成内部预测核心是 WAM 范式的关键转变。

## Figure 5 — Cascaded WAM 结构示意图(page 15)

![Figure 5](../../extracted/wam-survey/pages/p15.png)

**原文 caption**:Schematic comparison of cascaded WAM structures. 1(a) Learned Action: a world model generates an explicit pixel-space future plan, mapped to actions by a learned IDM. 1(b) Geometric Extraction: the explicit visual plan converted to actions via geometric extraction. 2(a) Latent Representation: intermediate planning carrier is a latent future representation, action model decodes from it.

**这张图讲什么**:Cascaded WAM 的三种子模式:1(a) Learned Action(UniPi/VLP/RoboEnvision——视频生成出像素未来,学习 IDM 提取动作);1(b) Geometric Extraction(AVDC/Im2Flow2Act——视频未来转光流,几何解析出 SE(3) 动作,无需动作标注);2(a) Latent Representation(VPP/LAPA/mimic-video——中间载体是 latent 未来表示,不生成像素,动作模型从 latent 解码)。这张图是 Cascaded 二分法(Explicit Pixel vs Implicit Latent)+ Explicit 二分(Learned vs Geometric)的可视化,对应 Section 4.1。

## Figure 6 — Joint WAM (Diffusion) 结构 taxonomy(page 21)

![Figure 6](../../extracted/wam-survey/pages/p21.png)

**原文 caption**:Taxonomy of the main architectural patterns in the diffusion-based joint WAMs. 1(a) Unified Stream: World and action integrated within one single DiT backbone. 2(a) Multi-Stream – Cross-Attention Coupled: separate video and action DiTs coupled through explicit cross-attention. 2(b) Multi-Stream – Hidden-State Coupling. 2(c) Multi-Stream – Shared Representation.

**这张图讲什么**:Diffusion-based Joint WAM 的四种子模式:1(a) Unified Stream(单 DiT 联合去诺,如 DreamZero/Cosmos-Policy/PAD/UWM——最紧耦合);2(a) Multi-Stream Cross-Attention(视频/动作双 DiT,cross-attention 耦合,如 CoVAR/LDA-1B/DUST);2(b) Hidden-State Coupling(视频 DiT 的隐藏状态条件化动作 DiT);2(c) Shared Representation(统一编码器先融合再分头解码)。这张图把 Joint WAM 最复杂的部分(Diffusion 路线的结构耦合)讲清,对应 Section 4.2.2。Unified vs Multi-Stream 是核心二分:前者紧但重,后者灵活但耦合机制要设计。

## Figure 7 — 数据生态景观图(page 26)

![Figure 7](../../extracted/wam-survey/pages/p26.png)

**原文 caption**:An overview of the embodied data landscape for training World Action Models, mapped across Transfer Difficulty (Y-axis) and Scaling Difficulty (X-axis).

**这张图讲什么**:训练 WAM 的四类数据在"迁移难度(Y)"vs"规模化难度(X)"二维平面的分布。Robot Teleop 迁移易但规模化难(成本高);UMI Human Demo 迁移中等、规模化中等;Simulation 迁移难(sim-to-real gap)但规模化易(自动生成);Internet Ego Video 迁移难但规模化最易(百万小时级,无动作标注)。这张图直观说明为什么 WAM 能利用 video-only 数据是个关键优势——它打开了 Internet Ego Video 这条最大规模的数据通路,是 WAM 相对 VLA 的数据生态优势。对应 Section 5。

---

## 用法

- 想看 WAM 全景导航:Figure 1(时间线+taxonomy,最该看)
- 想找某主题在哪节:Figure 2(四大维度路线图)
- 想理解 WAM 概念边界:Figure 3(VLA/WM/WAM 对比 + VAM/Video Policy 关系)
- 想看 Cascaded 架构:Figure 5(三种子模式)
- 想看 Joint Diffusion 架构:Figure 6(Unified/Multi-Stream 四子模式)
- 想理解 WM 外生工具 vs WAM 内生范式:Figure 4
- 想看数据生态:Figure 7(四类数据迁移-规模化平面)
