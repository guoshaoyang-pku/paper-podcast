# WAM Survey — 结构化卡片

> 论文:**World Action Models: The Next Frontier in Embodied AI**(Fudan University · Shanghai Innovation Institute · NUS, arXiv 2605.12090, 2026-05-12)
> 配套:`card.json`(机器可读) · `architecture.md`(架构图) · `podcast.md`(口播稿)

## 一句话定位

第一份系统梳理 World Action Model(WAM)的综述:把"统一预测未来状态和动作"这族方法正式定义,拆成 Cascaded vs Joint 两大架构范式,沿生成模态/条件机制/动作解码三维细分,并覆盖数据生态、评测协议、开放挑战——给 WAM 这个快速膨胀的领域一张可导航的地图。

## 这篇综述整理了什么

**背景**:VLA 只学 observation→action 映射、不显式建模世界动态,泛化受限;而把 world model 集成进 policy 的方法近一年爆发式增长(DreamZero/LingBot-VA/Cosmos-Policy/Fast-WAM 等),但文献碎片化——架构、学习目标、应用场景各说各话,缺统一概念框架。

**综述做的事**:正式定义 WAM、厘清与 VLA/Video Policy/AWM/VAM 的边界,给出结构化分类法(Cascaded vs Joint),梳理数据生态和评测协议,并指出开放挑战,给进入这个领域的人一张可导航的地图。

**为什么 prior work 不够**:之前没有系统综述把 WAM 作为独立范式梳理。各 WAM 论文用不同术语、不同分类轴,新进入者难以判断:显式视觉预测是否必需?Cascaded 和 Joint 架构在下游控制上到底差在哪?数据混合的最优配比是什么?评测上 PSNR/FVD 只测视觉不测物理,task success 只测动作不测视觉-动作因果一致性——这套脱节让 WAM 的真实能力被误判。

## WAM 的概念契约

| 范式 | 输入 | 输出 | 目标函数 |
|---|---|---|---|
| VLA | o, l | a | L = E[-log p(a|o,l)] |
| World Model | o, a | o' | L = E[-log p(o'|o,a)] |
| **WAM** | **o, l** | **a + o'** | **L = E[-log p(o',a|o,l)]** |

WAM 必须满足两准则:①Forward Predictive Modeling(生成或利用可量化的未来状态 o' 表示);②Coupled Action Generation(动作 a 严格对齐到预测的未来状态 o')。

**概念边界**:WAM 是 modality-agnostic 超集(不只视频,含点云/触觉/力);VAM 限于视频对齐动作;Video Policy 只继承视频 backbone 表示但不强制预测承诺;AWM 是早期术语,WAM 重定位为 Agent(World 和 Action 对等)而非 simulator。

### 两种因子分解(架构层 protocol)

```
【Cascaded WAM】p(o',a|o,l) = p(a|o',o,l) · p(o'|o,l)
  先合成未来状态表示,再从中派生动作;两阶段分离训练。

【Joint WAM】直接建模 p(o',a|o,l)
  状态预测和动作生成在统一模型里联合优化。
  - Autoregressive: 序列化成 token 左到右因果解码
    (Explicit Decoupled / Unified Discrete / Predictive Latent)
  - Diffusion-based: 多步去噪并行生成
    (Unified Stream: Explicit/Implicit Future Prediction;
     Multi-Stream: Cross-Attention / Hidden-State / Shared Representation)
```

**三个易混坑**:①language 在 Cascaded 里常作视频生成条件,在 Joint 里常 cross-attention 注入,不是统一序列拼接;②多视角处理随架构变(有的通道拼接,有的独立编码);③训练用未来状态作监督,推理时是否生成未来状态因方法而异(Fast-WAM/GigaWorld 推理跳过视频生成,LingBot-VA/Cosmos 必须生成)。

## 数据生态(详见 architecture.md)

| 数据源 | 特点 | 代表 |
|---|---|---|
| Robot Teleop | 高精度,规模化难 | OXE/DROID/AgiBot World/RoboMIND/ARIO |
| UMI Human Demo | 便携,跨本体潜力 | UMI/FastUMI/DexUMI/UMI on Legs/HoMMI |
| Simulation | 自动生成,sim-to-real gap | MimicGen/RoboCasa/RoboTwin 2.0/DexMimicGen |
| Internet Ego Video | 百万小时级,无动作标注 | EPIC-KITCHENS/Ego4D/HowTo100M/Ego-Exo4D |

**关键洞察**:WAM 能利用 video-only 数据(video prediction objective 不需 action),打开 Internet Ego Video 这条最大规模数据通路——这是 WAM 相对 VLA 的数据生态优势。

## 架构分类法(详见 architecture.md)

- **顶层二分**:Cascaded(两阶段分离)vs Joint(单模型联合)
- **Cascaded 细分**:Explicit Planning(Learned Action / Geometric Extraction)/ Implicit Planning(Latent Representation)
- **Joint 细分**:Autoregressive(Explicit Decoupled / Unified Discrete / Predictive Latent)/ Diffusion-based(Unified Stream: Explicit/Implicit Future Prediction;Multi-Stream: Cross-Attention / Hidden-State / Shared Representation)
- **核心 trade-off**:Cascaded 训练简单但两阶段误差累积;Joint 对齐紧但训练复杂、推理重

### 数值 sense

| 项 | 值 |
|---|---|
| 综述方法数 | 100+ 篇 WAM + 数十背景/数据/评测;Figure 1 时间线列 ~60 代表作 |
| 分类深度 | 2 层主分类 + 2-3 层细分 |
| backbone 跨度 | 视频扩散:Wan2.1-14B(DreamZero)/Wan2.2-5B(LingBot-VA/Fast-WAM)/CogVideoX-5B/Cosmos-Predict2-2B/LTX-Video-2B/Sora 2;AR VLM:PaliGemma-3B/Qwen3-VL-2B/VILA-U-7B/Chameleon |
| 参数跨度 | GR-1 195M(最小)~ DreamZero 14B(最大);主流 2-7B |
| 数据源 | 4 大类(Robot Teleop / UMI Human / Simulation / Internet Ego Video) |
| 评测维度 | 3:Visual Fidelity / Physical Commonsense / Action Plausibility |
| action policy benchmarks | General(MetaWorld/RLBench/CALVIN/LIBERO)+ Bimanual/Humanoid + Mobile + Contact/Deformation + Real-Device |
| latency 参考 | DreamZero 7Hz(优化极限)vs 非 generative VLA 50Hz |
| 开放挑战 | 7 个:架构对照 / 多模态物理状态 / 数据混合 / 推理延迟 / 评测方法论 / 长程层次 / 安全 |

## 核心 taxonomy(7 个)

1. **顶层二分:Cascaded vs Joint** — Cascaded 先合成未来状态再派生动作(两阶段分离);Joint 联合优化 p(o',a|o,l)。Cascaded 训练简单但误差累积;Joint 对齐紧但推理重。
2. **Cascaded 内部:Explicit Pixel vs Implicit Latent** — Explicit 用像素未来(Learned IDM 或 Geometric Extraction);Implicit 用 latent 表示。Geometric Extraction(AVDC/Im2Flow2Act)无需动作标注。
3. **Joint 内部:Autoregressive vs Diffusion-based** — AR 序列解码适合长horizon因果但慢;Diffusion 多步去诺并行高频但计算密集。
4. **Joint AR 内部:三种表示** — Explicit Decoupled(异构分头)/ Unified Discrete(共享词表)/ Predictive Latent(连续 latent,如 VLA-JEPA)。
5. **Joint Diffusion 内部:Unified Stream vs Multi-Stream** — Unified 单 DiT 紧耦合(DreamZero/Cosmos);Multi-Stream 多分支耦合(Cross-Attention/Hidden-State/Shared)。
6. **数据生态四源** — Robot Teleop(精度高规模难)/ UMI(便携)/ Simulation(自动 sim-to-real gap)/ Internet Ego Video(百万小时无动作)。WAM 能用 video-only 是核心数据优势。
7. **评测三维** — Visual Fidelity / Physical Commonsense / Action Plausibility。综述指出三维度脱节,缺联合评测指标。

## 核心发现(8 个)

1. **WAM 是 VLA 的概念继任者**:统一 p(o',a|o,l),术语从 AWM(simulator)转向 WAM(Agent)。应被视作机器人 foundation model 新主类。
2. **Cascaded vs Joint 核心trade-off是训练简单vs对齐紧**:工业部署重速度倾向 Cascaded,重精度倾向 Joint。
3. **显式像素预测是否必需仍是开放问题**:Implicit Latent / Predictive Latent 仍有效,未来可能走向 latent-only 跳过昂贵像素生成。
4. **WAM 能用 video-only 数据是相对 VLA 的核心数据优势**:打开 Internet Ego Video 百万小时级通路,scaling 潜力大于 VLA。
5. **推理 latency 是 WAM 部署最大障碍**:DreamZero 7Hz vs VLA 50Hz;未来方向 task-adaptive predictive fidelity。
6. **评测方法论脱节**:PSNR/FVD 测视觉但允许物理错误;task success 测动作但忽略因果。需联合指标(Counterfactual Consistency / Foresight-Conditioned Success)。
7. **多模态物理状态是未开发前沿**:触觉/力/声在像素不可见,需 modality-adaptive prediction。
8. **架构耦合缺对照实验**:领域被"架构时尚"驱动,需 rigorous ablation 走向 principled design。

## Insights

1. WAM 的定义性特征不是"用视频 backbone"而是"预测承诺"——必须显式或隐式预测未来状态 o' 作为推理输出。Video Policy 只继承 backbone 表示但不强制预测,不算 WAM(p5 Figure 3)。
2. Cascaded vs Joint 二分法是整个综述最有价值的贡献:它把所有 WAM 挂进统一框架,让读者能从任意新论文快速定位设计空间位置(p14)。
3. WAM 能用 video-only 数据是相对 VLA 的根本数据优势——可能比单个模型架构创新更重要(p26 Figure 7)。
4. 推理 latency 是 WAM 阿喀琉斯之踵;综述提出 task-adaptive predictive fidelity——识别任务所需最小充分世界模型,而非一味高保真加速(p43)。
5. 评测方法论脱节是 WAM 真实能力被误判的根源;综述提出联合指标是评测范式需补的核心空白(p43)。
6. 架构耦合缺对照实验是领域最大方法论空白——这本身是对当前 WAM 研究节奏的批评(p41)。

## vs 同类工作

- **vs 各 WAM 单篇论文**:那些论文各自提架构但用不同术语和分类轴;本综述用统一 Cascaded/Joint 框架全挂进设计空间,让横向比较成为可能。
- **vs 现有 VLA 综述**:VLA 综述聚焦 observation→action 映射,本综述把 WAM 作为独立范式梳理,补了"world modeling 集成进 policy"这个空白的系统综述。
- **vs 现有 world model 综述**:WM 综述多聚焦"WM 作 simulator/reward"的外生工具角色;本综述聚焦"WM 集成进 policy 成 WAM"的内生范式,是近一年爆发的新方向。
- **vs wam-vs-vla 鲁棒性研究**:那份研究在评测协议层面对比 WAM/VLA;本综述在架构/概念层面梳理 WAM 内部分类。两者互补。

## 局限

**论文自承**:
- 架构耦合缺对照实验,本综述只能罗列各范式,无法给"Cascaded vs Joint 在匹配条件下谁更优"的实证结论(p41)。
- 评测方法论脱节,本综述整理现有评测但未提出新联合评测指标,只指出方向(p43)。
- 多模态物理状态(触觉/力/声)是未开发前沿,本综述覆盖的 WAM 几乎全预测 RGB(p42)。
- 数据混合设计原理不清,本综述罗列四类数据但无法给最优配比(p42)。

**我们读出**:
- 综述覆盖方法截至 2026 年初,但 WAM 领域迭代极快(每月新论文),部分最新方法可能未纳入;Figure 1 时间线已显示 2026 年仍在爆发,综述时效性有限。
- 分类法本身是人为构造,Cascaded vs Joint 的边界在 hybrid 方法上模糊(如 π0.7 用 VLA 替 IDM 属 Cascaded 还是 Joint?MOTUS 用视频 backbone 但 action 走 VLM expert 算不算 WAM?)——综述承认 MOTUS 被排除因"action 不走 world backbone",但这类边界判断有主观空间。
- 综述是叙述性梳理,缺定量 meta-analysis(如各架构在统一 benchmark 上的成功率分布),读者难以直接判断哪种架构实证上更优——这需要像 wam-vs-vla 那样的对照研究配合。

## 可复现性

- Homepage:https://openmoss.github.io/Awesome-WAM
- Github:https://github.com/OpenMOSS/Awesome-WAM
- 性质:综述论文,无代码/权重;提供 Awesome-WAM 论文清单仓库供追踪

## 论文重要图(详见 `figures.md`)

| 图 | 页 | 重要性 | 一句话 |
|---|---|---|---|
| [Figure 1](../../extracted/wam-survey/pages/p02.png) | 2 | key | WAM 时间演化与架构 taxonomy,全文导航地图,最该看 |
| [Figure 2](../../extracted/wam-survey/pages/p04.png) | 4 | key | 四大维度路线图(背景/架构/数据/评测) |
| [Figure 3](../../extracted/wam-survey/pages/p05.png) | 5 | key | WAM 概念定义,VLA/WM/WAM 对比 + VAM/Video Policy 边界 |
| [Figure 5](../../extracted/wam-survey/pages/p15.png) | 15 | key | Cascaded WAM 三子模式(Learned/Geometric/Latent) |
| [Figure 6](../../extracted/wam-survey/pages/p21.png) | 21 | key | Joint Diffusion 四子模式(Unified/Multi-Stream) |
| [Figure 4](../../extracted/wam-survey/pages/p12.png) | 12 | supportive | WM 作 VLA 外生工具的四用法 |
| [Figure 7](../../extracted/wam-survey/pages/p26.png) | 26 | supportive | 数据生态景观(迁移-规模化平面) |

## 标签

`WAM` `survey` `taxonomy` `Cascaded WAM` `Joint WAM` `world model` `VLA` `video prior` `Fudan` `MOSS`
