# World Action Models: The Next Frontier in Embodied AI

2026-05-12，Fudan University 的 Siyin Wang 等人发布的论文，标题是 World Action Models: The Next Frontier in Embodied AI。

## 开场：这份综述要给你什么

这份综述是第一份把 World Action Model（WAM）作为一个**独立范式**系统梳理的工作——它正式定义 WAM、厘清它和相邻概念的边界、给出结构化分类法、梳理数据生态和评测协议、指出开放挑战。

为什么要做这件事？因为过去一年，WAM 这个方向爆发式增长。DreamZero、LingBot-VA、Cosmos-Policy、Fast-WAM、GigaWorld-Policy，每个月都有新论文。但文献碎片化严重——大家用不同术语，有人叫 Video Policy，有人叫 Action World Model，有人叫 World Action Model；分类轴也各说各话，有人按 backbone 分，有人按训练目标分。一个新进入这个领域的人，很难判断：显式视觉预测是不是必需的？Cascaded 和 Joint 架构在下游控制上到底差在哪？数据混合的最优配比是什么？评测上，PSNR、FVD 这些指标只测视觉不测物理，task success 只测动作不测视觉-动作的因果一致性——这套脱节让 WAM 的真实能力被误判。

这份综述要做的，就是把这些混乱和空白显式化，给一张能导航的地图。所以这篇口播稿我会讲得像一份导览：先讲 WAM 是什么、不是什么，再讲分类框架怎么用，最后讲边界和空白在哪。

## WAM 是什么：一个概率定义

先把 WAM 的概念契约讲清楚。这份综述用概率定义把 WAM 和相邻概念划清边界。

VLA，也就是 Vision-Language-Action 模型，学的是 observation 到 action 的映射，目标是预测动作 a 给定观测 o 和语言 l，写成 p(a|o,l)。它只关心动作，不显式预测世界会怎么变。

World Model，学的是给定当前状态 o 和动作 a，预测下一个状态 o'，写成 p(o'|o,a)。它只关心世界演化，不直接出动作。

WAM，World Action Model，把这两者合起来。它统一建模 p(o', a|o, l)——给定当前观测和语言，同时预测未来状态和动作。这就是 WAM 的概念契约。

但要算 WAM，光有这个目标函数还不够。综述定了两条准则。第一，Forward Predictive Modeling，模型必须生成或利用一个可量化的未来状态 o' 的表示——可以是显式的像素视频、latent、光流、点云，也可以是隐式的 physics-grounded latent。第二，Coupled Action Generation，动作 a 必须严格对齐到预测的未来状态 o'，可以是联合概率输出，也可以是 cascaded 或 joint 架构里的 policy conditioning。

这里有个关键判断我特别想强调。WAM 的定义性特征是"预测承诺"。一个模型如果只是继承了视频生成模型的表示能力，但只把观测映射到动作 p(a|o)，没有显式预测未来状态作为推理输出的一部分，那它只是 Video Policy，不算 WAM。这个区分把"装上视频 backbone 就叫 WAM"的滥用挡住了。

还有个术语细节。早期文献用 AWM，Action World Model。综述刻意改用 WAM，原因是：AWM 里名词是 World Model，定位是 augmented simulator；WAM 里 World 和 Action 对等，定位是 Agent。这个术语转变反映了一个战略重定位——WAM 不是辅助工具，是机器人 foundation model 的新主类。

## 怎么用这张地图：Cascaded vs Joint 二分法

现在讲分类框架怎么用。这是整份综述最有价值的贡献。

所有 WAM 方法，综述用一个非常干净的顶层二分法挂进统一框架：Cascaded WAM 还是 Joint WAM。

Cascaded WAM 把目标函数因子分解：p(o',a|o,l) 等于 p(a|o',o,l) 乘以 p(o'|o,l)。意思是先合成未来状态 o' 的表示，再从这个未来状态派生动作 a。两个阶段是分离训练的——世界模型管世界演化，动作模型管动作解码，各管各的。

Joint WAM 不分解，直接建模 p(o',a|o,l)。状态预测和动作生成在同一个模型里联合优化，强制模型内部化世界动态和动作控制的因果依赖。

这个二分法为什么有价值？因为它直接对应"世界和动作怎么耦合"这个最根本的架构选择。Cascaded 的耦合是松的——世界模型和动作模型解耦，训练简单，世界模型不用管机器人运动学，动作模型不用管长程场景预测。但代价是两阶段误差累积，未来状态表示和动作对齐弱。Joint 的耦合是紧的——联合优化让对齐很紧，但训练复杂、推理重。

任何一篇新 WAM 论文，你只要看它"世界和动作是分两阶段还是一起训"，就能立刻定位它在设计空间的位置，并判断它的架构选择带来什么 trade-off。这比按 backbone 分（autoregressive 还是 diffusion）更根本，因为 backbone 只是实现细节，耦合方式才是架构本质。

## Cascaded WAM 内部：显式像素还是隐式 latent

Cascaded 内部再分两层。Explicit Planning 用像素级未来作中间载体——视频、光流、点云。Implicit Planning 用 latent 未来表示，不生成像素。

Explicit 再分两种动作提取方式。第一种是 Learned Action Extraction，学一个 inverse dynamics model 从视频未来提取动作，像 UniPi、VLP、RoboEnvision。问题是 IDM 提取动作有 ill-posed 风险——同一个视觉未来可能对应多种合理动作。第二种是 Geometric Extraction，把视觉未来转成光流或轨迹，几何解析出 SE(3) 动作。这个特别有意思，因为 AVDC 和 Im2Flow2Act 这类方法**不需要任何动作标注**——它们从视频未来几何地算出动作，训练时只用视频。这打开了无动作标注数据的使用通路。

Implicit 用 latent 未来表示，像 VPP、LAPA、mimic-video。中间载体不是像素而是 latent 向量，动作模型从 latent 解码。比 Explicit 高效，因为不用生成像素，但 latent 表示的质量决定动作精度。

## Joint WAM 内部：Autoregressive 还是 Diffusion

Joint WAM 内部分两条生成路线。

Autoregressive 路线把未来状态和动作序列化成 token，左到右因果解码。适合长 horizon 因果一致性，但序列解码慢、有 compounding error。内部再分三种表示范式。Explicit Decoupled 是异构格式分头输出，像 GR-1、GR-2 用双分支一个出视觉 patches 一个出连续动作，灵活但跨模态 grounding 浅。Unified Discrete 是全量化进共享 LLM 词表，同一个 next-token head 出所有模态，像 CoT-VLA、WorldVLA、ℱ1，统一但离散控制高方差。Predictive Latent 放弃显式 token，在连续 latent 空间自回归，像 VLA-JEPA，高效抽象但放弃像素可解释性。

Diffusion 路线用多步去噪并行生成状态和动作，能高频执行适合闭环控制，但计算密集。内部再分 Unified Stream 和 Multi-Stream。

Unified Stream 是单 DiT 联合去噪，世界和动作在同一个主干里。最紧耦合，同步最强。再分 Explicit Future Prediction（未来观测作直接预测目标，像 DreamZero、Cosmos-Policy、PAD、UWM）和 Implicit Future Prediction（未来信息通过辅助 future token 引入，latent 对齐，像 FLARE、FRAPPE）。DreamZero 就是 Unified Stream Explicit 的代表——Wan2.1-14B backbone 加轻量 state/action encoder，联合去噪 video latent 和 action latent，闭环用 KV-cache 替换消除误差累积。

Multi-Stream 是多分支耦合，世界和动作在不同的 DiT 里，通过某种机制耦合。再分 Cross-Attention Coupled（像 CoVAR、LDA-1B、DUST，双 DiT 用 cross-attention 交换信息）、Hidden-State Coupling（视频 DiT 的隐藏状态条件化动作 DiT）、Shared Representation（统一编码器先融合再分头解码）。LingBot-VA 和 Fast-WAM 属于 Mixture-of-Transformers 这种 Multi-Stream 变体。

Unified vs Multi-Stream 的核心 trade-off：Unified 紧耦合但单主干承载所有计算重；Multi-Stream 灵活可异构 schedule 但耦合机制要专门设计。工业部署如果重推理速度可能倾向 Multi-Stream（可对动作流用更轻 backbone），如果重精度可能倾向 Unified。

## 给你一个数值 sense：这个领域多大

光说"综述覆盖很多论文"没感觉，我把规模念细一点。

综述覆盖 100 多篇 WAM 方法，加上数十篇背景、数据、评测论文。Figure 1 的时间线列了大约 60 个代表作。从 2023 年的 UniPi 起步，2024 年 GR-1 开始上量，2025 到 2026 年爆发——每个月都有新论文。这本身说明 WAM 是个快速膨胀的领域，也意味着任何综述的时效性都有限。

backbone 跨度很大。视频扩散这边，从 LTX-Video-2B（GE-Act 用）到 Wan2.1-14B（DreamZero 用）都有，主流是 2 到 6B：Cosmos-Policy 2B、LingBot-VA 5.3B、Fast-WAM 6B。自回归 VLM 这边，PaliGemma-3B（π0 用）、Qwen3-VL-2B（VLA-JEPA 用）、VILA-U-7B（CoT-VLA 用）、Chameleon（WorldVLA 用）。参数跨度从 GR-1 的 195M 到 DreamZero 的 14B，主流是 2 到 7B。

数据生态分四类。Robot Teleop 是机器人遥操作数据，像 OXE、DROID、AgiBot World、RoboMIND，精度高但成本高、规模化难。UMI Human Demo 是便携手持设备采的人类示教，像 UMI、DexUMI、UMI on Legs，跨本体潜力大。Simulation 是仿真数据，像 MimicGen、RoboCasa、RoboTwin 2.0，自动生成、可域随机化，但有 sim-to-real gap。Internet Ego Video 是互联网第一人称视频，像 EPIC-KITCHENS、Ego4D、HowTo100M，百万小时级，但没有动作标注。

这里有个关键洞察。WAM 能利用 video-only 数据——它的 video prediction objective 不需要 action 标注就能学世界动态。这打开了 Internet Ego Video 这条最大规模的数据通路。VLA 必须有 action 标注才能学，用不了这些数据。所以 WAM 相对 VLA 有个根本的数据生态优势，scaling 潜力更大。这可能比单个模型架构创新更重要。

评测分三维度。Visual Fidelity 测视觉质量，指标有 PSNR、SSIM、LPIPS、FVD。Physical Commonsense 测物理常识，benchmark 有 VideoPhy、PhyGenBench、WorldScore、Physics-IQ。Action Plausibility 测动作合理性，有 WorldSimBench。action policy benchmarks 覆盖 General（MetaWorld、RLBench、CALVIN、LIBERO）、Bimanual/Humanoid（RoboTwin、BiGym、HumanoidBench）、Mobile、Contact/Deformation（SoftGym、TacSL）、Real-Device（RoboArena、Maniparena）。

记住这套数字，后面读 WAM 论文时可以拿它当坐标系：100+ 篇方法、Cascaded/Joint 二分、视频 backbone 2-14B、四类数据、三维评测。最该记住的是"video-only 数据是 WAM 相对 VLA 的根本数据优势"。

## 这张地图的边界：WAM 不是什么

讲完了 WAM 是什么、怎么分类，现在讲这张地图的边界——WAM 不是什么。这对避免过度外推很重要。

第一，WAM 不是"用视频 backbone 的 policy"。前面讲过，Video Policy 只继承视频 backbone 的表示能力但不强制预测承诺，不算 WAM。综述在 Table 1 里专门把 MOTUS 排除，原因是它的 action 走单独的 VLM expert 而不是 world model backbone——这违反了"动作必须从预测未来状态派生"这条准则。这种边界判断有主观空间，但综述至少把判断标准显式化了。

第二，WAM 不是万能鲁棒。从配套的 wam-vs-vla 鲁棒性研究看，WAM 在视觉扰动（噪声、光照、背景）上比 VLA 强，但在相机视角和机器人初始状态这种几何配置变化上和 VLA 一样弱。视频先验对"视觉外观变化"有效，对"几何配置变化"无效。所以"WAM 必然比 VLA 强"是过度外推。

第三，WAM 不是已经能工业部署。综述在开放挑战里专门点出推理 latency 是最大障碍——DreamZero 7Hz 是算法加系统优化的极限，仍远低于非 generative VLA 的 50Hz。WAM 的 latency tax 威胁闭环控制可行性。所以现在谈 WAM 替代 VLA 还早。

## 怎么用这份综述

这份综述该怎么用？我给你三个具体场景。

第一个场景：你读到一篇新 WAM 论文，想快速定位它。先看它世界和动作是分两阶段还是一起训——分两阶段是 Cascaded，一起训是 Joint。Cascaded 再看中间载体是像素还是 latent，Joint 再看是 autoregressive 还是 diffusion。三步就能定位它在设计空间的位置，然后判断它的架构选择带来什么 trade-off。

第二个场景：你在设计一个新 WAM，想判断该选哪种架构。先问你的优先级是什么。如果重推理速度、能容忍两阶段误差累积，倾向 Cascaded 或 Multi-Stream（可对动作流用轻 backbone）。如果重精度、要紧耦合，倾向 Joint Unified Stream。如果数据有限、想用 video-only 数据，选支持 video prediction objective 的设计。如果担心 compounding error，选带闭环 KV-cache 替换的（像 DreamZero）或 Predictive Latent（像 VLA-JEPA）。

第三个场景：你想判断 WAM 这个方向是否值得投入。看数据生态和开放挑战。数据上，WAM 能用 Internet Ego Video 这条百万小时级通路，scaling 潜力大于 VLA，这是根本优势。但开放挑战也明确：架构耦合缺对照实验（领域被"架构时尚"驱动）、推理 latency、评测方法论脱节、多模态物理状态未开发。这些是风险也是机会——投入 WAM 不是稳赢，但如果你能解决这些开放问题里的一个，就是有价值的贡献。

## 一个最值得记住的 insight

这份综述里有个判断，我建议你重点记：**WAM 的定义性特征是预测承诺，不是视频 backbone**。

这句话分量很重。它意味着判断一个模型是不是 WAM，不看它用什么 backbone，看它有没有显式或隐式地预测未来状态作为推理输出的一部分。一个用 VLM backbone 但加了未来状态预测目标的模型可以是 WAM；一个用视频 backbone 但只做 observation-to-action 映射的模型只是 Video Policy。

这个判断对实践很重要。它说明 WAM 是一个更广的范式，不是某个特定技术路线（视频 diffusion）的代名词——任何把"预测世界怎么变"和"决定动作"统一起来的模型都算。这意味着 WAM 的设计空间比"视频 backbone + action head"大得多，未来可能出现完全不用视频 backbone 的 WAM（比如纯 latent 的 VLA-JEPA 路线）。这给了领域一个更开放的发展方向，不被"必须用视频生成模型"这个技术选择锁死。

## 开放挑战：这张地图没覆盖什么

最后讲这份综述指出的七大开放挑战。这是地图的空白，也是未来工作的机会。

第一，架构耦合缺对照实验。各种结构策略（cascaded、joint diffusion、离散 token、隐式对齐）并存，但没有系统对照实验在匹配规模、数据、评测下比较它们。领域目前被"架构时尚"驱动而非 principled design。综述把这点列为第一开放挑战——这本身是对当前 WAM 研究节奏的一个批评。

第二，多模态物理状态表示。WAM 几乎全预测 RGB，但接触密集操作的关键物理信息——触觉分布、接触力、声学、材料 compliance——在像素空间不可见。未来需要 modality-adaptive prediction，有传感器时多模态预测，无时优雅降级到视觉。

第三，数据混合设计原理不清。非机器人数据（尤其人类视频）的作用到底是语义增强还是动态学习？最优配比是什么？综述提出要从经验超参调优转向信息论视角的数据混合设计。

第四，推理 latency。DreamZero 7Hz vs VLA 50Hz。综述提出 task-adaptive predictive fidelity——不追求高保真加速，而是识别任务所需的最小充分世界模型，按任务需求动态调预测精度。这是比单纯堆 CUDA 优化更深刻的加速方向。

第五，评测方法论脱节。视觉指标允许物理错误（悬浮物体得分高），动作指标忽略视觉-动作因果。综述提出联合指标：Counterfactual Consistency（动作如何适应想象未来的扰动）、Foresight-Conditioned Success（执行轨迹是否严格遵循视觉计划）。这是评测范式需要补的核心空白。

第六，长程层次与时间上下文。当前 WAM 多是 System 1，需要多分辨率预测——粗粒度状态转移动作战略规划，细粒度物理细节作反应控制。还要扩展时间记忆，不能只靠标准 attention 的二次开销。

第七，安全验证。WAM 的预测能力让失败模式更危险——它能想象未来，但想象的未来可能错。综述提出 prediction-integrated safety，把不确定性估计作为 safety monitor 的一类输入，在执行前验证动作。

这七个挑战是 WAM 走向成熟的路标。如果你要进入这个领域，挑一个挑战深入，比跟着架构时尚发论文更有价值。

## 一句话收束

这份综述给 World Action Model 这个快速膨胀的领域画了一张导航地图：用 Cascaded vs Joint 二分法把所有方法挂进统一框架，沿生成模态、条件机制、动作解码三维细分，梳理四类数据生态和三维评测协议，指出七大开放挑战。它最有价值的贡献是把"WAM 是什么、不是什么、怎么分类、边界在哪"讲清楚了，而不是提出某个新模型——这给进入这个领域的人一个坐标系，也给领域自身一个反思：架构耦合缺对照实验，我们还在被"架构时尚"驱动。
