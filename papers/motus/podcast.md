# Motus: A Unified Latent Action World Model

2025-12-25，Tsinghua University (THBI/BNRist, Tsinghua-Bosch JMLC) 的 Hongzhe Bi 等人发布的论文，标题是 Motus: A Unified Latent Action World Model。

## 开场：这篇论文真正要解决什么

这篇论文要解决一个非常结构性的问题：**现在的具身智能被人为切成了五块独立建模，每块各训各的，既不能共享先验，又吃不了异构数据。**

哪五块？VLA，从观测和语言直接出动作；世界模型 WM，给定动作预测未来视频；逆向动力学 IDM，给定视频反推动作；视频生成模型 VGM，给定观测和语言想象未来；以及 video-action 联合预测。这五块在数学上是五个不同的条件分布，但它们背后用的是同一套感知和物理知识。你把它们拆成五个模型各自从头训，就等于把一个完整的能力切成五份，每份都缺其它四份的先验。

更麻烦的是第二件事：**action 专家一直没法预训练**。互联网视频、人类第一视角视频，这些数据规模远大于机器人示教，但它们没有动作标签。VLA 的 action 部分只能从有标签机器人数据学，等于把最大的数据源拒之门外。

Motus 的命题是：能不能用一个模型，把这五种建模统一进来，同时让 action 专家也能像 VLM、VGM 那样在海量无标签视频上预训练？它走通了这条路。

## 它的输入和输出到底是什么

你先在脑子里建一个三路输入的图。

输入有：**当前观测 o_t**，一张条件帧，经 Wan2.2-VAE 编码成 video latent；**语言指令**，经 Qwen3-VL-2B 这个理解专家处理；**动作 chunk**，这里有个关键——预训练阶段这个位置是 14 维的 latent action，微调阶段才换成真实机器人动作。本体感受在 joint position 控制下和动作同空间，论文没把它单列。

输出有两路，是同时出来的：**未来视频的 latent**，每 chunk 8 帧；以及 **action chunk**，真机上是 48 步、30 赫兹控制频率，一个 chunk 跨度 1.6 秒。RoboTwin 仿真上动作 chunk 是 16。

这两路输出走的是同一个主干，但机制和 DreamZero 不一样——后面讲。

### 输入到底怎么拼成一条序列

光说三路输入还不够具体，我把它拆成一条显式的拼接序列。

序列大致长这样：开头 `<bos_cond>` 包条件帧 o_t 的 VAE latent，`<eos_cond>`。然后 `<bos_lang>` 包语言指令，但注意——语言不是像 LLM 那样跟视觉 token 拼在一起做自回归，它是走 VLM 理解专家，然后通过 **Tri-model Joint Attention**，也就是共享多头自注意力层，注入到每个专家里。它是 cross-expert attention，不是序列拼接。接着 `<bos_video>` 包未来视频帧的 noisy latent，8 帧 5 赫兹；`<bos_action>` 包动作 chunk 的 noisy latent，48 步 30 赫兹。

这里有三个坑，听别的统一模型论文你也会反复遇到，必须分清。

**第一，语言怎么注入。** Motus 的答案不是单 DiT 加 cross-attention（那是 DreamZero 的做法），而是 Mixture-of-Transformer——三个专家各自保留独立的 Transformer 模块，独立的 QKV、FFN、AdaLN，只在多头自注意力层做共享拼接。语言经 VLM 专家出来后，是通过这个共享自注意力被另外两个专家读到的。所以从序列层面看，语言不在自回归序列里，它在 attention 层里。

**第二，video 和 action 在同一条去噪序列里，但被分配了不同的 timestep。** 这是最反直觉的一点。标准做法（DreamZero 那样）是 video 和 action 共享一个 timestep，一起插值一起去噪。Motus 不这样——它给 video 分配 τo，给 action 分配 τa，两个都独立从均匀分布采样。这就是 UniDiffuser 式调度的核心：**五种模式靠 (τo, τa) 起止值的不同组合切换，没有五个 head。** 想做 VGM，就把 video 从 Tτ 降到 0、action 保持 Tτ 噪声不动；想做 WM，就把 video 降、action 保持 0 干净作条件；想做 IDM，反过来；想做 VLA，video 保持噪声、action 降；想做 Joint，两个都从 Tτ 降到 0。一个权重，靠两个 timestep 的起止值切换五种推理行为。

**第三，训练和推理时这条序列结构是一样的，但哪些 token 真正被去噪、哪些只做条件，模式不同就不同。** 训练时五种模式的数据是混合统一训的；推理时你按目标固定 (τo, τa) 的起止值就行。多视角这篇没显式处理，真机是单视角双臂。

记住这条序列和这三个坑，后面听别的统一模型你会反复用到。

## 架构：三个专家，各自保留先验，只共享注意力

主干不是单一 DiT，是三个专家拼起来的 Mixture-of-Transformers。

**Video Generation Expert**：Wan2.2-TI2V-5B，一个 5B 的视频生成 diffusion 模型，自带 Wan2.2-VAE。**Understanding Expert**：Qwen3-VL-2B 的末层 token 出来，过一个 30 层、hidden 512、24 头的 Transformer。**Action Expert**：30 层、hidden 1024、24 头的 Transformer，和 Wan 同深度，每块含 AdaLN 注入 timestep、FFN、和那个 Tri-model Joint Attention。

三个专家各自有独立参数，但在多头自注意力层共享。这是 MoT 的精髓——**不拼 token，而是拼 attention**。这样做的目的是保住各自预训练先验不被互相稀释：你拿 VLM 去同时训 action，VLM 的理解能力会被带偏；你拿 VGM 同时训理解任务，VGM 的视频生成也会被打散。MoT 让每个专家的"功能身份"不被打散，又能通过共享 attention 互相读到对方隐状态做条件。

为什么不用 UWM 那种"把 observation token 和 action token 串进单一 transformer"的做法？因为 UWM 要么从头训丢先验，要么用小底座，缺 VLM 的视觉语言理解先验或 VGM 的物理交互先验。Motus 要的是把互联网级别的通用先验和机器人专用先验都吃进来。

## 光流 latent action：让无标签视频也能训 action 专家

这是 Motus 相对 VLA 路线最大的增量，单独讲。

问题是：互联网视频、人类第一视角视频没有动作标签，action 专家没法直接训。之前的 latent action 做法——LAPA 用 RGB 重建，会把任务无关的外观编进 latent；AdaWorld 用 β-VAE 解耦，但还是没直接对齐真实控制。

Motus 的解法是用**光流**。光流是相邻帧之间的像素级位移，天然剥掉外观只留运动。不管你是人、是 Franka、还是双臂机器人，只要动了就产生光流。所以光流是一种跨本体通用的运动语言。

具体管道：DPFlow 算光流→转成 RGB 图→DC-AE 这个深度压缩自编码器把它编码成 4×512 的 token→一个轻量 encoder 再压到 14 维。这 14 维是刻意对齐典型机器人动作空间的尺度，让 latent action 能直接映射到可执行控制。训练时 90% 是无标签数据做光流重建自监督，10% 是有标签数据（task-agnostic 的 Curobo 随机采样数据 + 真机示教）做弱监督对齐。loss 是重建 + 动作对齐 + KL 正则。

这个设计让 action 专家第一次能像 VLM、VGM 那样在海量无标签视频上预训练。这是 Motus 命题里"统一"二字的真正含义——既统一五种建模，又统一有标签和无标签数据。

## 一个反直觉的工程：把视频率压到动作率的六分之一

讲一个很小但很重要的设计，叫 Action-Dense Video-Sparse。

action chunking 下，默认你会让视频和动作的采样率一致。但视频每帧 latent 是上万维，action 每步才 14 维，token 维度悬殊。结果就是 Tri-model Joint Attention 里 video token 把 action token 淹没了，模型过拟合到 video 预测，action 预测能力反而弱。

Motus 的解法很简单粗暴：把视频采样率压到动作率的六分之一。视频 5 赫兹，动作 30 赫兹，每 chunk 8 帧视频对 48 步动作。看起来视频帧变少了，但这恰恰让两类 token 在 attention 里数量平衡，action 预测能力恢复。同时视频帧冗余减少，训练推理也更快。这是论文里一个很工程但很关键的设计。

## 给你一个数值 sense：这套模型到底多大

光说 8B 总参数没感觉，我把维度念细一点。

三个专家各自的规格：Action Expert 是 hidden 1024、30 层、24 头、GELU，641.5M 参数；Understanding projection 是 hidden 512、30 层、24 头，253.5M；VGM 是 Wan2.2-5B，5B 参数的 dense DiT；VLM 是 Qwen3-VL-2B，2.13B。加起来大概 8B。

VAE 是 Wan2.2-VAE，空间 16 倍下采样，时间 4 倍，latent 通道 16，总压缩比 T×H×W 是 4×16×16。注意这和 DreamZero 用的 Wan2.1-VAE 不一样——Wan2.1 是空间 8 倍，Wan2.2 是空间 16 倍，压缩更狠。所以 Wan2.2 单帧 latent 维更小，720P 输入下大概每帧 1.4 万维，远小于 Wan2.1 14B 的 10 万维。Motus 选 Wan2.2-5B 而不是 Wan2.1-14B，论文原话是"accessibility and ease of use"，用更小底座换易部署。

Chunk 设置：视频 8 帧 5 赫兹，动作 48 步 30 赫兹，两边都精确锁在 1.6 秒。上下文只条件当前帧 o_t，不像 DreamZero 有 6.6 秒历史记忆——这是 Motus 的一个明显短板，长程信息靠 VLM 隐式承担。

训练规模：三阶段，Stage1 视频 pretrain 大约 8000 GPU 小时，Stage2 统一训练带 latent action 大约 10000 GPU 小时，Stage3 真机 SFT 大约 400 GPU 小时，总共约 18400 GPU 小时。三阶段都是 batch 256，AdamW，学习率从 8e-5 降到 1 到 5e-5。推理 10 步 flow matching，Logit Normal 采样。

记住这套数字：8B 总参、Wan2.2-5B 主干、16 通道 latent、视频 5 赫兹对动作 30 赫兹、1.6 秒 chunk、18400 GPU 小时三阶段。这是个很大的训练量，但比 DreamZero 的 14B 单 DiT 要轻一些。

## 三阶段渐进训练：从视频到 latent action 到真动作

Motus 的训练不是一锅炖，是分三阶段层层叠加。

**Stage1**，约 8000 GPU 小时，只动 VGM。用多机器人轨迹和人类视频把视频生成模型在机器人视觉动态上适应一下。这阶段 action 专家不参与。

**Stage2**，约 10000 GPU 小时，是关键阶段。三专家全开，VLM 冻结，用 latent action 在 Level2 到 Level5 的大规模数据上做统一训练。这阶段 action 专家第一次被预训练——它从光流编码的 latent action 里学到了运动和交互知识。这一步是 Motus 增量最大的来源，消融实验里 Stage2 把 RoboTwin Randomized 从 Stage1 的 81.86% 拉到 87.02%。

**Stage3**，约 400 GPU 小时，在 Level6 的真机小数据上 SFT。这阶段 latent action 换成真实机器人动作，完成本体适配。每任务 100 轨迹，总共 2000 段。

六层数据金字塔从底到顶：Web Data、Egodex 人类视频、RoboTwin 合成、AnyPos task-agnostic、多机器人轨迹、目标机器人真机。底层量大质低无标签，靠 latent action 学；顶层量小质高有标签，直接对齐本体。latent action 是连接这两端的桥梁。

## 关键结果：统一模型不输专门模型

我挑几个有对照的数字。

**RoboTwin 2.0 仿真，50 任务多任务联合训**：Randomized 场景 Motus 87.02%，X-VLA 72.84%，π0.5 只有 43.84%。Clean 场景 Motus 88.66%。这就是 abstract 里说的"对 X-VLA +15%、对 π0.5 +45%"。注意 from-scratch 的 Motus 也有 77%，说明架构本身就有增益，预训练再加 10 个点。

**LIBERO-Long 长程任务**：Motus 97.6，追平 X-VLA 的 97.6，OpenVLA-OFT 是 94.5，π0 是 85.2。

**真机，AC-One 双臂 9 个长程任务**：Motus 平均 partial success 63.22%，π0.5 只有 14.79%，w/o-pretrain 25.86%。Agilex-Aloha-2 上 5 个任务 Motus 59.30%，π0.5 48.60%。注意真机只比了 π0.5，没和 DreamZero、GR00T N1.6 直接对照。

**一个最值得记住的对照**：Motus 在 IDM 模式下，action MSE 是 0.014，专门训的 IDM baseline——ResNet18+MLP 是 0.044，DINOv2+MLP 是 0.122。统一模型在"只做逆向动力学"这个单一任务上，打败了专门为这个任务训的模型。这说明共享权重带来的先验 > 专门优化的独立模型。

**另一个对照**：同一权重切模式，Joint 模式（同时生成 video 和 action）87.02%，VLA 模式（只生成 action）83.90%。把 video 也一起生成，反而让 action 更准。这给 DreamZero 式 WAM 的"联合生成优于纯 policy"论断添了正面证据。

## 局限：别替它过度外推

论文自己承认的：它还是 System 1，长程推理受限于上下文；未来要从互联网规模通用视频学 latent action——这句话的意思是，现在的 latent action 预训练主要还在 ego-centric human 和多机器人数据上，真正互联网规模的通用视频还没用上。

我们读出的几个问题。第一，上下文建模很弱，只条件当前帧 o_t，不像 DreamZero 有 6.6 秒历史记忆。长程任务的"记忆"实际靠 VLM 理解专家隐式承担，但论文没量化这部分贡献到底多大。

第二，真机 baseline 只比 π0.5，没和 DreamZero、GR00T N1.6、X-VLA 真机直接对照。而且 π0.5 在某些任务上本身不弱，比如 Touch Keyboard 在 Agilex 上 π0.5 是 72.5%，Motus 是 80%；但 Put Bread into Oven 在 Agilex 上 Motus 34%，π0.5 反而 36%。泛化收益不均匀。

第三，latent action 的 14 维是经验设定，论文原话"roughly matching typical robot action spaces"。不同本体动作维度差异大，Franka 7 自由度加夹爪、双臂二十几自由度，14 维是否够表达所有运动，论文没充分讨论。

第四，Stage2 用 latent action，Stage3 才换真实动作，这中间有个迁移跳跃，论文没给这个跳跃单独的消融。Stage3 真机微调对最终增益的贡献没被独立量化。

第五，五种模式的实验密度不均。Joint、VLA、IDM、WM 都有量化结果，但 VGM 只有可视化没有定量。"统一模型每种模式都强"这个卖点的证据强度不齐。

所以 Motus 是统一模型路线的一个强证据，但不是终点。它证明了"五模式统一 + action 专家可预训练"在跑分上能赢，但实时性、长程记忆、模式间均衡性都还有空间。

## 一句话收束

Motus 用一个 Mixture-of-Transformers 把 VLM、Wan2.2、Action 三个专家并到一个 8B 模型里，靠 UniDiffuser 式调度在 VLA/WM/IDM/VGM/Joint 五种模式间切换，并用光流编码的 latent action 让无动作标签的海量视频也能预训练 action 专家。它最有力的论证是"统一模型在 IDM 单任务上打败专门训的 IDM"——这把"统一"从口号变成了可量化的优势。
