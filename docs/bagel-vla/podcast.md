# BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation

2026-02-12，Tsinghua University 的 Yucheng Hu 等人发布的论文，标题是 BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation。

## 开场：这篇论文真正要解决什么

这篇论文要解决一个非常具体的问题：**长程操作任务里，全局指令隐含一串子阶段，现有 VLA 把它当黑盒映射，做不好。**

举个例子。你给机器人一个指令："把积木按红、黄、蓝、绿的顺序叠起来"。这个指令背后其实是一串阶段：先抓红的放到指定位置，再抓黄的叠上去，再蓝的，再绿的。每个阶段都是一次独立的抓放。但纯 VLA 看到这个全局指令，直接从观测出动作，中间没有显式分解。结果就是它在第一层可能做对，做到第二层第三层就乱了，因为它没有一个"我现在该抓哪个颜色"的显式推理。

现有方法要么只做语言规划（高层分解）但缺视觉前瞻，要么只做视觉预测（想象未来）但缺逻辑推理。两边割裂。

BagelVLA 的命题是：能不能让一个模型，**显式地交错做三件事**——先用文字想"下一步该干啥"，再想象"干完之后世界长什么样"，最后出动作。而且这三步要快，不能因为加了视觉想象就慢到没法实时。

## 它的输入和输出到底是什么

你先在脑子里建一个三专家的图。

主干是 Bagel，一个基于 Qwen2.5-LLM-7B 的统一理解+生成模型，本身已经是 MoT 架构，有两个专家：理解专家和生成专家。BagelVLA 在这基础上加了第三个——action expert，2B 参数。

输入有三路：**多视角观测**，Calvin 仿真用 2 个视角预测第 3 个，RoboTwin 和真机用 3 个视角（主视角加左右腕）。每个视角同时过两个 encoder——SigLIP2 出理解特征给 LLM 专家，FLUX VAE 出生成特征给生成和 action 专家。**全局指令**，一段文字。**本体感受**，Calvin 不用，RoboTwin 和真机用。

输出有三路，是按顺序交错出来的：先是**子任务文字**，比如"下一个抓红色积木"；然后是**关键帧图像**，子任务完成时世界应该长什么样；最后是**动作 chunk**，Calvin 是 10 步，RoboTwin 是 16 步（实际 horizon 48），真机是 24 步，双臂 14 自由度。

这三路输出走的是同一个 MoT 主干，但机制和 DreamZero、Motus 都不一样——后面讲。

### 输入到底怎么拼成一条序列

光说三路输入还不够具体，我把它拆成一条显式的拼接序列。

序列大致长这样：开头 `<bos_obs>` 包多视角观测，每个视角同时有 SigLIP2 特征和 FLUX VAE 特征，`<eos_obs>`。然后 `<bos_lang>` 包全局指令 L，`<eos_lang>`。接着 `<bos_plan>` 包子任务文字 l_t，这个是模型自己自回归生成的，`<eos_plan>`。再 `<bos_keyframe>` 包关键帧的 noisy latent，经 FLUX VAE，`<eos_keyframe>`。最后 `<bos_action>` 包动作 chunk 的 noisy latent，`<eos_action>`。

这里有三个坑，听别的统一模型论文你也会反复遇到，必须分清。

**第一，语言和子任务规划都是序列拼接进自回归。** 这点很关键，和 DreamZero、Motus 都不一样。DreamZero 的语言是 cross-attention 注入，不在自回归序列里；Motus 的语言走 VLM 专家再经共享 attention 注入，也不在序列里。但 BagelVLA 把文本规划当成可生成的 token，**显式写在序列里**，模型真的会一段一段吐出"下一个抓红色积木"这样的文字。视觉条件则通过 MoT 共享自注意力注入，不是 token 拼接。

**第二，多视角在序列层拼接，不是通道拼接。** 每个视角都有一套 SigLIP2 加 VAE 的 token，在序列里排开。Calvin 预测第 3 视角，RoboTwin 和真机预测主视角。这点和 DreamZero 把多视角在通道维拼成一帧不同——BagelVLA 的多视角是 token 级拼接。

**第三，训练和推理时这条序列结构是一样的，但 image 和 action 这两个 flow matching 怎么交互，有三种方案，默认用 Single-step RFG。** 这是 BagelVLA 最核心的设计，单独讲。

记住这条序列和这三个坑，后面听别的 cascaded WAM 你会反复用到。

## 架构：三个专家，Bagel 底座，加一个 2B action expert

主干是 Bagel，并非从零训的。Bagel 本身是统一理解+生成的 MoT 模型，基于 Qwen2.5-LLM-7B，7B 激活、14B 总参。它已经有两个专家：理解专家 7B，负责语言推理和子任务生成；生成专家 7B，负责图像生成。这两个专家共享多头自注意力层，各自保留独立 Transformer 模块。

BagelVLA 在这基础上加了第三个专家：action expert，2B 参数。它和 Qwen2.5 同架构，但 MLP 的 intermediate size 从 18944 缩到 3584，也就是原来的五分之一。为什么缩这么小？因为它要高频跑。真机上 action 频率要 40 赫兹，还得支持 KV-cache 异步执行，2B 是个权衡——够用又能快。

三个专家各自独立 Transformer 模块，在多头自注意力层共享。这和 Motus 的 Tri-model Joint Attention 是同一套思想：保住各自预训练先验不被互相稀释，又能跨模态融合。

为什么用 Bagel 当底座？因为 Bagel 从互联网规模数据里学到了强语言推理和图像生成先验。这是 Cosmos Policy 那种纯视频模型 policy 给不了的——Cosmos 直接 fine-tune 大视频模型当 policy，缺专门 VLM backbone，指令跟随差，更别说做算术式的 CoT 推理了。BagelVLA 能做"算术式摆积木"这种任务，靠的就是 Bagel 底座的语言推理。

## 交错规划：文字→关键帧→动作的显式因果链

这是 BagelVLA 区别于其它 WAM 最根本的一点。

它把联合分布 p(a_t, v_{t+k}, l_t | v_t, L) 因式分解成三步：第一步，p(l_t | v_t, L)，理解专家自回归出子任务文字，比如"下一个抓红色积木"；第二步，p(v_{t+k} | v_t, L, l_t)，生成专家用 flow matching 去噪出关键帧图像，就是子任务完成时世界应该长什么样；第三步，p(a_t | v_t, L, l_t, v_{t+k})，action 专家用 flow matching 去噪出动作 chunk。

三步是因果链：后一步能 attend 前一步的隐状态做条件。每一步都有显式监督——文字用 Cross-Entropy，图像和动作都用 flow matching 的 MSE。

为什么要这么显式分解？因为长程任务的全局指令隐含多阶段，纯 VLA 黑盒映射做不好。RoboTwin 上加文字规划，成功率从 54% 涨到 75%，整整 21 个点。这是显式交错规划最强的证据。

## 给你一个数值 sense：这套模型到底多大

光说 16B 总参数没感觉，我把维度念细一点。

三个专家都是 hidden 3584、28 层，和 Qwen2.5-LLM-7B 同架构。理解专家和生成专家的 MLP intermediate 是 18944，就是 Qwen2.5 原配；action 专家缩到 3584，五分之一。所以理解 7B、生成 7B、action 2B，加起来大概 16B。

图像分辨率是 256×256。多视角：Calvin 用 2 个视角预测第 3 个，RoboTwin 和真机用 3 个视角（主视角加左右腕）预测主视角。FLUX VAE 做空间 8 倍下采样，256×256 变成 32×32，latent 通道 16，每帧大概 1.6 万维。SigLIP2 那边是 patch token 序列，做理解用。

Chunk 设置因场景而异：Calvin 是 10 步，RoboTwin 是 16 步（每 3 步采一次，实际 horizon 48），真机是 24 步。真机上 action 频率 40 赫兹，chunk 48 步，一个 chunk 跨 1.2 秒；异步执行能提到 72 赫兹。

训练分两阶段。Stage1 预训练，64 张 A800，batch 大概 1600，2 万步，学习率 1e-5，用 FSDP。这阶段只训理解和生成专家，目标是注入语言规划和视觉动态能力，同时混入通用 VQA 数据保住底座的通用语言能力。Stage2 finetune，Calvin 用 8 张 A800 batch 192 训 3 万步，RoboTwin 8 张 A800 batch 128 训 6 万步，真机 32 张 A800 batch 512 训 5 万步。这阶段三专家一起训，把语言、视觉、动作对齐。

推理时图像 flow matching 默认 50 步去噪，动作 10 步。但 Single-step 模式下图像只跑 1 步。timestep 采样：图像用 LogitNormal(0,1)，动作用 Beta(1.5,1)——动作偏 向低噪声，收敛快。

记住这套数字：16B 总参、Bagel 7B+7B 底座、2B action expert、256×256 输入、chunk 24、40 赫兹同步 72 赫兹异步、两阶段训练。这是个非常大的训练量，但比 DreamZero 的 14B 单 DiT 多了三专家的灵活性。

## 最关键的设计：Single-step Denoise + RFG

这是 BagelVLA 最聪明的地方，也是它能把视觉前瞻塞进 action 循环还能实时的根本。单独讲。

问题是这样的。你要在 action 循环里先生成关键帧图像再出动作，naive 做法是先跑完图像的 N1 步去噪，再跑动作的 N2 步去噪，总共 6.04 秒一个 chunk——完全没法实时。

BagelVLA 试了三种方案。第一种 Complete，先图像 50 步再动作 10 步，6.04 秒。第二种 Joint，图像和动作同步去噪 N 步，2.90 秒。第三种 Single-step，动作只看图像第 1 步去噪的 KV-cache，1.23 秒。

这里有个反直觉的发现：**Single-step 不仅最快，还最准。** Calvin ABC-D 上，Complete 是 2.480，Joint 是 2.038，Single-step 是 3.345，RFG 变体是 3.600。最快的反而最好。

为什么？因为 Complete 和 Joint 在 OOD 测试时——比如颜色变化——图像 flow matching 的中间去噪态会进入 OOD 分布，污染 action。而 Single-step 只取图像第 1 步的 KV，中间态不暴露给 action，反而更鲁棒。这是个很重要的工程 insight：少去噪比多去噪好，前提是你只取第 1 步。

但 Single-step 有个问题。它用纯高斯噪声 N(0,I) 作图像初始，action 从纯噪声读条件，信息弱，而且关键帧生成需要很多步才能出像样的图。

RFG 就是解决这个的。它把初始噪声从 N(0,I) 换成 N(v_t, I)——以当前观测为均值的高斯。等于让 flow matching 学"相对当前帧的残差变化"，而不是"从零重建整帧"。

效果立竿见影。Figure 5 可视化显示，RFG 在 10 步就能生成高质量未来帧，Naive 几乎还是噪声。原因是当前帧注入让模型聚焦动态区域——机器人手臂的运动——而不是重建静态背景。背景已经给定在初始噪声里了，模型只需要学怎么变。

action 学习也收敛更快，延迟保持 1.23 秒不变。Calvin ABC-D 从 Naive 的 3.345 提升到 RFG 的 3.600。

所以 RFG 的本质是：**把"生成未来帧"变成"生成相对当前帧的残差"**。这是 BagelVLA 整篇论文里杠杆比最高的一个设计。

## 异步执行：40 赫兹到 72 赫兹

还有一个提频的 trick，叫异步执行。

问题是这样：每个 chunk 都要重算理解和生成专家的 KV，包括图像去噪，40 赫兹还是不够快。BagelVLA 的观察是，长程任务里子任务和关键帧不会每个 chunk 都变，只在阶段切换时才需要重新规划。

训练时它随机用前一帧替换当前帧，让模型学会"在观测略滞后时也能出动作"。推理时理解和生成专家的 KV 不每 chunk 更新，只更新本体感受输入给 action expert，让 action expert 出新 chunk。

这样频率从 40 赫兹提到 72 赫兹。代价是牺牲少量视觉新鲜度，但对长程任务的结构来说正合适。当然这个假设对高动态任务可能不成立——这是论文没量化的一个局限。

## 关键结果：长程任务上的优势

我挑几个有对照的数字。

**Calvin ABC-D**：BagelVLA 平均完成长度 4.405，VPP 4.329，UP-VLA 4.078，π0 是 3.648。这是 1000 个长度 5 的任务，平均能完成多长。

**RoboTwin 2.0**：Clean 场景 BagelVLA 75.26%，w/o-keyframe 56.72%，UP-VLA 52.92%，π0 46.42%。Randomized 场景 BagelVLA 20.87%，虽然绝对分低，但仍是最好的。注意 w/o-textual 是 54%，加上文字规划到 75%，**整整 21 个点的提升**，这是显式交错规划最强的证据。

**真机 Basic Task 9 类平均**：BagelVLA 75.5%，π0 65.0%，VPP 59.5%。

**真机长程任务**，这是 BagelVLA 真正发力的地方。Stack Cubes 按指定顺序叠积木，BagelVLA 平均 73.3%，w/o-keyframe 53.3%，π0 40.0%，VPP 25.0%。Calculate and Place Symbols 算术式摆积木，BagelVLA 63.3%，w/o-keyframe 50.0%，π0 31.7%，VPP 23.3%。这两个任务都需要多阶段规划，BagelVLA 的优势非常明显。

**推理延迟**：Single-step RFG 1.23 秒一个 chunk，Complete 要 6.04 秒，Joint 2.90 秒。单 A800 上测的。

## 一个最值得记住的 insight

论文里有个观察，我建议你重点记：**长程任务上，规划准确率接近 90%，但任务成功率只有 63 到 73%。**

翻译过来：模型"想对了"——它能正确识别该抓哪个积木、该放哪里——但"手没跟上"——动作执行精度不够，把正确的规划搞砸了。

这个 gap 说明什么？说明 BagelVLA 的交错规划是对的，瓶颈不在规划层，而在 action mapping 的精度。这是个很重要的诊断：改进方向应该是更精细的动作控制或更高质量的动作数据，而不是更复杂的规划。这个 insight 对整个 WAM 路线都有参考价值。

## 局限：别替它过度外推

论文自己承认的：长程任务上规划准确率和任务成功率有 gap，action mapping 精度是瓶颈，受限于模型和数据集的精细控制能力。RoboTwin Randomized 场景绝对分仍低（20.87%），强 domain randomization 下所有方法都 struggle。

我们读出的几个问题。第一，RFG 的 N(v_t, I) 假设"未来帧和当前帧结构相似"，对快速运动或视角剧烈变化的任务可能不成立。比如机器人快速移动导致视角大变，当前帧就不再是好的残差起点。论文没给 RFG 失效场景分析。

第二，异步执行 72 赫兹隐含假设"子任务和关键帧不每 chunk 变"。对需要每 chunk 重新规划的高动态任务，异步会掉点，论文没量化这个精度损失。

第三，Single-step 只取图像第 1 步 KV，等于放弃了完整关键帧的信息。如果任务强依赖精确未来视觉目标，比如精细对位，Single-step 可能不够。论文只在 Calvin 这种简单任务上对比了 Single-step 和 Complete，没在精细任务上对比。

第四，真机只比了 π0 和 VPP，没和 DreamZero、Motus、X-VLA 真机直接对照。长程任务的优势部分来自 baseline 本身没显式规划，比较不够公平。

第五，三专家总 16B，单张 RTX 5090 跑 1.23 秒一个 chunk 依赖异步和 KV-cache，部署门槛仍高于纯 VLA。Action expert 2B 的容量是否够复杂任务，论文没讨论。

所以 BagelVLA 是 cascaded WAM 路线的一个强证据，但不是终点。它证明了"显式交错规划 + RFG 单步残差"能在长程任务上赢，但实时性、精细控制、模式间均衡性都还有空间。它和 DreamZero 形成一个有趣的对照：DreamZero 重端到端实时，BagelVLA 重显式规划长程，两条路都还没走到终点。

## 一句话收束

BagelVLA 在 Bagel 统一理解+生成底座上接一个 2B action expert，把"文字规划→关键帧预测→动作生成"三步显式交错进一条序列，并用 Residual Flow Guidance 把视觉前瞻的延迟从 6 秒压到 1.23 秒。它最有力的论证是"规划准确率 90% 但成功率 73%"这个 gap——这把长程任务的瓶颈从规划层精确地定位到了动作执行层。
