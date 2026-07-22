# HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Models

2026-05-11，Peking University 的 Qiuxuan Feng 等人发布的论文，标题是 HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Models。

## 开场:这篇论文真正要解决什么

这篇论文要解决一个很具体的结构性矛盾:**现有的世界动作模型 WAM 分两条路,一条泛化好但不准,一条准但泛化崩,你只能二选一。**

这两条路长什么样。第一条叫 Imagine-then-Execute,先让世界模型生成未来视频,再用 inverse dynamics 模型从视频反推动作。它的好处是 transit——也就是物体之间的移动、接近——泛化特别强,即使换背景、换位置,它都能生成对的未来,然后跟着走。但一到 interaction——也就是真正的接触、抓取、插入、堆叠——它就崩,因为 inverse dynamics 只能从像素反推,缺接触级的精度。

第二条叫 Joint Modeling,直接把 video 和 action 一起联合建模。它的好处是 interaction 精度高,能做精细操作。但它的 transit 在 OOD 下崩——因为 action 被 SFT 数据分布锁死了,机器人根本到不了没见过的目标位置。

HarmoWAM 的核心命题是:**如果你承认这两条路线各有最优解,且各自的最优解都在对方最弱处,那正确的做法是分工而非折中**——让一个共享的世界模型同时提供两路条件,一路管泛化、一路管精度,再用一个门控按任务阶段硬路由。

它走通了,顺带把 OOD 平均成功率做到 82%,比最强 VLA 高 33 个点、比最强 WAM 高 29 个点。

## 它的输入和输出到底是什么

你需要在脑子里先建一个模块图。

输入有三路:**视觉观测**,3 个视角的 RGB(单臂是 1 个第三人称加 2 个腕部,双臂是 1 个全局加 2 个腕部),RealSense 640×480;**语言指令**,一个冻结的 T5 编码;**本体感受**,单臂 7 自由度、双臂 14 自由度。

输出有两路,但注意——**这两路不是同时出的**。一路是世界模型生成的 13 帧未来视频,256×320 分辨率,5 步去噪;另一路是 action chunk,12 步连续动作,生成频率 48 赫兹。这两路输出背后其实是两个不同的 action expert,由门控二选一,不是并行。

### 输入到底怎么拼成一条序列

光说三路输入还不够具体,我把它拆成一条显式的拼接序列,你在脑子里能建出来。

序列大致长这样:开头是 `<bos_ctx>`,里面是当前观测的 3 路 RGB;然后 `<bos_lang>` 包语言指令经 T5,`<eos_lang>`;接着 `<bos_wm>` 是世界模型要处理的 noisy latent,13 帧未来视频的噪声 latent,`<eos_wm>`;再 `<bos_cond>` 是世界模型当前步的隐含 latent,形状是 B×80×3072,`<eos_cond>`;然后 `<bos_gate>` 是门控用的 SigLIP patch 特征,经 MLP 出一个 s_t 在 0 到 1 之间,`<eos_gate>`;最后 `<bos_chunk>` 是当前要生成的动作——**要么**走 predictive expert(diffusion 去噪出 12 步动作),**要么**走 reactive expert(DINOv2 提 patch 特征 concat 上 latent 经卷积 decoder 出动作),二选一,`<eos_chunk>`。

这里有几个坑要特别说清,不然容易误解。

第一,语言指令**不是**像 LLM 那样跟视觉 token 拼在一起做自回归——它是用 cross-attention 注入到世界模型的 DiT block 和 predictive expert的 DiT block 里,是个条件,不是序列的一部分。

第二,多视角也不进入序列拼接。它是在模块层分工处理掉的——世界模型只用其中一路视角生成未来视频(单臂用第三人称、双臂用全局),另一路视角喂给 predictive expert 的 SigLIP 提供实时闭环视觉。所以从序列层面看,多视角是按模块分工,不是序列级融合。

第三,训练和推理时这条序列长得基本一样,但未来视频的来源不同。训练时,世界模型用 ground-truth 视频做 teacher forcing 学;推理时,世界模型自己生成未来视频喂给下游。门控训练时,监督标签 y 是用 keyframe pipeline 自动标的——夹爪状态变化或末端高度阈值作为 key event,前后各 20 帧窗口标成 interaction,其余标成 transit。

记住这条序列,后面听别的 WAM 论文你会反复遇到类似的拼接。HarmoWAM 的答案是:context 是当前观测+世界模型预测的未来,language 是 cross-attention,门控是阶段感知的硬路由,两个 expert 由门控二选一。

## 架构:一个共享世界模型喂两个专门化 expert

主干是 Wan2.2-TI2V-5B,一个 5B 的 text-and-image-to-video diffusion 模型,带 MoE 架构。HarmoWAM 的策略是先在 190 万条机器人轨迹上预训练它注入物理先验,再在自采数据上全参 finetune 任务化,然后冻结。

冻结之后,世界模型干两件事:生成 13 帧未来视频(显式),和产出隐含 latent(隐式)。这两路条件分别喂给两个 expert。

**Predictive expert** 是一个 1B 参数的 DiT,28 个 transformer block。它吃世界模型当前步的隐含 latent F_V_t,经 cross-attention 注入,加上 SigLIP 的图像特征和 T5 的文本特征,用 diffusion 去噪生成 12 步动作。它用的是标准 diffusion denoising loss。它管 interaction 精准。

**Reactive expert** 是一个 DINOv2-base 加一个 Orientation Decoder。它对世界模型生成的每帧未来视频用 DINOv2 提 patch 级几何特征,再把世界模型的 latent 池化到 768 维,在 token 维 concat,经多尺度卷积 decoder 出动作。它用的是 Smooth L1 loss。它管 transit 泛化。

注意两个 expert 的设计哲学完全不同。Predictive 吃的是隐含 latent——时序物理先验,适合精细操作因为它知道"这一步动作会让世界怎么连续地变"。Reactive 吃的是显式未来视频——泛化视觉先验,适合空间导航因为它能从世界模型预测的未来里反推 IDM,跳出 SFT 分布。但两者共享同一个世界模型的条件,物理先验不割裂。

这就是 HarmoWAM 名字里 "harmonizing" 的物理含义——不是把两个模型凑一起,是让一个世界模型的不同表征服务于不同目标,再用门控协调。

## 给你一个数值 sense:这套模型到底多大

光说 5B 加 1B 可能没感觉,我把维度念细一点。

世界模型是 Wan2.2-TI2V-5B,5B 参数,带 MoE,hidden dimension 是 3072。它用的是 Wan2.2-VAE,压缩比是 16×16×4——空间 16 倍下采样、时间 4 倍下采样、latent 通道 16。所以输入是 256×320 分辨率,VAE 压到 16×20 的 latent patch,通道 16,再 patch 化成 80 个 token,每个 token 3072 维。论文里那个 R^{B×80×3072} 就是这么来的——80 token 是 patch 数,3072 是 Wan2.2-5B 的 hidden dim。

Predictive expert 是 1B 参数的 DiT,28 个 transformer block,做 diffusion 去噪。Reactive expert 用 DINOv2-base,embedding 维度 768,patch size 14;对一张未来视频帧它能提 1369 个 patch token(论文显式给 R^{B×1369×768}),每个 768 维。这两路在 token 维 concat 之前,世界模型的 latent 要从 3072 池化到 768 才能对齐。

回到 chunk 设置:世界模型预测 13 帧未来视频,action chunk 是 H=12 步,世界模型去噪 5 步。这里有个很重要的消融——3 步去噪成功率 80%,5 步 85%,10 步还是 85%,50 步 87%。也就是说从 5 步开始边际递减非常明显,更多步只是视觉质量略升但成功率几乎不动,推理频率反而掉。所以 5 步是 quality-speed 的甜点。这条结论对后续 WAM 工程化很关键——世界模型视频"够用"就行,不用追求高保真。

动作那边是连续相对位姿。单臂 7 自由度——3 维位移、3 维 Euler 角、1 维二值夹爪;双臂 14 自由度,每臂 7 维。控制频率 48 赫兹,chunk 12 步。

训练规模:8 张 H20 GPU,两阶段。Stage 1 世界模型全参 flow matching finetune,先在 190 万条轨迹预训练注入物理先验,再在自采 6 任务上任务化。Stage 2 冻结世界模型,只训两个 expert 和门控。loss 是加权 sum——predictive 的 diffusion loss 主导,reactive 的 Smooth L1 权重 0.1,门控的 BCE 权重 0.05。这个权重设计有意思:predictive 管 interaction 最难,所以让它主导;reactive 和门控作辅助。门控离线准确率 96.95%,1637 个测试帧对。

## 真正让 HarmoWAM 成立的:门控硬路由

这是论文最核心的设计杠杆,值得单独讲。

双 expert 怎么协同是个真问题。最 naive 的做法是平均——两个 expert 各出一份动作,数值平均一下。但作者做了消融,结果很打脸:在 position OOD 上,Averaging 掉了 46 个点,Keyframe-Averaging(只在 interaction 阶段平均)掉了 31 个点,而门控硬路由几乎不掉。

为什么会这样。因为两个 expert 在错误阶段发力会互相干扰。transit 阶段如果让 predictive 发力,它会过度精细,机器人反而到不了目标——因为 transit 要的是泛化的大范围移动,不是精细微调。interaction 阶段如果让 reactive 发力,它精度不够,夹爪会偏。

门控的解法是阶段感知的硬二选一。它是一个轻量 MLP,用 SigLIP 的 patch 特征出 s_t 在 0 到 1 之间。推理时 s_t 大于 0.5 走 predictive expert,小于等于 0.5 走 reactive expert。训练监督是用 keyframe pipeline 自动标的——夹爪状态变化(open 到 close 是 grasping、close 到 open 是 releasing)或者任务相关的末端高度阈值(insertion、pouring、placing 这种垂直运动任务)作为 key event,前后各 20 帧窗口标成 interaction,其余标成 transit。双臂任务里任一臂满足即标 interaction。

这里还有一个细节。Figure 3 给了机理证据——两个 expert 最后一层的 attention map。Predictive expert 的注意力集中在被操作物体上,说明它在做接触级精细规划。Reactive expert 的注意力散在夹爪和任务相关的周围环境上,说明它在做空间导航。这说明双 expert 不是凭空分的,是它们天然 attend 到不同区域,门控只是把这个天然分工显式化。

## Motivation 实验:把 trade-off 做成结构化诊断

HarmoWAM 最让我佩服的是它先做了一个非常扎实的 motivation 实验,把"为什么需要新框架"这个问题钉死了——模型设计反倒在其次。

它在两个代表任务上测——Put Flowers in Vase(长程双臂协调)和 Stack Coke Cans(精密堆叠)。把每个任务拆成 transit 和 interaction 两阶段,在 ID 加 3 档 OOD 各做 10 次试验。关键设计是:Joint Modeling 的 interaction 用"初始化到目标附近"来测——这样能解耦 transit 失败和 interaction 精度。

结果很决定性。Imagine-then-Execute 的 transit 在 OOD 几乎 10/10 满分,但 interaction 掉到 55%。Joint Modeling 的 OOD transit 掉到 0 到 5/10,但 interaction(初始化到目标附近)能 10/10。

这意味着什么。意味着 Joint Modeling 的瓶颈根本不是精度,是探索——它到不了目标,但只要到了目标附近它就准。这正好和 Imagine-then-Execute 互补——后者能到目标但 interaction 不准。

这个 motivation 直接驱动了整个 HarmoWAM 设计:既然两条路各自最优解都在对方最弱处,那就让世界模型的隐含 latent 喂 predictive 补 Joint Modeling 的精度,让世界模型的显式未来喂 reactive 补 Imagine-then-Execute 的泛化,再用门控按阶段路由。整个架构不是拍脑袋,是从 motivation 实验长出来的。

## 关键结果:不仅赢,而且赢在 OOD 最难的地方

我挑几个有对照的数字。

**ID 平均成功率**,6 个任务:HarmoWAM 89%,最强 baseline 是 Cosmos-Policy 78%、π0.5 74%、Wan+AnyPos 67%。从零训的 VLA 基本被甩开。

**OOD 平均成功率**,3 档 OOD × 6 任务:HarmoWAM 82%。最强 baseline 是 Wan+AnyPos 53%、π0.5 49%、Cosmos-Policy 44%。论文 abstract 声称的"+33% vs 最强 VLA、+29% vs 最强 WAM"就是这里来的。

**Unseen Position OOD**,这是最难的一档——目标放在训练覆盖区域之外的空间不相交区域。HarmoWAM 80%,π0.5 掉到 32%,Cosmos-Policy 掉到 26%。这档最能说明问题,因为它直接戳中 Joint Modeling 的痛点——SFT 分布锁死探索。HarmoWAM 用 reactive expert 借世界模型泛化跳出这个锁死,效果立竿见影。

**门控准确率**,离线 96.95%,1637 个测试帧对。这说明 keyframe pipeline 自动标注够可靠,门控能稳定区分 transit 和 interaction。

**去噪步数消融**:3 步 80%,5 步 85%,50 步 87%。从 5 步开始边际递减,5 步是甜点。

## 一个最值得记住的 insight

论文里有一个观察,我建议你重点记:**世界模型的隐含 latent 比显式视频更关键。**

Figure 5c 消融:去掉世界模型的 latent 特征,predictive expert 的 ID 成功率从 95% 掉到 62%,掉了 33 个点;reactive expert 掉到 65%。这说明显式视频更多服务于泛化导航,但真正撑起精细操作精度的是时序相干的隐含表征。

这个 insight 分量很重。它意味着对 WAM 来说,世界模型的价值不只在"生成漂亮的未来视频",更在它中间层的 latent 编码了"动作会让世界怎么连续地变"这种时序物理先验。这对后续 WAM 的设计有指导意义——与其堆视频生成质量,不如确保 latent 表征的时序相干性。

## 局限:别替它过度外推

论文自己承认的:世界模型固定生成 13 帧 horizon,下游任务必须保同样的未来视频 horizon 才能对齐它学到的时空动态,这限制了适应不同时间尺度任务的灵活性;pixel 级未来生成有冗余开销,未来要探索 latent 级预测表征做更高效的 action 生成;三类失败(Tilted stacking、Insertion misalignment、Zipper grasp slip)都是精度或硬件极限,第三人称视角对前后位移小偏移不敏感、UMI gripper 摩擦力不足。

我们读出的:门控是硬二选一(s_t>0.5),没有平滑过渡,在阶段边界附近可能抖动切换,论文没给门控在边界帧的稳定性分析;世界模型只用一路视角生成视频,其它视角喂 expert,多视角融合在特征层而非世界模型层,可能丢部分空间信息;所有 baseline 都在同一自采数据上 finetune,但 Wan+AnyPos 的 AnyPos 是 from scratch 训的,其它用预训练 checkpoint,baseline 之间预训练基础不完全对齐,公平性有细微偏差;96.95% 门控准确率是离线评测,实际闭环时门控基于实时观测,若观测因 OOD 退化门控可能误判阶段,论文未给门控在 OOD 下的准确率。

所以,HarmoWAM 是 WAM 路线解决泛化-精度 trade-off 的一个强证据,但它支持的是"双 expert + 门控分工"这个方向,不等于所有 WAM 任务都需要这么复杂的架构——对单一性质的任务(纯 transit 或纯 interaction),单 expert 可能更高效。

## 一句话收束

HarmoWAM 把 WAM 两条旧路线的结构性 trade-off,用一个共享世界模型提供隐含 latent 和显式视频两路条件、两个专门化 expert 分别管精度和泛化、一个过程自适应门控按任务阶段硬路由,统一了起来。它最有力的论证是 motivation 实验,而非跑分——把两条路线各自的失败根因钉死,然后用分工而非折中去解决。
