# OpenVLA: An Open-Source Vision-Language-Action Model

2024-09-05，Stanford 的 Moo Jin Kim 等人发布的论文，标题是 OpenVLA: An Open-Source Vision-Language-Action Model。

## 开场:这篇论文真正要解决什么

这篇论文要解决 VLA 落地的两个非常具体的采用障碍:**第一,现有 VLA 比如 RT-2-X 是闭源的,你看不到架构、看不到训练数据混合、看不到超参,更没法 fine-tune 它到自己任务;第二,prior work 完全没研究怎么高效 fine-tune VLA,尤其怎么在消费级 GPU 上跑。**

OpenVLA 的核心命题是:如果你做一个完全开源的 VLA——数据、权重、代码全公开——并且系统研究怎么用 LoRA 和量化在消费级硬件上 fine-tune 和部署它,那 VLA 就能从少数大公司的黑箱,变成社区能迭代的研究对象。

它走通了,顺带做到一件很打脸的事:一个 7B 的开源 VLA,在 29 个任务上平均超 55B 闭源 RT-2-X 16.5 个点,参数少 7 倍。这不是靠架构创新,是靠更仔细的数据、更好的视觉编码器、和把 LLM 生态工具搬过来。

## 它的输入和输出到底是什么

你需要在脑子里先建一个模块图。

输入有两路:**图像观测**,单张 224×224 RGB,第三人称视角。注意是单张,没有多视角,没有历史帧——这点论文自己承认是局限。**语言指令**,一个任务描述,套在一个模板里:"What should the robot do to {task}? A:"。这个模板是关键,它把动作预测框成了一个"回答问题"的 VLM 任务。

输出就一路:**7 维机器人动作**,单步。7 维是 delta 末端位姿——3 维平移、3 维 Euler 角、1 维夹爪。关键设计是这 7 维被离散成了 token,而非连续值。每维独立离散成 256 个 bin,bin 宽用训练数据的 1% 和 99% 分位定,这样抗 outlier。这 256 个 token 覆写 Llama tokenizer 词表里最后 256 个最少用的 token——因为 Llama 只留 100 个 special token 位,不够 256 个动作 token。

### 输入到底怎么拼成一条序列

光说两路输入还不够具体,我把它拆成一条显式的拼接序列,你在脑子里能建出来。

序列大致长这样:开头是图像,224×224,分别过 DINOv2 和 SigLIP 两个视觉编码器,各自出 patch 特征,channel-wise 拼起来,再经一个 2 层 MLP 投影到 Llama 的词嵌入空间。然后是语言指令,套在模板里,经 Llama tokenizer。最后是 7 个 action token,模型自回归生成,再 de-tokenize 回连续动作值。

这里有几个坑要特别说清,不然容易误解。

第一,language 和 action 都是 LLM 的序列 token,统一自回归。这点和后面 π0.7、Qwen-VLA 完全不同——后两者的 action 是 DiT 的连续张量,通过 hidden state 拼接和 VLM 耦合。OpenVLA 走的是最简单路线,action 直接走 LLM token 通道,用标准 next-token cross-entropy 训,不需要 DiT、不需要 flow matching、不需要额外 action expert。

第二,多视角不支持。单图输入,没有历史帧,论文自己承认这是局限。这是为了简单——224×224 单图,patch token 数控制在 256 左右,context 不会爆炸。

第三,训练和推理序列长得一样,都是单图加语言出 7 个 action token。但训练时 action token 是 ground-truth teacher forcing,推理时是模型自回归生成。而且没有 action chunk——每一步都要完整前向,推理就是 6Hz 左右。这是离散 token 路线的代价。

记住这条序列,后面听 π0.7、Qwen-VLA 你会看到明显的对比。OpenVLA 的答案是:context 是单图加语言,language 是 LLM token,action 也是 LLM token,统一自回归。简单,但牺牲了 action chunk 和连续动作多模态建模。

## 架构:Prismatic-7B VLM 直接 fine-tune

主干是 Prismatic-7B 这个 VLM。它由三部分组成:一个 600M 的融合视觉编码器,把 DINOv2 和 SigLIP 的特征 channel-wise 拼起来;一个 2 层 MLP projector,把视觉特征投到 Llama 词嵌入空间;一个 Llama 2 7B 作为 LLM backbone。

视觉编码器的融合是 OpenVLA 一个关键设计。DINOv2 是自监督学的,给低层空间特征,对精确抓取位姿有用;SigLIP 是语言对齐的,给高层语义,对选对物体有用。Prismatic 论文证明这个融合比单用 SigLIP 或 CLIP 在多物体语言 grounding 任务上强约 10 个点。这就解释了为什么 OpenVLA 7B 能超 RT-2-X 55B——融合视觉编码器补上了数据规模差距的一部分。

action 这边没有独立 expert。7 个 action token 直接走 LLM 的 token 通道,用 next-token cross-entropy 训,只在 action token 上算 loss。推理时 LLM 自回归生成 7 个 token,再 de-tokenize 回连续值,直接执行。

这个设计的核心思想是"最简单可扩展 VLA"。不引入 DiT、不引入 flow matching、不引入 action expert,直接把机器人动作预测框成 VLM 的 next-token 预测任务。好处是直接继承 LLM 生态的所有工具:FSDP 做分布式训练、AMP 做混合精度、FlashAttention 加速、HuggingFace 集成、LoRA 做参数高效 fine-tune、QLoRA 做量化。最小代码改动就能 scale 到 billion 参数。

## 给你一个数值 sense:这套模型到底多大

光说 7B 可能没感觉,我把维度念细一点。

LLM backbone 是 Llama 2 7B,标准 transformer decoder,32 层,每层 32 个注意力头,hidden dimension 4096,ffn 11008。这是 Llama 2 公开规格,不神秘。

视觉编码器 600M。DINOv2-base 是 patch size 14,embedding 768;SigLIP 也是 patch 14,embedding 1152。224×224 的图像过 patch 14 切成大概 256 个 patch token,每个视角一路,两路 concat 后经 MLP 投到 Llama 的 4096 维。

输入分辨率 224×224。这里有个细节——他们试过 384×384,没看到性能差异但训练慢 3 倍,所以选 224。在很多 VLM benchmark 上更高分辨率更好,但 VLA 上还没看到这个趋势,这跟机器人任务对像素细节要求没 VQA 那么高有关。

action 是 7 维单步。每维 256 bin,用 1% 到 99% 分位定 bin 宽,抗 outlier。7 个 token 覆写 Llama 词表最后 256 个最少用 token。无 action chunk,每步单 token 序列预测。

训练规模是这篇论文的一个亮点。64 张 A100 跑 14 天,总共 21500 A100-hours,batch size 2048,27 个 epoch。学习率 2e-5 固定,不带 warmup。训练到 action token accuracy 超 95% 才停。这几个数字里有几个反 LLM 直觉的发现,我后面单独讲。

推理上,bfloat16 精度下需要 15GB 显存,在一张 RTX 4090 上大概 6Hz——没编译、没推测解码,就是裸跑。4bit 量化能压到 7GB,在 A5000 上 3Hz。这个速度对 5Hz 的 BridgeData V2 任务刚好够用,但对 50Hz 的 ALOHA 就不够了——论文自己承认这是局限。

## 真正让 OpenVLA 成立的三个工程决策

OpenVLA 不是靠架构创新赢的,是靠三个工程决策。

**第一,融合视觉编码器。** DINOv2 加 SigLIP,不是单 SigLIP。这给低层空间细节加高层语义,在多物体语言 grounding 任务上比单一编码器强约 10 个点。这是 OpenVLA 7B 能超 RT-2-X 55B 的一个关键——RT-2-X 用的是 PaLI 的视觉编码器,没融合 DINOv2。

**第二,VLA 训练 recipe 反 LLM 直觉。** 他们在 BridgeData V2 上做了大量小规模消融,发现 VLA 训练和 LLM/VLM 训练有几个反直觉差异。一是 epoch 数——LLM/VLM 通常跑 1-2 个 epoch 就停,但 VLA 要跑 27 个 epoch,action token accuracy 要训到 95% 以上真实机器人性能才继续涨。二是 vision encoder 要 fine-tune——VLM 训练习惯冻结视觉编码器保预训练特征,但 VLA 必须解冻,因为预训练视觉不含足够细粒度空间细节做精确控制,冻结掉性能直接掉到 47%。三是学习率 2e-5 固定不带 warmup——这个学习率刚好和 VLM 预训练一样,warmup 没收益。这套 recipe 后来被后续 VLA 工作广泛参照。

**第三,LoRA 加 4bit 量化让 VLA 民主化。** 这是论文最实用的贡献。LoRA 应用到所有 linear 层,rank 32,只训 97.6M 参数,占 7B 的 1.4%。性能 68.2%,和 full fine-tune 的 69.7% 统计上持平。单张 A100 上 10-15 小时 fine-tune 一个任务,比 full fine-tune 省 8 倍算力。4bit 量化把显存压到 7GB,性能 71.9% 持平 bfloat16 的 71.3%。这让 OpenVLA 能在消费级 GPU 上 fine-tune 和推理,不再需要服务器集群。这是 VLA 从大公司专属走向社区可迭代的关键。

## 关键结果:7B 开源打 55B 闭源

我挑几个有对照的。

**BridgeData V2 WidowX 17 任务**:OpenVLA 70.6%,RT-2-X 55B 是 50.6%,Octo 只有 20%,RT-1-X 18.5%。这是 170 个 rollout 的结果,覆盖 5 类泛化——visual、motion、physical、semantic、language grounding。OpenVLA 在除 semantic 外所有类别超 RT-2-X。RT-2-X 在 semantic 泛化上仍领先,因为它 Internet 预训练规模更大,而且 co-fine-tune 时混了 Internet 数据保先验,OpenVLA 只在机器人数据上 fine-tune。

**Google Robot 12 任务**:OpenVLA in-distribution 85.0%、OOD 78.3%,RT-2-X 是 88.0% 和 72.0%。两者基本持平,都显著超 RT-1-X 和 Octo。注意 RT-2-X 是 55B,OpenVLA 是 7B,7 倍参数差距下持平甚至反超。

**29 任务总平均**:OpenVLA 比 RT-2-X 高 16.5 个绝对点。这是 abstract 里那个标志性数字。但要注意,这不是 OpenVLA 架构碾压 RT-2-X,是数据更多(970k vs 350k)、视觉编码器更好(DINOv2+SigLIP 融合)、数据清洗更仔细(比如过滤了 Bridge 的全零动作)的复合结果。

**Franka fine-tune 7 任务**:OpenVLA 是唯一所有任务都达到 50% 以上的方法,aggregate 最高。Diffusion Policy 在窄单指令任务上更强(动作平滑精确,有 action chunk 和 temporal smoothing),但 OpenVLA 在多样多指令任务上更强(语言 grounding 好)。这给了一个实用结论:OpenVLA 是 downstream fine-tune 的强默认初始化,尤其任务涉及多样语言指令时;如果是窄但高灵巧的任务,Diffusion Policy 仍可能更好。

**LoRA 和 4bit 量化**:LoRA rank 32 性能 68.2% 持平 full FT 69.7%,只训 1.4% 参数。4bit 量化性能 71.9% 持平 bfloat16 71.3%,显存 7GB。这两个数字是 VLA 民主化的硬证据。

## 一个最值得记住的 insight

论文里有一个发现我建议你重点记:**VLA 训练反 LLM 直觉,但反得有道理。**

具体说就是三件事:27 个 epoch 而不是 1-2 个;fine-tune vision encoder 而不是冻结;lr 2e-5 不带 warmup。

为什么反直觉但合理。LLM/VLM 训练用 1-2 个 epoch,是因为 Internet 数据太大会过拟合,一次过够。但机器人数据只有 970k,远小于 Internet,且 action 分布比文本复杂——每个 action token 要学的是连续动作的精细映射,需要更多迭代让 action token accuracy 上去,95% 是个经验阈值。vision encoder 要解冻,是因为预训练视觉特征是为 VQA 这种语义任务学的,不含机器人需要的细粒度空间细节,比如夹爪和物体接触点的精确相对位姿,这些必须 fine-tune 才能学到。lr 2e-5 不带 warmup,是因为这个学习率刚好和 VLM 预训练一致,模型已经在稳定区间,warmup 多余。

这个 insight 分量很重。它给了后续 VLA 工作一个可复现的 recipe 起点。π0.7 的 knowledge insulation、Qwen-VLA 的 T2A 分阶段训练,本质上都是在解 OpenVLA 暴露的"VLM 已预训练而 action 模块要学"这个不对称——只是解法不同。OpenVLA 是最直接的,直接 fine-tune 视觉加 LLM 一起;π0.7 用 knowledge insulation 隔离梯度;Qwen-VLA 用 T2A 先建 action 先验。这是 VLA 训练方法论的演进脉络。

## 局限:别替它过度外推

论文自己承认的:只支持单图观测,无多视角无 proprio 无历史帧,真实机器人异质传感器输入未支持;推理吞吐不足以支撑高频控制(如 ALOHA 50Hz),action chunk 或推测解码是潜在解;可靠性仍未达高(典型 <90% 成功率);因算力限制,base VLM 规模影响、VL+action co-training、最佳视觉特征等设计问题未充分探索。

我们读出的:无 action chunk 每步完整前向,6Hz 在高频任务受限,Diffusion Policy 用 T=16/X=8 的 chunk 在窄灵巧任务上动作更平滑,OpenVLA 没集成这个;离散 action token 有 256 bin 量化损失,且无 action 多模态建模——同一状态多种合理动作会被平均,后续 π0/Qwen-VLA 用 flow matching 正是补这个;970k OpenX 虽多样但仍是机器人示教,没像 π0.7 那样混 autonomous、失败、人类视频、web 多模态数据,数据多样性有上限;DROID 试过 10% mixture weight 但 action token accuracy 低被移除,暗示 OpenVLA 对高多样性大数据集拟合不足,可能需要更大模型或更高 weight。

所以,OpenVLA 是 VLA 开源化的里程碑,但它支持的是"VLM fine-tune + 离散 action token + LoRA/量化"这条简单路线,不等于所有 VLA 都该这么做。它的简单是优点也是上限——后续工作继承它的 VLM fine-tune 框架,但几乎都换成了连续 action expert 来补 action chunk 和多模态。

## 一句话收束

OpenVLA 把 Prismatic-7B VLM 在 970k Open X-Embodiment 数据上 fine-tune,把 7 维动作离散成 256 个 token 塞进 Llama 词表用 next-token 预测训,7B 开源 VLA 在 29 任务上超 55B 闭源 RT-2-X 16.5 个点,且用 LoRA(1.4% 参数)和 4bit 量化(7GB 显存)在消费级 GPU 上 fine-tune 和推理。它最有力的论证在跑分之外——把 VLA 从大公司闭源黑箱变成了社区可迭代的开源对象,后续几乎所有 VLA 工作都建立在它奠定的 VLM fine-tune 框架之上。
