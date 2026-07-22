# FAST — 结构化卡片

> 论文:**FAST: Efficient Action Tokenization for Vision-Language-Action Models**(Physical Intelligence · UC Berkeley · Stanford, arXiv 2501.09747, 2025-01-16)
> 配套:`card.json`(机器可读) · `architecture.md`(架构图) · `podcast.md`(口播稿)

## 一句话定位

用离散余弦变换(DCT)+BPE 把高频机器人动作序列压缩成短 token,让自回归 VLA 第一次能在灵巧高频任务上训起来,并且比 diffusion VLA 训练快 5 倍。

## 它要解决的问题

**问题**:自回归 VLA(把动作离散成 token、走 next-token prediction 的那一类)在低频任务(BridgeV2/RT-1)上工作,但一上高频灵巧任务(T恤折叠 50Hz、收桌 20Hz、DROID 15Hz)就完全训不动。

**为什么 prior work 不够**:之前所有自回归 VLA(RT-2、OpenVLA)都用 per-dimension、per-timestep binning:每维每步独立分到 256 个 bin,1 秒 50Hz 双臂 14 维要产生 700 个 token。高频信号相邻 token 高度相关,"复制上一个动作"就能拿到极低 next-token loss,模型陷入 trivial local optimum,真正该学的细微变化学不到。根因在 tokenization 让学习信号接近零,与模型容量无关。diffusion VLA(π0)绕开了 tokenization,但要训 300M action expert,训练算力贵,且在 DROID 上会忽略语言指令。

## 输入 / 输出

| 方向 | 名称 | 类型 | 说明 |
|---|---|---|---|
| 输入 | 图像 | image 224×224 | 1 张第三视角 + 每臂 1 张腕部;各自过视觉编码器后 token 拼接 |
| 输入 | 语言指令 | text | LLM 自带 tokenizer 编码 |
| 输入 | proprio | vector | 256-bin 离散化成整数串,当文本 token(输入侧,不进 loss) |
| 输出 | 动作 token 序列 | discrete | FAST 压缩后 ~30 tokens/秒/臂,覆盖 1 秒 chunk |
| 输出 | 动作 chunk | continuous | detokenize 后的 1 秒动作,维度=|A|,步数=控制频率 |

控制频率:5~50Hz 覆盖;chunk 固定 1 秒;推理 π0-FAST ~750ms/chunk(diffusion π0 ~100ms)。

### 输入拼接 protocol

FAST 是一个 action 编解码器,而非序列拼接方案。它的"protocol"是 token 在主干里的进出流程:

```
<bos_vision>[VisEnc(img_3rd) ⊕ VisEnc(img_wrist1) (⊕ VisEnc(img_wrist2))]<eos_vision>
<bos_text>TokLLM(instruction) ⊕ TokLLM(proprio_binned)<eos_text>
<bos_action>[FAST(action_chunk_1s) = T_1, ..., T_n]<eos_action>
```

逐段解释:

- `<bos_vision>...<eos_vision>`:多张图各自过预训练视觉编码器,token 在序列维拼接(不是通道维)。DROID 设置:1 张第三视角 + 1 张腕部;训练时第三视角从 DROID 提供的两个外视角里随机选,推理能换视角、不需要相机标定。
- `<bos_text>...<eos_text>`:语言指令 + proprio 都走 LLM 自己的文本 tokenizer。proprio 被预先 256-bin 离散化成整数串,再当文本 token。作者强调:proprio 是输入,不参与 next-token loss,所以用 binning 完全没问题——问题只在"要预测的动作输出"。
- `<bos_action>...<eos_action>`:这是 FAST 真正改造的地方。原始 1 秒 action chunk(维度 |A|、步数=控制频率)经过 [量化归一化 → DCT → scale-and-round → 低频优先 flatten → BPE] 压成 n 个动作 token,直接覆盖 LLM 词表里最少用的 n 个 token。推理时模型自回归吐出这串 token,再 detokenize 回连续动作。

**关键**:chunk 内是开环执行 1 秒动作,不接新观测——这是和 DreamZero 闭环 KV-cache 替换完全不同的控制范式。

## 数据集

| 数据 | 规模 | 备注 |
|---|---|---|
| Libero (sim) | 270k samples | Spatial/Object/Goal/Long 合并,40k iter/40 epoch |
| Table Bussing | 1 dataset | UR5 单臂 20Hz,~70 物体;评估 12 物体未见配置 |
| T-Shirt Folding | ~150 shirts | ARX 双臂 50Hz;最难高频灵巧任务 |
| Grocery Bagging | — | UR5 单臂 20Hz,7 物品装袋 |
| Toast out of Toaster | — | Trossen ViperX 双臂 50Hz |
| Laundry Folding | — | ARX 双臂 50Hz;最难 long-horizon |
| DROID | 75k episodes / 21M samples | Franka 单臂 15Hz;首次完全未见环境 zero-shot 部署 |
| FAST+ 通用 tokenizer 训练集 | 1M 1-秒 chunks | 跨本体:单臂/双臂/移动/dex/humanoid/nav;joint/EEF/cam-frame;5~60Hz |

## 架构(详见 architecture.md)

- **主干**:π0 (PaliGemma-3B);消融用 OpenVLA (Prismatic-7B)
- **FAST 本身**:参数量≈0(只学一个 BPE 词表)
- **结构**:autoregressive next-token VLA + 外挂 DCT-BPE 动作编解码器
- **多视角**:每张图独立编码后 token 拼接

**为什么这样设计**:根因诊断是 next-token 学习信号正比于 T_i|T_{1:i-1} 的边际信息量;高频 binning 让相邻 token 高度相关,边际信息趋零。解法不是堆模型容量,是先把动作压成低相关短序列。DCT 是分析式、确定、近零训练成本的连续信号压缩(JPEG 同款),BPE 再把稀疏 DCT 系数无损压成定长词表——整个 tokenizer 只有两个超参(γ、BPE 词表大小),远比 VQ-VAE/FSQ 简单可控。低频优先 flatten 让模型先预测决定整体形状的低频系数,rollout 更稳。

### 数值 sense

| 项 | 值 |
|---|---|
| BPE 词表大小 | 1024(单数据集与 FAST+ 通用版都用) |
| 动作 token 表示 | LLM 词表里的整数 ID(覆盖最少用的槽位) |
| 压缩比 vs naive | Bridge 1.75× / DROID 3.6× / Bussing 5.0× / Shirt Fold 13.2× |
| tokens/chunk | FAST 稳定 ~30/秒/臂(双臂 ~60),与频率无关 |
| 主干 | π0 = PaliGemma-3B;OpenVLA = Prismatic-7B |
| 图像 | 224×224,每张独立编码后 token 拼接 |
| action chunk | 1 秒;步数=频率;维度 \|A\| 从 7(单臂)到 40(humanoid H1) |
| rounding scale γ | 10(所有单数据集实验同值) |
| 训练 | π0-FAST 5× fewer GPU hours 比 diffusion π0;DROID:240k iter × batch 256,3 epoch,8×H100 约 4 天;lr 5e-5, AdamW, EMA 0.999 |
| 推理延迟 | π0-FAST ~750ms/chunk;diffusion π0 ~100ms/chunk |

**给听众的标尺**:1024 词表、γ=10、~30 tokens/秒/臂、1 秒 chunk、3B 主干、8×H100 训 4 天(DROID)。压缩比是 FAST 最该记住的数字——它随频率涨,binning 是线性涨、FAST 是常数。

## 关键技术

1. **DCT 频域压缩做 action tokenization** — 对每维动作独立做 DCT,把高频冗余从时域搬到频域的少数低频系数,从根上恢复 next-token 的边际信息量。toy 实验证明:同数据同模型,只换 DCT 压缩,高频误差从陡升变平。
2. **BPE 无损压缩稀疏 DCT 系数** — 按"所有维最低频先排"flatten,再 BPE 把频繁系数组合 merge 成 1024 词表。lossless 可逆,直接并入 LLM 词表。消融证明去 BPE 策略仍优于 binning 但明显差于带 BPE。
3. **FAST+ 通用预训练 tokenizer** — 在 1M 跨本体 chunks 上预训练 BPE 词表,发布成 HuggingFace AutoProcessor。新机器人只需 quantile 归一化后直接 tokenize。FAST+ 压缩比与频率无关,逼近动作信号内在复杂度,可黑盒用于新机器人。
4. **token 覆盖策略复用 LLM 词表** — FAST 输出 token 直接覆盖 LLM 词表最少用的槽位,主干全参数 fine-tune,不需改 backbone 结构。OpenVLA 消融证明 FAST 与 backbone 解耦。
5. **π0-FAST 匹敌 diffusion VLA** — 自回归路线第一次在性能和算力效率上同时追平 diffusion π0;compute-matched 下更强;DROID 上语言跟随更好。代价是推理慢(750ms vs 100ms)。

## 关键结果

| 指标 | FAST | 最强 baseline | setup |
|---|---|---|---|
| token 压缩比 (Shirt Fold 50Hz 双臂) | 13.2× (700→53) | naive 700 tokens | 1-秒 chunk,14 维 @50Hz |
| token 压缩比 (Bussing 20Hz 单臂) | 5.0× (140→28) | naive 140 tokens | 1-秒 chunk,7 维 @20Hz |
| T-Shirt Folding 成功率 | 可训出有效策略 | naive binning 0%(无法训) | ARX 双臂,π0 backbone |
| Table Bussing 收敛速度 | 3× fewer steps 达同等 | diffusion π0 | UR5 20Hz,大数据集 |
| π0-FAST vs diffusion π0 generalist | 匹敌(5 任务平均持平或略胜) | diffusion π0 | 10k 小时跨本体数据 |
| 训练算力 | 5× fewer GPU hours | diffusion π0 | 同性能点 |
| compute-matched generalist | π0-FAST 明显胜出 | compute-matched diffusion π0 | 同 GPU 小时,5 任务 |
| OpenVLA + FAST (T-Shirt Fold) | 显著提升,可训 | OpenVLA 原生 binning(训不动) | 证明 FAST 与 backbone 解耦 |
| 推理延迟 | π0-FAST ~750ms/chunk | diffusion π0 ~100ms/chunk | NVIDIA 4090;FAST 主要短板 |
| DROID zero-shot 跨校园 | 首次完全未见环境零样本 | 原 DROID paper/OpenVLA 只做 co-training/fine-tune | Berkeley/Stanford/U.Washington |

## Insights

1. 根因不在模型容量,在 tokenization:next-token 学习信号正比于 T_i|T_{1:i-1} 的边际信息量;高频 binning 让相邻 token 高度相关,边际信息趋零,模型学到"复制上一步"就够低 loss,真正细微变化学不到(p3, Figure 3 toy 实验)。
2. 压缩比与频率无关是关键信号:FAST 在所有数据集上稳定 ~30 tokens/秒/臂,说明它逼近了动作信号的内在复杂度,而 binning 的 token 数随频率线性涨——binning 是在编码冗余,FAST 是在编码信息(p7 Table I)。
3. FAST 不是"压得最狠"的,是"高保真下仍能压"的:低 fidelity 区 VQ 方法压得更狠,但向高 fidelity scaling 时 FAST 远胜——这决定了 FAST 适合精细控制,VQ 适合粗略重建(p16 Figure 12)。
4. 自回归 VLA 的语言跟随比 diffusion VLA 更好:DROID 评测中 diffusion π0 经常忽略语言直接抓物体,π0-FAST 听语言。作者未深究但留作 future work(p9)。
5. FAST 把"换 tokenizer"从工程杂活变成了 VLA 架构选择的关键杠杆:一个几乎零参数的 DCT-BPE 编码器,让自回归 VLA 在性能和算力上同时追平 diffusion VLA(p10-11)。
6. FAST+ universal tokenizer 把"每数据集重训 BPE"变成"即插即用",降低了自回归 VLA 的使用门槛,可能比单个性能数字更有长期影响(p6)。

## vs 同类工作

- **vs naive binning (RT-2/OpenVLA)**:binning 是 per-dim per-step 独立分 bin,不利用时间相关性,高频下 token 数线性涨、学习信号趋零;FAST 用 DCT 把时间冗余搬到频域稀疏系数,再 BPE 压成定长短序列,token 数与频率解耦。
- **vs FSQ/VQ-VAE**:VQ 要训 encoder-decoder,超参敏感、低 fidelity 区压得狠但高 fidelity scaling 差;FAST 是分析式(DCT)+ lossless(BPE),几乎免训练,高保真下优势明显(p16 Figure 12)。
- **vs diffusion VLA (π0)**:diffusion π0 用 300M action expert + flow matching 绕开 tokenization,推理快(100ms)但训练算力贵、DROID 上忽略语言;π0-FAST 训练快 5×、compute-matched 下更强、语言跟随更好,代价是推理慢(750ms)。两条路线 trade-off 被清晰量化。
- **vs semantic action reps (keypoints/lang sub-tasks)**:那些方法样本高效但需 hand-designed 低层控制器,泛化受限;FAST 直接输出低层控制命令,不依赖任务特定设计,是更通用的 low-level tokenization。

## 局限

**论文自承**:
- 推理慢。π0-FAST ~750ms/chunk(diffusion π0 ~100ms),因为要自回归解码 30~60 个 token 且用全 2B backbone。静态任务可接受,动态任务会是问题(p9)。
- 只在静态操作器上做了真实 policy 评测;FAST+ 在 mobile/dex/humanoid 上只做了离线压缩测试,真实 policy 性能未验证(p11)。
- autoregressive vs diffusion 哪个 VLA 架构更优仍未定,需更系统对比训练速度/语言 grounding/表达力(p11)。

**我们读出**:
- chunk 内开环执行 1 秒动作,不接新观测——和 DreamZero 闭环 KV-cache 替换完全不同,在需要中途纠错的动态任务上可能受限(论文未测高频动态任务如接球)。
- γ=10 是手调超参,论文说"不敏感"但只给了一组数据集上的扫描图,跨本体/跨任务最优 γ 是否一致未充分论证。
- compression 是 lossy 的,论文报告的压缩比是"comparable reconstruction accuracy"下的,但 reconstruction error 到 policy 性能的映射没给——高保真 ≠ 高 policy 性能,这条链路论文留白。

## 可复现性

- 代码:https://pi.website/research/fast (HuggingFace AutoProcessor: `physical-intelligence/fast`)
- 权重:FAST+ universal tokenizer 开源;π0-FAST policy 检查点随 π0 release
- 仿真基准:Libero (Spatial/Object/Goal/Long)
- 真机评测:UR5 单臂 / ARX 双臂 / Trossen 双臂 / Franka(DROID)+ 三校园 zero-shot 部署

## 论文重要图(详见 `figures.md`)

| 图 | 页 | 重要性 | 一句话 |
|---|---|---|---|
| [Figure 4](../../extracted/fast/pages/p04.png) | 4 | key | FAST tokenizer 五步流水线,全文最该看 |
| [Figure 3](../../extracted/fast/pages/p03.png) | 3 | key | toy 实验干净证明根因是 tokenization 不是模型/数据 |
| [Figure 2](../../extracted/fast/pages/p02.png) | 2 | key | 频率 vs 分数,binning 高频崩 FAST 全频稳 |
| [Figure 6](../../extracted/fast/pages/p08.png) | 8 | key | 四种 tokenizer 主结果,FAST 最强 FAST+ 追平专属 |
| [Figure 7](../../extracted/fast/pages/p08.png) | 8 | key | DROID 首次完全未见环境 zero-shot,三校园部署 |
| [Figure 9](../../extracted/fast/pages/p09.png) | 9 | key | FAST vs diffusion π0 单任务,大数据集收敛快 3× |
| [Figure 11](../../extracted/fast/pages/p10.png) | 10 | key | π0-FAST 匹敌 diffusion π0,训练快 5× |
| [Figure 15](../../extracted/fast/pages/p18.png) | 18 | key | compute-matched 下 π0-FAST 明显胜出 |
| [Figure 1](../../extracted/fast/pages/p01.png) | 1 | key | 门面图,5× 训练加速 |
| [Figure 10](../../extracted/fast/pages/p10.png) | 10 | supportive | Laundry Folding rollout,FAST 解锁最难 long-horizon |
| [Figure 8](../../extracted/fast/pages/p08.png) | 8 | supportive | FAST+ 跨本体压缩比 |
| Figure 5/12/13/14 | — | supportive | 见 figures.md |

## 标签

`VLA` `action tokenization` `DCT` `BPE` `autoregressive` `high-frequency control` `π0` `Physical Intelligence` `compression`
