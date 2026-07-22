# FAST: Efficient Action Tokenization for Vision-Language-Action Models

2025-01-16，Physical Intelligence 的 Karl Pertsch 等人发布的论文，标题是 FAST: Efficient Action Tokenization for Vision-Language-Action Models。

## 开场：这篇论文真正要解决什么

FAST 是一个**动作 tokenizer**——决定"连续的机器人动作信号，怎么变成离散 token 喂给自回归 VLA"的那个组件。它不是一个新模型，也不是一个新的训练方法。但就是这么个看起来很工程的小东西，作者发现它才是自回归 VLA 在高频灵巧任务上完全训不动的根因。

问题长这样：自回归 VLA，也就是 RT-2、OpenVLA 那一脉，把动作离散成 token、走 next-token prediction，在低频任务上工作得不错——5Hz 的 BridgeV2、RT-1 都没问题。但一上高频灵巧任务，50Hz 的 T 恤折叠、20Hz 的收桌、15Hz 的 DROID，就完全训不动。同一套模型架构、同样的数据规模，低频能学，高频学不会。

diffusion VLA 的 π0 绕开了 tokenization，用 300M 的 action expert + flow matching 直接出连续动作，能跑高频，但训练算力贵，而且在 DROID 上还经常忽略语言指令。

作者要回答的问题是：**自回归 VLA 在高频失败的根因到底是什么？能不能不换架构、不堆参数，就把这个根因修掉，让自回归路线重新有竞争力？**

## 根因诊断：不是模型不够大，是学习信号被 tokenization 吃掉了

这是这篇论文最干净的一段论证，值得讲透。

自回归模型训练时，学的是"给定前面所有 token，预测下一个 token"。所以它的学习信号，正比于第 i 个 token 给定前面 i-1 个 token 的**边际信息量**——也就是这个 token 有多"出乎意料"。

现在看传统 binning 怎么做：每个 action 的每个维度、每个时间步，独立分到 256 个 bin 之一。一个 1 秒的 action chunk，50Hz、双臂 14 维，就是 50 × 14 = 700 个 token。问题是高频信号相邻时间步高度相关——50Hz 下相邻两个动作差值极小。这意味着什么？意味着第 i 个 token 给定第 i-1 个 token 的边际信息量趋近于零。"复制上一个动作"就能拿到极低的 next-token loss。

模型于是学到一个 trivial local optimum：直接抄上一步。它拿到了低 loss，但真正该学的细微变化一点没学。这就解释了为什么 binning 在高频崩——根因是 tokenization 把学习信号消灭了,与模型容量无关。

论文用一个非常漂亮的 toy 实验证明这点（Figure 3）。任务很简单：预测过 4 个随机点的三次样条。作者只改一个变量——采样率，从 25 到 800。数据分布不变、模型容量不变、训练步数不变，只换采样率。结果：binning 的预测误差随采样率升高陡升，最后模型直接抄第一个动作；FAST 全程平稳。这是论文最干净的因果论证：根因就是 tokenization，不是别的。

## FAST 的解法：先把动作压成短序列，再 next-token

既然根因是相邻 token 太相关，解法就很直接：先把动作信号**压缩**成低相关的短 token 序列，再喂给自回归模型。

FAST 的全名是 Frequency-space Action Sequence Tokenization，五步流水线，对应 Figure 4，我按顺序讲。

**第一步，quantile 归一化**。把每个 action 维度的训练数据，1st 到 99th 分位映射到 -1 到 1 区间。用分位数不用 min/max，是为了抗离群点——大机器人数据集偶有异常动作。这步还顺带让不同本体的数据（动作 scale 不一样）能统一处理。

**第二步，离散余弦变换，DCT**。对每个 action 维度独立做 DCT，把时域信号转到频域。DCT 把信号表示成一组余弦基的加权和：低频系数承载整体形状，高频系数承载细节。因为机器人动作通常平滑，DCT 后大部分能量集中在少数低频系数，高频系数权重极小——这是 JPEG 压缩同款性质，不是作者发明的，是借来的。

**第三步，scale-and-round 量化**。把 DCT 系数乘以 γ（默认 10）后四舍五入。高频小系数被量化成 0，得到一个稀疏整数矩阵。γ 是唯一控制"压缩 vs 重建"权衡的超参。

**第四步，低频优先 flatten**。把 |A|×H 的稀疏矩阵展平成 1D 序列，但顺序很讲究：所有维度的最低频系数先排，再所有维度的次低频，以此类推。为什么这样？因为自回归模型要先预测决定整体形状的低频系数，等全局形状建立了再预测高频细节，rollout 才稳。如果按行优先（先把单维所有频率排完），模型在还没建立全局形状时就被迫预测细节，会乱。

**第五步，BPE 无损压缩**。对 flatten 后的整数序列训一个 byte-pair encoding 词表，默认 1024 大小。BPE 把频繁共现的系数组合 merge 成单 token，把稀疏的 0 全压掉。这步是 lossless、可逆的，产出的词表能直接并入 LLM 既有词表。

整个流水线可逆：detokenize 时 BPE 逆→unflatten→inverse DCT→denormalize，精确还原连续动作。

注意，整个 FAST 只有两个超参：rounding scale γ 和 BPE 词表大小。论文说都不敏感，所有单数据集实验都用同一组值。这跟 VQ-VAE、FSQ 这种学习式压缩比，简单太多了——那些要训一个 encoder-decoder 网络，超参敏感，重建质量还依赖数据集。

## 输入到底怎么拼成一条序列

FAST 不是一个序列拼接方案，它是一个 action 编解码器，外挂在 VLA 主干上。我把 token 在主干里的进出流程写成一条显式序列，你在脑子里能建出来。

序列大致长这样：开头是 `<bos_vision>`，里面是多张图——一张第三视角、每臂一张腕部相机，每张图各自过预训练视觉编码器，得到的 token 在序列维拼接（注意是序列维，不是通道维）。然后 `<eos_vision>`。接着 `<bos_text>`，里面是语言指令，加上 proprio 状态——proprio 被预先 256-bin 离散化成整数串，再当文本 token 编码进去。`<eos_text>`。最后 `<bos_action>`，这是 FAST 真正改造的地方，原始 1 秒 action chunk 经过五步流水线压成 n 个动作 token，`<eos_action>`。

这里有几个坑要特别说清，不然容易误解。

第一，**proprio 用 binning 完全没问题**。作者专门强调这点。binning 在动作输出上崩，是因为动作输出要参与 next-token loss，高频让学习信号消失。但 proprio 是输入，不参与 next-token loss，用简单 binning 就够了。不要把"binning 在动作上崩"推广成"binning 哪里都不行"。

第二，**动作 token 是直接覆盖 LLM 词表里最少用的那些槽位**。这是 RT-2、OpenVLA 的标准做法，FAST 沿用。好处是不需要改 backbone 结构，任何预训练 autoregressive transformer 都能直接接 FAST。论文专门做了 OpenVLA 消融证明这点：把 OpenVLA 原生 binning 换成 FAST，T 恤折叠从训不动变成能训。这说明 FAST 和 backbone 是解耦的。

第三，**训练和推理时 action 这段的形态不一样**。训练时，action chunk 是教师强制的 ground truth，FAST 编码一次性给出整串 target token，模型学"看到图像+语言+proprio，吐出这串 token"。推理时，模型从 `<bos_action>` 起逐 token 贪心解码（双臂任务加个 0.7 的温度帮助跳出 home position），吐完 n 个 token 后 detokenize 成 1 秒连续动作，**开环执行整段 chunk**。

这就是一个关键区别：FAST 的 chunk 内是开环的，1 秒动作执行期间不接新观测。这跟 DreamZero 那种闭环 KV-cache 替换完全不同。在需要中途纠错的动态任务上，开环 1 秒可能是个限制——论文没测高频动态任务，比如接球。

## 给你一个数值 sense：这套 tokenizer 到底多大

光说"1024 词表"可能没感觉，我把维度念细一点。

BPE 词表是 1024，单数据集实验和 FAST+ 通用版都用这个大小。动作 token 本身就是 LLM 词表里的整数 ID，覆盖那些最少用的槽位。

压缩比是 FAST 最该记住的数字。论文 Table I 给了四个数据集的对比，全部用 1 秒 chunk、对比相同重建精度下的 token 数。BridgeV2 是 5Hz 单臂 7 维，naive 35 个 token，FAST 20 个，压缩 1.75 倍。DROID 是 15Hz 单臂 7 维，naive 105 个，FAST 29 个，压缩 3.6 倍。Bussing 是 20Hz 单臂 7 维，naive 140 个，FAST 28 个，压缩 5 倍。Shirt Fold 是 50Hz 双臂 14 维，naive 700 个，FAST 53 个，压缩 13.2 倍。

看出规律了吗？频率越高、维度越多，FAST 的压缩比越大。但更关键的数字是这个：**FAST 在所有数据集上稳定在每秒每臂约 30 个 token**，几乎与频率无关。这说明什么？说明 FAST 逼近了动作信号的内在复杂度——它编码的是信息，不是冗余。而 binning 的 token 数随频率线性涨，它编码的是冗余。

主干方面，π0 用的是 PaliGemma-3B，3B 参数；消融实验用 OpenVLA，Prismatic-7B。图像输入 224×224，每张图独立过视觉编码器后 token 拼接。FAST 本身参数量约等于零，只学一个 BPE 词表。

action chunk 固定 1 秒，步数等于控制频率，维度 |A| 从单臂的 7 到 humanoid H1 的 40 都有。rounding scale γ 默认 10，所有单数据集实验同值。

训练规模：π0-FAST 在 10k 小时跨本体数据上训练，比 diffusion π0 少用 5 倍 GPU 小时。DROID 那个设置具体点：240k 迭代 × batch 256，跑 3 个 epoch，8 张 H100 大约 4 天。学习率 5e-5，AdamW 优化器，EMA 权重 0.999。

推理延迟是 FAST 的主要短板：π0-FAST 大约 750 毫秒一个 chunk，因为要自回归解码 30 到 60 个 token，还得用完整的 2B backbone。对比之下 diffusion π0 只要 100 毫秒——它只要 10 步去噪，外加只跑 300M 的 action expert。这个 7.5 倍的推理速度差距，是 FAST 用自回归换来的代价。

记住这套数字，后面听别的 VLA 论文时，可以拿它当标尺：1024 词表、γ=10、30 token/秒/臂、1 秒 chunk、3B 主干、8×H100 训 4 天。压缩比随频率涨，binning 线性涨、FAST 是常数——这是 FAST 最有信息量的一个数字。

## FAST+：把"每数据集重训 BPE"变成"即插即用"

整个 FAST 只有一个需要训练的组件——BPE 词表。每个新数据集都要重训这个词表，虽然只要几分钟，但增加使用摩擦。

作者于是训了一个 FAST+，通用 tokenizer。在 100 万条 1 秒跨本体 action chunks 上预训练一个 BPE 词表，覆盖单臂、双臂、移动、dex 手、humanoid、nav，joint、EEF、cam-frame 各种动作空间，5 到 60Hz 各种频率。发布成 HuggingFace 的 AutoProcessor，三行代码就能用。

为什么通用化能成立？因为 FAST 的压缩比与频率无关、稳定 30 token/秒/臂，说明它逼近了动作信号的内在复杂度，不依赖具体本体。实测 FAST+ 在所有没参与训练的数据集上都至少 2 倍压缩，部分到 10 倍。而且 policy 训练性能追平数据集专属 FAST。

这点我特别想强调。FAST+ 的存在意义，可能比单个性能数字更有长期影响。它把"换个机器人就要重训 tokenizer"这个使用门槛拆掉了，让自回归 VLA 真正变成即插即用。这对整个 VLA 社区的 Adoption 是实打实的贡献。

## 关键结果：不仅赢，还在算力效率和语言跟随上赢

我挑几个有对照的数字。

**压缩比**：T 恤折叠 50Hz 双臂，13.2 倍，从 700 个 token 压到 53 个。这是最高的，因为频率高、维度多。收桌 20Hz 单臂，5 倍。压缩比随频率涨，但 FAST 的绝对 token 数稳定。

**T 恤折叠成功率**：FAST 能训出有效策略，naive binning 是 0——完全训不动。这是定性结论：FAST 解锁了 binning 完全做不了的任务。

**收桌收敛速度**：FAST 用 3 倍更少的训练步数达到 diffusion π0 同等水平。在大数据集上 FAST 收敛明显快。

**π0-FAST vs diffusion π0 generalist**：5 个 generalist 任务平均，π0-FAST 匹敌甚至略胜 diffusion π0，但训练用 5 倍更少 GPU 小时。这是论文最终卖点：性能追平、算力效率胜出。

**compute-matched 对比**：两边用同样 GPU 小时，π0-FAST 在 5 任务平均上明显胜出。这补充论证的含义是"同算力下 FAST 更强"（而非"FAST 用更少算力达到同等"）——根因是收敛更快。

**OpenVLA + FAST**：把 OpenVLA 原生 binning 换成 FAST，T 恤折叠从训不动变成能训。证明 FAST 与 backbone 解耦，能移植到任何预训练 autoregressive transformer。

**DROID zero-shot 跨校园**：这是 DROID 数据集第一次被训出"完全未见环境零样本"policy。原 DROID paper 和 OpenVLA 都只做 co-training 或 fine-tuning 评测。FAST 训出的 policy 在 Berkeley、Stanford、U.Washington 三地部署，能做 pick-and-place、开关抽屉、开水龙头，失败 trial 也表现合理（比如接近微波炉门把手）。

**推理延迟**：750 毫秒 vs diffusion π0 的 100 毫秒。这是 FAST 必须承认的短板。

还有个有意思的观察：在 DROID 评测里，diffusion π0 经常忽略语言指令，直接伸手抓物体；π0-FAST 听语言。作者没深究原因，留作 future work，但这暗示自回归 VLA 的语言 grounding 可能比 diffusion VLA 更紧——这对很多应用是关键。

## 一个最值得记住的 insight

论文里有个判断，我建议你重点记：**FAST 的压缩比与频率无关，稳定在每秒每臂约 30 个 token**。

这句话分量很重。它意味着两件事。第一，FAST 逼近了机器人动作信号的内在复杂度——不管你 5Hz 还是 50Hz、单臂还是双臂、关节空间还是末端执行器，1 秒动作的信息量是相对恒定的。第二，binning 的 token 数随频率线性涨，说明 binning 在高频下编码的全是冗余，不是信息。

这个 insight 给了 tokenization 设计一个非常清晰的方向标：好的动作 tokenizer 应该让 token 数与信号的内在复杂度挂钩，而不是与采样率挂钩。FAST 用 DCT 实现了这点，但它不一定是唯一解——任何把高频冗余搬走的压缩方法（包括未来的学习式方法）只要满足这条，理论上都能让自回归 VLA 在高频工作。这是论文最有普适意义的一个判断。

## 局限：别替它过度外推

论文自己承认的：推理慢，750 毫秒一个 chunk，diffusion π0 只要 100 毫秒。静态任务可接受，动态任务会是问题，论文专门点出这点留作 future work。还有只在静态操作器上做了真实 policy 评测，FAST+ 在 mobile、dex 手、humanoid 上只做了离线压缩测试，真实 policy 性能没验证。autoregressive vs diffusion 哪个 VLA 架构更优仍未定。

我们读出的：chunk 内开环执行 1 秒动作，不接新观测。这和 DreamZero 闭环 KV-cache 替换是完全不同的控制范式。在需要中途纠错的动态任务上可能受限——论文没测像接球这种高频动态任务，所以"FAST 让自回归 VLA 重新能打"这个结论，目前只在静态操作范围内成立。

γ=10 是个手调超参。论文说"不敏感"，但只给了一组数据集上的扫描图，跨本体、跨任务的最优 γ 是否一致，没充分论证。

还有一点比较微妙：compression 是 lossy 的——量化丢高频。论文报告的压缩比是"comparable reconstruction accuracy"下的，但 reconstruction error 到 policy 性能的映射没给。高保真不等于高 policy 性能，这条链路论文留白了。这是 FAST 后续工作可以补的地方。

## 一句话收束

FAST 用一个几乎零参数的 DCT 加 BPE 编解码器，把高频机器人动作压成与频率无关的短 token 序列，让自回归 VLA 第一次能在灵巧高频任务上训起来，并在性能和算力效率上同时追平 diffusion VLA。它最有力的论证在跑分之外,是那个"压缩比与频率无关、稳定 30 token/秒/臂"——这把"好的动作 tokenizer 该长什么样"这个问题,给了一个非常清晰的方向标。
