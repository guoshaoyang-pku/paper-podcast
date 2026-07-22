# FAST — 论文重要图

> 配套 `card.json` / `card.md`。论文共 15 张 Figure,全部列出。每张图给出:所在页、原文 caption(精简)、我们写的"这张图到底在讲什么"。
> 页图存在 `extracted/fast/pages/pXX.png`,可对照查看。

## Figure 1 — Overview(page 1)

![Figure 1](../../extracted/fast/pages/p01.png)

**原文 caption**:We propose FAST, a simple yet effective approach for tokenization of robot action trajectories via time-series compression. FAST enables training of autoregressive VLAs that solve complex dexterous manipulation tasks and generalize broadly. We use it to train π0-FAST, a generalist robot policy that matches the performance of the state-of-the-art π0 diffusion VLA on dexterous and long-horizon manipulation tasks, while training 5x faster.

**这张图讲什么**:门面图。上方训练曲线:π0-FAST(蓝)在 table bussing + T-shirt folding 平均分上比 diffusion π0 早很多达到同等水平,5× 训练加速。下方灵巧任务可视化(折衣/收桌)说明 FAST 真让自回归 VLA 能做高频灵巧操作(作用远超"只压 token")。核心信息:一个看似工程的小改(tokenizer),把自回归 VLA 从"低频专属"推进到"匹敌 diffusion VLA 且更省算力"。

## Figure 2 — 为什么需要 FAST(page 2)

![Figure 2](../../extracted/fast/pages/p02.png)

**原文 caption**:Left: FAST tokenization enables training of autoregressive Transformers for dexterous robot control via simple next token prediction. Right: FAST outperforms popular binning tokenization schemes, e.g., used in OpenVLA, particularly for high-frequency robot data.

**这张图讲什么**:右图是核心诊断:横轴控制频率(2/5/10/20Hz),纵轴策略分数。OpenVLA 风格 binning 在 2Hz 还行,频率一高分数崩到接近 0;FAST 全频段都稳。左图示意 FAST 让自回归 VLA 用 next-token 就能学灵巧任务。这张图直接对应论文的根因论断:binning 在高频失败源于 tokenization 让学习信号消失,与模型无关。

## Figure 3 — toy 实验:采样率 vs 预测误差(page 3)

![Figure 3](../../extracted/fast/pages/p03.png)

**原文 caption**:Effect of sampling rate on prediction performance. We train a small autoregressive transformer model on a didactic interpolation task, in which the network must predict the black dashed curve given the four circles. We find that models trained with the binning tokenization approach used in prior VLAs produce increasingly poor predictions as we increase the sampling frequency. Our FAST tokenization approach, based on the discrete cosine transform (DCT), addresses the problem.

**这张图讲什么**:一个受控 toy 任务:预测过 4 个点的三次样条。横轴采样率(25→800 步)。上:MSE 曲线,binning(蓝)随采样率升高误差陡升,最后模型直接抄第一个动作(右下可视化);FAST 全程低误差。这是论文最干净的因果论证:数据分布不变、模型容量不变,只换 tokenization,结论完全不同。说明根因是 tokenization 而非数据或模型。

## Figure 4 — FAST Tokenization Pipeline(page 4)★ 最重要

![Figure 4](../../extracted/fast/pages/p04.png)

**原文 caption**:Overview of the FAST action tokenization pipeline. Given a normalized chunk of actions, we apply discrete cosine transform (DCT) to convert the signal to the frequency domain. We then quantize the DCT coefficients and use byte-pair encoding (BPE) to compress the flattened sequence of per-dimension DCT coefficients into the final action token sequence.

**这张图讲什么**:全文最该看的图。5 步流水线:①归一化 action chunk(1st/99th 分位映射到 [-1,1]);②对每维单独做 DCT,转到频域;③scale-and-round 量化(γ=10),得到稀疏系数矩阵(大量 0);④按低频优先 flatten(所有维的最低频系数先排,再次低频...),让 autoregressive 先预测整体形状;⑤BPE 无损压缩成定长词表的 token。这张图同时回答"怎么编码、为什么可逆、超参在哪"。对应 Algorithm 1。

## Figure 5 — Evaluation Environments(page 6)

![Figure 5](../../extracted/fast/pages/p06.png)

**原文 caption**:We test FAST across 7 evaluation environments: 6 real-robot tasks and 1 simulation environment. The tasks are designed to test VLA performance on highly dexterous tasks, like folding cloths from a laundry basket ("Laundry Folding"), and generalization, e.g., zero-shot table-top manipulation in unseen environments ("DROID").

**这张图讲什么**:评测套件总览:Libero(sim)、Table Bussing(20Hz UR5)、T-Shirt Folding(50Hz ARX 双臂)、Grocery Bagging(20Hz UR5)、Toast(50Hz Trossen 双臂)、Laundry Folding(50Hz ARX 双臂,最难 long-horizon)、Zero-shot DROID(15Hz Franka,首次完全未见环境评测)。覆盖频率 5~50Hz、单臂/双臂、sim/real,说明结论不是单一 setup 的偶然。

## Figure 6 — 四种 tokenizer 主结果(page 8)

![Figure 6](../../extracted/fast/pages/p08.png)

**原文 caption**:Comparison of policy performance using different tokenization approaches. We find that tokenization approaches that compress action targets (FAST, FSQ) lead to substantially more efficient training than the naïve binning tokenization used in prior VLAs. Overall, we find that FAST leads to more effective policy training than FSQ, particularly on dexterous real-robot tasks. Our universal tokenizer, FAST+, matches the performance of dataset-specific tokenizers.

**这张图讲什么**:4 任务 × 4 tokenizer(Naive/FSQ/FAST/FAST+)主结果柱状图。Naive 在 50Hz Shirt Fold 和 20Hz Bussing 上完全打不出分;FSQ 比 Naive 好但仍输 FAST;FAST 在所有任务最强或并列;FAST+ universal tokenizer 几乎追平数据集专属 FAST。两个关键结论:①压缩是关键(不止 FAST 特殊);②FAST 比 FSQ 这种学习式压缩更稳更简单。

## Figure 7 — DROID zero-shot 三校园(page 8)

![Figure 7](../../extracted/fast/pages/p08.png)

**原文 caption**:Evaluation environments of FAST policy trained on DROID. We find that the same policy checkpoint generalizes robustly, and performs various simple table-top tasks zero-shot across three university campuses.

**这张图讲什么**:Berkeley/Stanford/U.Washington 三地部署同一 DROID policy 的照片。关键定性结论:这是 DROID 数据集第一次被训出"完全未见环境零样本"policy(原 DROID paper 和 OpenVLA 都只做 co-training/fine-tuning 评测)。policy 能做 pick-and-place、开关抽屉、开水龙头;失败 trial 也表现合理(如接近微波炉门把手)。

## Figure 8 — FAST+ 跨本体压缩比(page 8)

![Figure 8](../../extracted/fast/pages/p08.png)

**原文 caption**:Universal tokenizer. We test the compression rate achieved by our FAST+ tokenizer vs. naïve tokenization across diverse robot datasets, unseen during tokenizer training. We find that FAST is effective across a wide range of robot morphologies, action spaces and control frequencies.

**这张图讲什么**:横轴数据集(单臂/dex/humanoid/nav 共 ~15 个,均未参与 FAST+ 训练),纵轴 naive/FAST 压缩比(对数)。FAST+ 在所有形态上都至少 2× 压缩,部分到 10×。证明 FAST+ 不是只在训练分布内有效,可作为通用黑盒 tokenizer。对应 Table III 的完整数据集列表。

## Figure 9 — FAST vs diffusion π0(单任务)(page 9)

![Figure 9](../../extracted/fast/pages/p09.png)

**原文 caption**:Comparison of diffusion π0 to our π0 model with FAST decoding on single-task training. On small datasets (Libero, T-Shirt Folding), both perform comparably. On large datasets (Table Bussing), FAST converges faster. In DROID, we find that FAST follows language instructions better.

**这张图讲什么**:4 任务柱状图。小数据集(Libero/T-Shirt)<50h 两者持平;大数据集(Table Bussing)FAST 用 3× 更少步数达同等水平;DROID 上 FAST 分数更高——作者归因 diffusion π0 经常忽略语言指令(会直接抓物体),FAST 自回归 VLA 更听语言。这是 FAST 主卖点之一:不仅快,还在某些维度更好。

## Figure 10 — Laundry Folding rollout(page 10)

![Figure 10](../../extracted/fast/pages/p10.png)

**原文 caption**:Rollout of π0-FAST on the laundry folding task. FAST tokenization enables autoregressive VLAs to perform complex, long-horizon, and dexterous tasks that were impossible with previous tokenization schemes.

**这张图讲什么**:π0-FAST 在 laundry folding(从篮取衣→摊平→折→叠放)的 10 步 rollout。这是论文最难任务,以前 binning tokenizer 完全训不出。FAST 让纯自回归 VLA(无 diffusion)也能做多阶段灵巧操作。是"FAST 解锁新能力"的视觉证据。

## Figure 11 — 通用 VLA:π0-FAST vs diffusion π0(page 10)

![Figure 11](../../extracted/fast/pages/p10.png)

**原文 caption**:Comparison of π0-FAST and diffusion π0 generalist policies. π0-FAST matches the performance of diffusion π0 while requiring significantly less compute for training.

**这张图讲什么**:5 个 generalist 任务平均柱状图。π0-FAST(自回归+FAST)在 T-Shirt/Bussing/Grocery/Toast/Laundry 平均上匹敌 diffusion π0,但训练用 5× 更少 GPU 小时。这是论文最终论点:FAST 让自回归 VLA 在"性能"和"算力效率"两个维度同时追平甚至超过 diffusion VLA。

## Figure 12 — 压缩↔重建权衡曲线(page 16)

![Figure 12](../../extracted/fast/pages/p16.png)

**原文 caption**:Comparison of compression-reconstruction tradeoff on six training datasets. FAST performs well across a wide range of scales. In particular, although it is less efficient than VQ-based tokenizers at low fidelities, it exhibits much better scaling to higher reconstruction fidelity, making FAST much more applicable to fine-grained control problems.

**这张图讲什么**:6 数据集上横轴 token 数、纵轴重建误差的权衡曲线。FAST 在低 fidelity 区不如 FSQ 极致压缩,但向高 fidelity 区 scaling 好得多——这对精细控制(高频灵巧任务)是决定性的。说明 FAST 的优势不在"压得最狠",而在"高保真下仍能压"。

## Figure 13 — 5 个评测任务的初始配置(page 17)

![Figure 13](../../extracted/fast/pages/p17.png)

**原文 caption**:Sampled initial configurations of evaluation tasks: Table Bussing / T-Shirt Folding / Grocery Bagging / Toast out of Toaster / Laundry Folding.

**这张图讲什么**:5 个真实评测任务的初始场景照片。给读者一个"评测难度"的直观感受:Bussing 12 物体混 trash/plates、T-Shirt 5 件不同色码、Grocery 7 物品入袋、Toast 双臂取吐司、Laundry 从篮取衣折放。是结果可信度的环境证据。

## Figure 14 — DROID 定量评测场景(page 18)

![Figure 14](../../extracted/fast/pages/p18.png)

**原文 caption**:Setups used for quantitative DROID evaluation. 16 tasks, 44 trials total per policy.

**这张图讲什么**:DROID 定量评测的 16 个任务/44 trial 的初始场景照片。配合 Table II 的任务清单(放勺入碗、关抽屉、擦白板等)。定义了 DROID zero-shot 评测的口径——完全未见环境、视角、物体,只靠自然语言 prompt。

## Figure 15 — compute-matched 对比(page 18)

![Figure 15](../../extracted/fast/pages/p18.png)

**原文 caption**:Comparison of π0-FAST and compute-matched diffusion π0 generalist policies. π0-FAST clearly outperforms the diffusion VLA when trained with the same amount of training compute, due to its faster convergence.

**这张图讲什么**:5 任务 compute-matched 柱状图(两边用同样 GPU 小时)。π0-FAST 在 T-Shirt/Bussing/Grocery/Toast 平均上明显超过 compute-matched diffusion π0。这是 Figure 11 的补充论证:其含义是"同算力下 FAST 更强"(而非"FAST 用更少算力达到同等")——根因是收敛更快。

---

## 用法

- 想看 tokenizer 怎么工作:Figure 4(流水线)、Figure 3(toy 因果论证)
- 想看根因诊断:Figure 2(频率 vs 分数)、Figure 3(采样率 vs 误差)
- 想看主结果:Figure 6(tokenizer 对比)、Figure 9(单任务 vs diffusion)、Figure 11(generalist vs diffusion)、Figure 15(compute-matched)
- 想看通用性:Figure 7(DROID 三校园)、Figure 8(FAST+ 跨本体压缩)
- 想看权衡:Figure 12(压缩↔重建)
- 想看任务难度:Figure 5(评测套件)、Figure 13(任务初始配置)、Figure 14(DROID 评测场景)
