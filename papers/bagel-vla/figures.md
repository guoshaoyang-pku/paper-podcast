# BagelVLA — 论文重要图

> 配套 `card.json` / `card.md`。论文共 13 张 Figure（Figure 12/13 是 prompt template，略）。每张图给出：所在页、原文 caption（精简）、我们写的"这张图到底在讲什么"。
> 页图存在 `extracted/bagel-vla/pages/pXX.png`，可对照查看。

## Figure 1 — 框架总览（page 2）

![Figure 1](../../extracted/bagel-vla/pages/p02.png)

**原文 caption**：Overview of our framework. We present BagelVLA, a unified model that integrates linguistic planning, visual forecasting, and action generation within a single framework. We construct a massive hybrid dataset combining general multimodal data with large-scale robotic datasets. Robotic datasets with sub-tasks and keyframes are annotated to transfer the foundation model's general reasoning and visual generation abilities to embodied settings.

**这张图讲什么**：门面图。左侧是混合数据源（通用多模态 QA + 大规模机器人轨迹，机器人数据带子任务和关键帧标注），中间是 BagelVLA 统一模型，右侧是交错规划输出流（文字子任务→关键帧图像→动作）。核心信息：**语言/视觉/动作从一个统一底座里交错生成（而非各自独立的模块）**，且能用通用数据保底座能力。

## Figure 2 — BagelVLA MoT 架构（page 5）★ 最重要

![Figure 2](../../extracted/bagel-vla/pages/p05.png)

**原文 caption**：Illustration of the BagelVLA framework. BagelVLA utilizes a Mixture-of-Transformers (MoT) architecture, comprising three independent transformers specialized for linguistic, visual, and action modalities. To tackle long-horizon tasks and semantic generalization, we formulate language-conditioned action learning as a long-sequence interleaved planning problem.

**这张图讲什么**：核心架构图。三个独立 transformer（LLM expert / Generation expert / Action expert）经 MoT 共享自注意力。序列里交错排了观测、全局指令、子任务文字、关键帧 noisy latent、动作 noisy latent。展示 dual flow-matching 怎么在一条序列里同时训两个 FM 任务。配合 Figure 3 的三种 scheme 看。

## Figure 3 — 三种 Dual Denoise 方案（page 6）★ 核心机制

![Figure 3](../../extracted/bagel-vla/pages/p06.png)

**原文 caption**：Illustration of different types of dual denoising schemes. (a) Complete Denoise: Image prediction and action generation are performed separately, requiring a total of N1 + N2 denoising steps. (b) Joint Denoise: Image prediction and action generation are performed simultaneously, denoising together for N steps. (c) Single-Step Denoise: Action generation is conditioned directly on the context from the first denoising step of the image prediction.

**这张图讲什么**：全文最该看的机制图。三种 image-action FM 交互：(a) Complete 先跑完 N1 步图像去噪再跑 N2 步动作，延迟 6.04s；(b) Joint 两路同步 N 步去噪，延迟 2.90s；(c) Single-step action 只看图像第 1 步的 KV，延迟 1.23s。Table 4 显示 Single-step 不仅最快还最准（3.345 vs Complete 2.480），因为 Complete/Joint 在 OOD 中间态会崩。**RFG 是 Single-step 的变体（初始噪声用当前帧）**。

## Figure 4 — 交错规划真机可视化（page 9）★ 关键

![Figure 4](../../extracted/bagel-vla/pages/p09.png)

**原文 caption**：Visualization of interleaved planning results on real-world robotic tasks. Given a global instruction and the current observation, BagelVLA leverages the context to identify the immediate subtask, predicts a goal image for that subtask, and subsequently generates actions.

**这张图讲什么**：三个真机任务的交错规划输出：叠积木按指定顺序、算术式摆积木、Agibot 数据集任务。每个任务展示：全局指令→当前观测→生成的子任务文字→预测的关键帧图像→执行动作。证明模型真的在"想一步画一步动一步"，而不是黑盒出动作。

## Figure 5 — RFG vs Naive 单步去噪（page 11）★ 关键

![Figure 5](../../extracted/bagel-vla/pages/p11.png)

**原文 caption**：Predicted images using different denoising steps. The figure displays the generation results for the naive single-step denoise (Eq. 2) and RFG (Eq. 3) across varying denoising steps in real-world basic tasks and Robotwin randomized (unseen) scenarios. RFG demonstrates the capability to preserve backgrounds and achieve high-quality generation with very few steps.

**这张图讲什么**：RFG（N(v_t,I) 初始噪声）vs Naive（N(0,I) 纯噪声）在不同去噪步数下的关键帧生成质量。RFG 在 10 步就能生成高质量未来帧，Naive 几乎是噪声。证明**把当前帧注入初始噪声让模型聚焦动态区域而非重建静态背景**——这是 RFG 能把延迟压到 1.23s 还不掉点的原因。

## Figure 6 — 真机 Basic Task 消融（page 11）★ 关键

![Figure 6](../../extracted/bagel-vla/pages/p11.png)

**原文 caption**：Ablation on Real-World Basic Tasks.

**这张图讲什么**：真机 9 个 basic task 上的消融：BagelVLA（full） vs w/o pretrain vs RFG vs Naive。预训练在 pick&place OOD 和中等程任务（sweep rubbish/pour fries/stack cubes）上提升明显，说明语言规划+视觉动态预训练能迁移到 action。RFG 在多任务上优于 Naive。

## Figure 7 — Attention Mask（page 22）

![Figure 7](../../extracted/bagel-vla/pages/p22.png)

**原文 caption**：Attention Mask used for Different Conditioning Schemes. (a) Complete Denoise: Image prediction and action generation are performed separately... (b) Joint Denoise... (c) Single-Step Denoise: Action generation is conditioned directly on the context from the first denoising step of the image prediction.

**这张图讲什么**：Figure 3 的 attention mask 实现细节。展示如何用统一多任务 mask 在一条序列里同时算多个 FM loss，防止模态间信息泄漏，并处理训练/推理去噪步数不一致问题。是 Figure 3 的工程落地。

## Figure 8 — Basic Task 真机演示（page 24）

![Figure 8](../../extracted/bagel-vla/pages/p24.png)

**原文 caption**：Demos videos of BagelVLA on Basic Tasks.

**这张图讲什么**：9 类 basic task 的真机执行视频帧（pick&place/water flower/stack cubes/put flowers in vase/stack bowls/pour fries/sweep rubbish/press button/drawer）。是 Table 2 真机结果的视觉佐证。

## Figure 9 — Long-Horizon 真机演示（page 24）

![Figure 9](../../extracted/bagel-vla/pages/p24.png)

**原文 caption**：Demos videos of BagelVLA on Long-Horizon Planning Tasks.

**这张图讲什么**：两类长程任务的真机执行：Stack Cubes in Requested Order（按指定顺序叠积木，1-3 层）和 Calculate and Place Symbol Blocks（算术式摆积木，需 CoT 推理）。是 Table 3 长程结果的视觉佐证，展示模型在多阶段任务上的规划能力。

## Figure 10 — 更多交错规划可视化（page 25）

![Figure 10](../../extracted/bagel-vla/pages/p25.png)

**原文 caption**：Visualizations of interleaved planning results on diverse robotic tasks. Given a global instruction and the current observation...

**这张图讲什么**：Figure 4 的扩展版，更多任务的交错规划输出。每个都展示"指令→子任务文字→关键帧→动作"的完整链路。

## Figure 11 — RFG vs Naive 补充对比（page 26）

![Figure 11](../../extracted/bagel-vla/pages/p26.png)

**原文 caption**：Predicted images using different denoising steps. The figure displays the generation results for the naive single-step denoise (Eq. 2) and RFG (Eq. 3)...

**这张图讲什么**：Figure 5 的补充，更多场景下 RFG vs Naive 的关键帧生成对比。进一步证明 RFG 在少步去噪下的优势。

---

## 用法

- 想看模型长什么样：Figure 2（MoT 架构）、Figure 3（三种 scheme）
- 想理解核心机制：Figure 3（Single-step 最快最准）、Figure 5（RFG 残差）
- 想看消融：Figure 6（预训练+RFG 贡献）
- 想看真机结果：Figure 4（交错规划可视化）、Figure 8/9（basic + long-horizon 演示）
- 想看 attention 实现：Figure 7（mask 细节）
