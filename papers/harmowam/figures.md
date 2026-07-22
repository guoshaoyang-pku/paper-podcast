# HarmoWAM — 论文重要图

> 配套 `card.json` / `card.md`。论文共 11 张 Figure,全部列出。每张图给出:所在页、原文 caption(精简)、我们写的"这张图到底在讲什么"。
> 页图存在 `extracted/harmowam/pages/pXX.png`,可对照查看。

## Figure 1 — Overview + 四类 OOD 对比(page 2)

![Figure 1](../../extracted/harmowam/pages/p02.png)

**原文 caption**:We propose HarmoWAM, an end-to-end WAM that jointly achieves generalizable transit and precise manipulation through a world model that provides physical dynamics priors and adaptively coordinates a predictive action expert and a reactive action expert. HarmoWAM achieves SOTA performance in ID settings and exhibits a substantial advantage in OOD scenarios.

**这张图讲什么**:门面图。上半把核心机制画清:世界模型出未来视频 + latent,Process-Adaptive Gating 用 Yes/No 决定走 Predictive expert(隐含 latent→结构化动作,精准)还是 Reactive expert(显式未来帧→反应式动作,泛化);下半对比 Cosmos-Policy / Wan+Anypos 在 ID/OOD 三维(Position/Background/Objects)上的表现,HarmoWAM 在 OOD 全绿。一句话讲清论文 thesis:不是又一个大模型,是把两条对立路线用门控统一。

## Figure 2 — Framework(page 5)★ 最重要

![Figure 2](../../extracted/harmowam/pages/p05.png)

**原文 caption**:HarmoWAM adopts an adaptive framework that tightly integrates a generative world model with two complementary action experts. The world model provides both explicit future predictions and implicit latent representations. Conditioned on current latent features, the predictive expert generates structured actions for precise manipulation, while the reactive expert leverages future predicted frames and their latent features to perform reactive inference for generalizable transit. A Process-Adaptive Gating mechanism predicts the task stage from observations and dynamically routes control between the two action experts.

**这张图讲什么**:全文最该看。左路:输入(I_t, l_t)→ Wan Encoder/T5 → 30 DiT block 世界模型(diffusion ×T 步)→ 13 帧未来视频 V + 隐含 latent F_V。中间世界模型把 F_V_t 喂给两个 expert。Predictive(上):28 DiT blocks,self-attn + cross-attn(条件 F_img_t 来自 SigLIP、F_text、F_V_t),diffusion 出 a_{t+1..t+3}。Reactive(下):DINOv2 ×12 ViT + Patch Embedding + Pooling,把 patch 特征和 latent concat,经 Conv/FFN 出 a。右侧 Process-Adaptive Gating:从 SigLIP 取 F_img → MLP → s_t,if s_t>0.5 走 predictive 否则 reactive。一张图同时回答:世界模型怎么共享、两 expert 怎么取不同条件、门控怎么路由。

## Figure 3 — Attention 可视化 + 执行过程(page 7)

![Figure 3](../../extracted/harmowam/pages/p07.png)

**原文 caption**:The upper part presents attention map visualizations from the last-layer features of the reactive and predictive experts, while the lower part illustrates the robot execution process.

**这张图讲什么**:上半是两个 expert 最后一层 attention map 的可视化。关键观察:Predictive expert 注意力集中在"被操作物体"(说明它在做接触级精细规划),Reactive expert 注意力散在"夹爪和任务相关周围环境"(说明它在做空间导航/接近)。这给双 expert 分工提供了机理证据——不是凭空分,是它们天然 attend 到不同区域。下半是执行过程帧序。

## Figure 4 — OOD 泛化实验设置(page 9)

![Figure 4](../../extracted/harmowam/pages/p09.png)

**原文 caption**:Generalization experiments. Red boxes highlight unseen objects, background variations, and manipulated object positions, while blue boxes indicate original training configurations.

**这张图讲什么**:三个 OOD 维度的视觉定义。Unseen Background(加 5-8 个未见干扰物 + 光照变化)、Unseen Position(目标放在训练覆盖区域之外的空间不相交区域)、Unseen Objects(胡萝卜换辣椒、可乐换红牛、花换不同颜色数量)。蓝框=训练配置,红框=测试配置。定义了 Table 3 所有 OOD 结果的评测口径——三档 OOD 是正交维度,不是简单"换背景"。

## Figure 5 — Ablation Study(page 9)

![Figure 5](../../extracted/harmowam/pages/p09.png)

**原文 caption**:We investigate (a) HarmoWAM Structure, (b) Efficacy of Process-Adaptive Gating, and (c) Impact of world model latent features on both action experts. The "-vid" suffix indicates that video latent features are excluded from the action expert's conditioning.

**这张图讲什么**:三组消融。(a) 去掉 Reactive expert → position OOD 掉到 14%(证明 reactive 负责探索);去掉 Predictive → position/objects OOD 掉到 56%/60%(证明 predictive 负责精度)。两 expert 互补,缺一不可。(b) 门控 vs Averaging vs Keyframe-Averaging:Averaging 在 position OOD 掉 46%,门控只掉很少——证明阶段感知硬路由 > 数值平均。(c) 去掉世界模型 latent 特征:reactive-vid ID 65%/OOD 54%,predictive-vid ID 从 95% 掉到 62%——证明世界模型 latent 是两 expert 的关键条件,不是装饰。三组消融各自论证一个设计选择。

## Figure 6 — 硬件 setup(page 15)

![Figure 6](../../extracted/harmowam/pages/p15.png)

**原文 caption**:Real-World robot setup and experimental assets.

**这张图讲什么**:硬件照片。单臂:Franka FR3 + 3D 打印 UMI gripper + 3 个 Intel RealSense(前视 D435 45°、俯视 D455、腕部 D435)。双臂:两台并行 FR3 + 同 UMI gripper + 1 全局 + 2 腕部。SpaceMouse 遥操。说明实验在真实 Franka 上做,不是仿真;UMI gripper 是为接触级精度定制的 3D 打印件。

## Figure 7 — 两条旧路线的失败案例(page 18)

![Figure 7](../../extracted/harmowam/pages/p18.png)

**原文 caption**:Representative failure cases of the two World Action Models paradigms under OOD scenarios. The first three rows show typical failures of the Joint Modeling baseline and the last three rows show common failures of the Imagine-then-Execute baseline.

**这张图讲什么**:支撑 Table 1 motivation 的失败可视化。前 3 行 Joint Modeling 失败:OOD 下背景/位置变化让预测轨迹偏离目标区,末端到不了物体上方(transit 失败,根本进不了 interaction)。后 3 行 Imagine-then-Execute 失败:transit 能到目标附近,但 IDM 反推的动作在 interaction 偏——抓取点偏移、夹爪提前开合。两组失败各自精确对应论文说的 trade-off:一边死于探索、一边死于精度。这是 HarmoWAM 要统一的对象。

## Figure 8 — 六任务完整执行序列(page 19)

![Figure 8](../../extracted/harmowam/pages/p19.png)

**原文 caption**:Visualization of complete execution sequences on six real-world manipulation tasks.

**这张图讲什么**:HarmoWAM 在 6 个任务上的完整执行帧序:Pick Fruit / Stack Cans / Pour Coke / Write 'Yes' / Put Flowers in Vase / Put Items to Bag and Zip。强调运动轨迹平滑连续,尤其 Stack Cans 的对齐、Pour Coke 的持续角度控制、Flower insertion 的紧公差对齐。是 Table 2 ID 结果的可视化证据。

## Figure 9 — 三档 OOD 完整示例(page 20)

![Figure 9](../../extracted/harmowam/pages/p20.png)

**原文 caption**:Visualization of the complete examples under three OOD settings: unseen background, unseen position, and unseen objects.

**这张图讲什么**:HarmoWAM 在三档 OOD 下的完整执行示例,对应 Table 3。Unseen Background(干扰物+光照)、Unseen Position(空间不相交区域)、Unseen Objects(换物体)。是 OOD 82% 平均成功率的可视化证据,展示模型能在未见背景/位置/物体上完成完整任务。

## Figure 10 — 去噪步数对比(page 23)

![Figure 10](../../extracted/harmowam/pages/p23.png)

**原文 caption**:Visual comparison of generated videos under different denoising steps.

**这张图讲什么**:世界模型 3/5/10/50 步去噪生成的未来视频对比。3 步明显模糊缺细节(对应 success 80%);5 步保留关键动作阶段和物体位置清晰度(85%);10/50 步视觉质量略升但 success 仅 85%/87%,推理频率反而降。支撑 Table 9 结论:5 步是 quality-speed 甜点,更多步边际递减。

## Figure 11 — HarmoWAM 失败案例(page 24)

![Figure 11](../../extracted/harmowam/pages/p24.png)

**原文 caption**:Failure case visualization of HarmoWAM. We visualize representative failures in real-world Franka experiments, with red boxes highlighting the key error regions.

**这张图讲什么**:HarmoWAM 自己的三类失败:(1)Tilted stacking——堆叠时前后位移估计不准(第三人称视角对该方向小偏移不敏感),放上去后倾斜滑落;(2)Insertion misalignment——插花时抓取位置变化改变花茎外露长度,插入姿态对不齐;(3)Zipper grasp slip——拉链夹持摩擦力不足,持续拉拽中脱出。三类失败都不是"理解错任务"而是"精度/硬件极限"——诚实展示方法边界。

---

## 用法

- 想看模型长什么样:Figure 2(框架)、Figure 1(overview)
- 想看结果:Figure 8(ID 执行)、Figure 9(OOD 执行)
- 想理解关键设计:Figure 3(attention 分工证据)、Figure 5(三组消融)
- 想理解 motivation:Figure 7(两路线失败)、Table 1(数值)
- 想理解工程权衡:Figure 10(去噪步数)、Figure 11(失败边界)
- 想看硬件:Figure 6
