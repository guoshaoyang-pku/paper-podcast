# OpenVLA — 论文重要图

> 配套 `card.json` / `card.md`。论文共 11 张 Figure,全部列出。每张图给出:所在页、原文 caption(精简)、我们写的"这张图到底在讲什么"。
> 页图存在 `extracted/openvla/pages/pXX.png`,可对照查看。

## Figure 1 — Overview(page 1)

![Figure 1](../../extracted/openvla/pages/p01.png)

**原文 caption**:We present OpenVLA, a 7B-parameter open-source vision-language-action model (VLA), trained on 970k robot episodes from the Open X-Embodiment dataset. OpenVLA sets a new state of the art for generalist robot manipulation policies.

**这张图讲什么**:门面图。970k robot episodes → ViT+Llama 2 7B base VLM → OpenVLA。支持多机器人 out-of-the-box + 高效 fine-tune 到新域。强调 fully open-source(Data/Weights/Code)。下方示例:用户 "Wipe the table" → OpenVLA 出 [Δx, Δy, Δz, Δθ, Δ, Grip] 7 维动作。核心信息:7B 开源 VLA,7× 少参数超 55B RT-2-X。

## Figure 2 — Architecture(page 4)★ 最重要

![Figure 2](../../extracted/openvla/pages/p04.png)

**原文 caption**:OpenVLA model architecture. Given an image observation and a language instruction, the model predicts 7-dimensional robot control actions. The architecture consists of three key components: (1) a vision encoder that concatenates DinoV2 and SigLIP features, (2) a projector that maps visual features to the language embedding space, and (3) the LLM backbone, a Llama 2 7B-parameter large language model.

**这张图讲什么**:全文最该看。三组件:(1)视觉编码器=DINOv2(patch)+SigLIP(patch)channel-wise concat;(2)MLP projector 映射到 Llama 词嵌入空间;(3)Llama 2 7B LLM。输入图像+语言 "Put eggplant in bowl" → 模板 "What should the robot do to {task}? A:" → LLM 自回归出 7 个 action token → Action De-Tokenizer → 7D 机器人动作(Δx,Δy,Δz,Δθ 等)。一张图说清:无 DiT,无 action chunk,action 直接走 LLM token 通道。这是"最简单可扩展 VLA"的设计图。

## Figure 3 — BridgeData V2 WidowX 评测(page 7)

![Figure 3](../../extracted/openvla/pages/p07.png)

**原文 caption**:BridgeData V2 WidowX robot evaluation tasks and results. We evaluate OpenVLA and prior state-of-the-art generalist robot policies on a comprehensive suite of tasks covering several axes of generalization, as well as tasks that specifically assess language conditioning ability.

**这张图讲什么**:BridgeData V2 WidowX 上 17 任务评测结果(170 rollouts)。5 类泛化:visual(未见背景/干扰/外观)、motion(未见位置/朝向)、physical(未见大小/形状)、semantic(未见物体/指令/Internet 概念)、language grounding(多物体场景操纵指定物体)。OpenVLA 在除 semantic 外所有类别超 RT-2-X,RT-1-X/Octo 几乎全崩(0-2 success/10)。核心:7B OpenVLA 在多数类别超 55B RT-2-X,只因 RT-2-X Internet 预训练规模更大且 co-fine-tune 保先验故 semantic 类仍领先。

## Figure 4 — Google Robot 评测(page 8)

![Figure 4](../../extracted/openvla/pages/p08.png)

**原文 caption**:Google robot evaluation results. We evaluate generalist robot policies on in-distribution and out-of-distribution (OOD) tasks on the mobile manipulator used in RT-1 and RT-2 evaluations.

**这张图讲什么**:Google Robot(移动平台)上 in-distribution + OOD 评测(60 rollouts)。OpenVLA 与 RT-2-X 持平(78.3 vs 72.0 OOD,85.0 vs 88.0 ID),都显著超 RT-1-X(26.7/33.3)和 Octo(32.0/44.0)。Move Coke Can to Taylor Swift 任务展示 semantic 泛化(Taylor Swift 是 Internet 概念不在机器人数据)。核心:7B OpenVLA 在 Google Robot 上匹配 55B RT-2-X。

## Figure 5 — 新机器人 fine-tune 适配(page 9)

![Figure 5](../../extracted/openvla/pages/p09.png)

**原文 caption**:Adapting to new robot setups. We evaluate the state-of-the-art Diffusion Policy trained from scratch on seven Franka Emika Panda tasks, as well as generalist robot policies Octo and OpenVLA fine-tuned on the same data.

**这张图讲什么**:Franka-Tabletop + Franka-DROID 上 7 任务 fine-tune(10-150 demonstrations)。3 类:narrow single-instruction、diverse multi-instruction、visual robustness。Diffusion Policy 在窄单指令任务强(动作平滑精确),OpenVLA 在多样多指令任务强(语言 grounding)。OpenVLA 是唯一所有任务都 ≥50% 的方法,且 aggregate 最高。核心:OpenVLA 是 downstream fine-tune 的强默认初始化,尤其涉及多样语言指令;窄高灵巧任务 Diffusion Policy 仍优。

## Figure 6 — 推理速度 + 量化(page 10)

![Figure 6](../../extracted/openvla/pages/p10.png)

**原文 caption**:OpenVLA inference speed for various GPUs. Both bfloat16 and int4 quantization achieve high throughput, especially on GPUs with Ada Lovelace architecture (RTX 4090, H100).

**这张图讲什么**:不同 GPU 上 bfloat16 vs int4 的推理频率。RTX 4090/H100(Ada Lovelace)上最高吞吐。int4 量化在多数 GPU 上因减少显存传输而比 int8 更快。核心:OpenVLA 可在消费级 GPU(RTX 4090/A5000)上实时推理,无需服务器卡;4bit 量化性能不掉(71.9% vs bfloat16 71.3%)且显存 7GB(可跑在更多消费卡上)。这是 VLA 民主化的关键证据。

## Figure 7 — BridgeData V2 评测任务图例(page 22)

![Figure 7](../../extracted/openvla/pages/p22.png)

**原文 caption**:BridgeData V2 evaluation task illustrations.

**这张图讲什么**:BridgeData V2 WidowX 17 个评测任务的视觉图例,对应 Figure 3 的数值。展示每个任务的具体场景(放茄子进锅/叠杯/翻锅/举电池/移头骨等)。强调评测比 prior work 更难——末端执行器初始化在固定位置(非直接在目标上方),要水平 reach。这是 RT-1-X/Octo 比 prior work 报告低的原因。

## Figure 8 — 原 BridgeData V2 sink 任务(page 25)

![Figure 8](../../extracted/openvla/pages/p25.png)

**原文 caption**:Original BridgeData V2 sink environment tasks. Images from sample demonstrations in the sink environment from the original BridgeData V2 dataset reveal that all demonstrations in this environment were initialized such that the robot's end-effector was positioned immediately above the target object.

**这张图讲什么**:原 BridgeData V2 sink 环境的 7 个训练任务示例(Flip Pot/Put Carrot on Plate 等)。关键观察:训练数据里末端执行器都初始化在目标正上方,而 OpenVLA 评测初始化在固定位置要水平 reach。这解释了为何 prior work(如 Octo)在 OpenVLA 评测 suite 上比原报告低——评测更难,要求更强泛化。是 Figure 3/7 评测口径的对照。

## Figure 9 — Google Robot 评测任务图例(page 27)

![Figure 9](../../extracted/openvla/pages/p27.png)

**原文 caption**:Google robot evaluation task illustrations.

**这张图讲什么**:Google Robot(移动平台)评测任务的视觉图例,对应 Figure 4 数值。展示 in-distribution 和 OOD 任务的具体场景(移可乐到 Taylor Swift 海报/Pick Coke Can 等)。Taylor Swift 任务是 semantic 泛化测试——Internet 概念不在机器人数据。是 Figure 4 评测口径的图例。

## Figure 10 — Franka fine-tune 任务图例(page 29)

![Figure 10](../../extracted/openvla/pages/p29.png)

**原文 caption**:Franka fine-tuning task illustrations.

**这张图讲什么**:Franka-Tabletop + Franka-DROID 上 7 个 fine-tune 任务的视觉图例,对应 Figure 5 数值。展示具体任务场景(put carrot in bowl/pour corn/multi-instruction 等)。强调这些是 OpenVLA fine-tune 适配新本体的目标任务,Franka 7-DoF 单臂。是 Figure 5 评测口径的图例。

## Figure 11 — Franka 评测环境(page 31)

![Figure 11](../../extracted/openvla/pages/p31.png)

**原文 caption**:Franka evaluation environments.

**这张图讲什么**:Franka-Tabletop(固定桌面 5Hz)和 Franka-DROID(可移动站立桌 15Hz)两个 fine-tune 评测环境的硬件照片。说明 OpenVLA 能适配不同控制频率和本体配置。是 Figure 5 硬件对照。

---

## 用法

- 想看模型长什么样:Figure 2(架构)
- 想看结果:Figure 3(BridgeData V2)、Figure 4(Google Robot)、Figure 5(Franka fine-tune)
- 想理解工程民主化:Figure 6(推理速度+量化)
- 想看评测口径:Figure 7/9/10(任务图例)、Figure 8(训练 vs 评测差异)
- 想看硬件:Figure 11
