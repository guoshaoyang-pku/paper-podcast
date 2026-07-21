# VLA-World: Learning Vision-Language-Action World Models for Autonomous Driving

2026-04-10，Shanghai Jiao Tong University (MoE Key Lab of AI, AI Institute) 的 Guoqing Wang 等人发布的论文，标题是 VLA-World: Learning Vision-Language-Action World Models for Autonomous Driving。

## 开场：这篇论文真正要解决什么

这篇论文要解决自动驾驶里一个非常结构性的矛盾：**VLA 直接从观测出动作，缺前瞻；世界模型能生成未来但只能"模拟"不能"推理评估"想象出来的未来。两边各有半截能力，合不到一起。**

为什么合不到一起？VLA 方法像 OmniDrive、ELM 这些，直接把观测映射到轨迹，缺对其它动态 agent 的时空建模——它不知道周围的车人会怎么动。世界模型像 DriveDreamer、OccWorld 这些，能生成未来视频，但它只是"模拟"——它能想象未来长什么样，却不会评估这个未来安不安全、该不该那样走。

FSDrive 算是第一个试着把两边接起来的：它用 Qwen2-VL 生成未来帧作为思维链的中间步。但它有两个问题：只生成前视，左转右转场景没法想象；而且它生成完未来帧就直接回归 waypoint，不对那个未来做反思评估。

VLA-World 的命题是：能不能让模型**先想象未来，再对自己想象出来的未来做反思，然后修正轨迹**？就像人开车——直觉想象下一步，发现行人突然进车道，立刻反思"如果继续走会撞"，然后修正。它走通了这条路。

## 它的输入和输出到底是什么

你先在脑子里建一个五步 pipeline 的图。

输入有三路：**多视角观测**，nuScenes 的 6 个相机视角，加上自车的 CAN 信号（速度、加速度、yaw rate）；**导航指令**，就是 left、right、forward 这种高层目标。

输出是一个完整的五步结构化序列。第一步是**感知结果**，检测周围动态 agent、3D 位置、路肩距离、可行驶区域，是个结构化的场景描述。第二步是**短期预测**，下 0.5 秒的 waypoint 和行驶方向。第三步是**想象未来帧**，用 VQGAN 自回归生成下一帧的视觉 token 序列，可以指定任一视角。第四步是**反思**，对想象出来的未来帧做推理，识别重要 agent 和潜在风险。第五步是**最终轨迹**，修正后的 3 秒 horizon、0.5 秒间隔、6 个 BEV waypoint。

这五步走的是同一个 Qwen2-VL-2B 主干，纯自回归 next-token prediction。

### 输入到底怎么拼成一条序列

光说五步还不够具体，我把它拆成一条显式的拼接序列。

序列大致长这样：开头 `<bos_obs>` 包 6 个相机视角加 ego status，经 Qwen2-VL 编码，`<eos_obs>`。然后 `<bos_goal>` 包导航指令，`<eos_goal>`。接着 `<bos_perception>` 包感知文本——检测到的 agent、3D 位置、路肩距离，`<eos_perception>`。再 `<bos_prediction>` 包 0.5 秒 waypoint 加方向，`<eos_prediction>`。然后 `<bos_visual>` 包 VQGAN 生成的未来帧视觉 token 序列，`<eos_visual>`。接着 `<bos_think>` 包反思文本，`<eos_think>`。最后 `<bos_action>` 包高层动作，`<bos_answer>` 包 3 秒轨迹 waypoint 列表。

这里有三个坑，听别的统一模型论文你也会反复遇到，必须分清。

**第一，所有模态都是自回归 token 序列拼接，包括视觉未来帧。** 这点和 DreamZero、Motus 完全不同。DreamZero 的语言是 cross-attention 注入，视觉未来是连续 latent diffusion。Motus 走 MoT 共享自注意力。但 VLA-World 走纯 LLM 自回归范式——所有东西都是 token，包括视觉未来帧。视觉未来帧是用 VQGAN 离散化成 codebook token，然后像生成文字一样 next-token prediction 出来。这是它最根本的架构选择。

**第二，多视角在序列层拼接，6 个视角的 token 都在序列里，而且生成时可指定任一视角。** 这点是对 FSDrive 的关键扩展。FSDrive 只生成前视，左转右转场景没法想象。VLA-World 在 Stage1 预训练时显式约束多视角一致性，所以推理时可以按预测方向请求对应视角——左转请求左视，右转请求右视，想象更贴合驾驶意图。

**第三，训练和推理时这条序列结构是一致的，但三阶段侧重点不同。** Stage1 只激活视觉生成，序列只到 `<eos_visual>`；Stage2 SFT 全 pipeline 监督，六个标签都有；Stage3 RL 用 GRPO 对完整输出采样打分。推理时是完整 pipeline 一次自回归出全部六段。

记住这条序列和这三个坑，后面听别的 cascaded WAM 你会反复用到。

## 架构：Qwen2-VL-2B 加 VQGAN，纯自回归

主干是 Qwen2-VL-2B，一个 2B 参数的多模态 LLM。VLA-World 初始化时继承了 FSDrive 的对齐策略。视觉生成用的是 VQGAN——把图像离散化成 codebook token 序列，这样视觉生成就天然兼容 LLM 的自回归范式，不需要额外的 diffusion head。

为什么用这么小的 2B 模型？因为自动驾驶的轨迹规划是 BEV 2D waypoint，不是机器人那种高维关节控制，2B 的推理能力够用。而且 Stage1 预训练已经激活了视觉生成能力，底子打好了。

为什么用 VQGAN 离散 token 而不是连续 latent diffusion？因为离散 token 让视觉生成天然兼容 LLM 自回归，一套主干搞定所有模态。代价是生成质量受 codebook 限制——它的 FID 是 9.8，只比 diffusion 类的 GEM 10.5 略好，没有显著超越。而且离散化会损失高频细节，对交通灯颜色这种安全关键信息可能不利。这是个权衡。

## 核心机制：想象 + 反思闭环

这是 VLA-World 最关键的设计，单独讲。

它的因式分解是这样的：p(轨迹, 未来帧 | 观测, 目标) = p(轨迹 | 观测, 目标) · p(未来帧 | 观测, 短期轨迹)。纯 VLA 只关注左因子，直接出轨迹；纯世界模型只关注右因子，生成未来。VLA-World 把两者接起来：先用短期轨迹 τ̂ 引导生成未来帧 x̂，这是想象；再对 x̂ 做反思推理 f_ref，输出修正轨迹 τ̃。

关键洞察是：短期预测的未来帧天然编码了丰富的时空信息——既包括自车运动，也包括周围 agent 的行为。所以这个想象出来的未来帧，是反思的可靠依据。反思能修正直觉预测忽略的风险，比如行人突然进入车道。

消融实验很有说服力。去掉反思模块，L2 误差从 0.30 涨到 0.85，整整涨了 0.55 米。去掉感知模块涨 0.45，去掉生成模块涨 0.38。**反思的贡献最大**，比生成本身还大。这说明"对想象未来做推理"比"生成未来"更重要——这是 VLA-World 最有力的论断。

## 给你一个数值 sense：这套模型到底多大

光说 2B 没感觉，我把维度念细一点。

主干是 Qwen2-VL-2B，2B 参数的标准 VLM transformer。VQGAN tokenizer 把图像离散化成 codebook token 序列，每个 token 是 codebook 的一个 entry。生成图像分辨率是 128×192，和 FSDrive 同款。这个分辨率其实不高——实际驾驶摄像头都是 1080P 以上，128×192 能不能支持远距离行人的精细识别是个问题，论文没讨论。

每帧的 token 数论文没明示，但 128×192 图像用 VQGAN 典型是几百到上千个 token。6 个视角全部进序列，token 数量很大。

时间维度：推理步长 0.5 秒，轨迹 horizon 是 3 秒，0.5 秒间隔，共 6 个 BEV waypoint。nuScenes 是 2Hz 采样。注意这个 2Hz 帧率对快速动态场景——比如突然切入——响应可能滞后，这是它的一个局限。

训练规模：Stage1 预训练 30 个 epoch，AdamW 学习率 5e-4，per-device batch 16，8 张 80GB GPU，480k 图像-指令对。Stage2 SFT 12 个 epoch，学习率 1e-4，用 nuScenes-GR-20K 这个自建数据集。Stage3 GRPO 1 个 epoch，学习率 1e-6，global batch 16，每个 prompt 采样 8 个候选。三阶段都在 8 张 80GB GPU 上。

记住这套数字：2B Qwen2-VL、VQGAN 离散 token、128×192 生成、0.5s 步长、3s horizon 6 个 waypoint、三阶段训练。这是个中等规模的训练，比 DreamZero 的 14B、BagelVLA 的 16B 都小很多。

## 三阶段训练：生成知识 → 概念知识 → 推理知识

VLA-World 的训练不是一锅炖，是分三阶段层层叠加，对应三种知识。

**Stage1 视觉预训练**，480k 图像-指令，30 个 epoch。目标是冷启动——让模型学会理解驾驶世界，能生成未来帧。这阶段只激活视觉生成能力，序列只到 `<eos_visual>`。关键是它显式约束多视角一致性，这是对 FSDrive 只生前视的扩展。

**Stage2 监督微调**，nuScenes-GR-20K 这个自建数据集，12 个 epoch。用 imitation learning 建 perception→prediction→generation→think→action 的完整因果链。这阶段全 pipeline 监督，六个标签都有。这是建概念知识的阶段。

**Stage3 强化学习**，GRPO，20K，1 个 epoch，每个 prompt 采样 8 个候选。让模型在生成未来上交互探索，学会自我验证、丢弃不安全轨迹。这是探索推理知识的阶段。

消融有个重要结论：**SFT 比 RL 重要**。去掉 SFT 的 L2 是 0.85，去掉 RL 的 L2 是 0.71。RL 没有 SFT 冷启动就无法导航大搜索空间，SFT 建因果链是根基，RL 只是精修。这个结论对其它想用 GRPO 强化 VLA 的工作有参考价值——冷启动监督不能省。

## GRPO：value-free RL 加 5 个 rule-based reward

Stage3 用的 GRPO 是个值得讲的细节。

它不用 critic，区别于 PPO。机制是：对每个 prompt 采样 G=8 个候选 rollout，用 5 个 rule-based reward 打分，然后组内归一化算优势 A_i = (r_i - μ)/σ。这个组内归一化提供动态 baseline，减少内存开销。

5 个 reward 覆盖整条 pipeline：R_fmt 管格式（六个标签结构），R_pred 管短期预测加长短期一致性，R_vis 管视觉 token 长度和 codebook 有效性，R_act 用 F1 评动作，R_traj 管轨迹精度和运动学一致性。加权组合成 R_all。

为什么用 rule-based 而不是神经 reward model？为了避免 reward hacking，而且 rule-based 可解释、可控。KL 项防止偏离 SFT checkpoint。让模型学会 Self-Verification，丢弃幻觉和不安全轨迹。

## 关键结果：规划安全和生成质量双赢

我挑几个有对照的数字。

**轨迹规划**，nuScenes ST-P3 协议带 ego-state：VLA-World 平均 L2 误差 0.26 米，FSDrive 是 0.28，OmniDrive 是 0.33，EMMA 是 0.32。Collision rate VLA-World 0.08%，FSDrive 0.10%，OmniDrive 0.30%。UniAD 协议下 VLA-World L2 0.42，FSDrive 0.45，OccWorld 1.40。两个协议下 VLA-World 都是最好。

**未来帧生成**，FID：VLA-World 9.8，FSDrive 10.1，GEM 10.5，Doe-1 15.9。虽然 VLA-World 的主要目标不是生成，但它甚至比专门的 diffusion 生成模型还好一点点。

**动作预测 F1**：forward 95.88%，left 74.22%，right 75.06%。基座 Qwen2-VL-2B 在 nuScenes 上微调后（Qwen2-VL-2B†）forward 92.60%，left 61.78%。VLA-World 全面胜出，转向动作提升最大——这说明它的多视角 goal-conditioned 生成对转向场景特别有用。

**3 秒长程**：VLA-World 在 3s horizon 上优势最明显，因为传统 VLA 在 3s 累积误差大，VLA-World 靠反思修正减少漂移。

## 一个最值得记住的 insight

论文里有个消融结论，我建议你重点记：**反思模块的贡献最大，比生成模块还大。**

去掉反思，L2 从 0.30 涨到 0.85，涨了 0.55 米。去掉生成，只涨 0.38。去掉感知，涨 0.45。反思 > 感知 > 生成。

这说明什么？说明 VLA-World 这套架构的真正价值不在"能生成未来帧"，而在"能对生成的未来做推理评估"。生成只是手段，反思才是目的。这给"想象+反思"这条路线一个非常明确的指引：改进反思模块的收益会比改进生成模块更大。这是个很重要的诊断。

## 局限：别替它过度外推

论文没有显式列 Limitations 章节，这是个小问题。以下是我们读出的几个局限。

第一，生成分辨率只有 128×192，远低于实际驾驶需求。实际驾驶摄像头都是 1080P 以上，128×192 这种低分辨率未来帧能不能支持远距离行人的精细识别，论文没讨论。这是个安全相关的隐患。

第二，短期预测只有 0.5 秒，长程轨迹只有 3 秒。高速公路场景 3 秒 horizon 可能不够。而且 0.5 秒生成一帧，相当于 2Hz 帧率，对快速动态场景——比如突然切入——响应可能滞后。

第三，GRPO 的 5 个 reward 都是 rule-based，设计依赖人工先验。reward 之间的权重 λ 论文没给具体值，调参空间大。复杂城市场景——施工区、非标准车道——rule-based reward 是否够用没讨论。

第四，VQGAN 离散视觉 token 虽然兼容 LLM 自回归，但生成质量受 codebook 限制。FID 9.8 只比 GEM 10.5 略好，没有显著超越。而且离散化损失高频细节，对交通灯颜色这种安全关键信息可能不利。

第五，6 个视角全部进序列，token 数量大，但论文没报告推理延迟。实时驾驶需要 10Hz 以上决策，2B 模型加完整 pipeline 自回归一次的延迟是否达标，论文没给。这是部署的关键问题。

第六，消融只在 nuScenes 单一 benchmark 上做，没验证跨数据集泛化——比如 Waymo、Argoverse。nuScenes 是 2Hz 采样的特性，可能让方法过拟合到低帧率场景。

所以 VLA-World 是"想象+反思"路线在驾驶场景的一个强证据，但不是终点。它证明了反思比生成更重要，但实时性、分辨率、跨数据集泛化都还有空间。它和机器人 WAM 形成一个有意思的对照：机器人 WAM 重端到端实时控制，VLA-World 重显式反思推理，两条路都还没走到工业部署。

## 一句话收束

VLA-World 把自动驾驶的 VLA 和世界模型合并：用 Qwen2-VL-2B 加 VQGAN 自回归生成未来帧，再对这个想象出来的未来做反思修正轨迹，三阶段训练含 GRPO 强化推理链。它最有力的论证是消融显示"反思模块贡献最大、比生成还大"——这把"想象+反思"路线的改进重心精确定位到了反思层。
