# VLA-World — 结构化卡片

> 论文：VLA-World: Learning Vision-Language-Action World Models for Autonomous Driving（上交 / 华为，arXiv 2604.09059，2026-04-10）
> 配套：`card.json`（真相源）｜`architecture.md`｜`figures.md`｜`podcast.md`
>
> 注：PDF 文件名为 `VLA-World_Autonomous_Driving`，任务原给 slug=cosmos-policy，但读论文确认真实身份是 **VLA-World**（不是 Cosmos-Policy），按指示改用真实 slug=vla-world 产出。

## 一句话定位

把自动驾驶的 VLA 和世界模型合并：先用短期轨迹预测引导生成下一帧未来图像，再对这个自己想象出来的未来帧做反思推理修正长期轨迹，用 Qwen2-VL-2B + VQGAN 自回归出视觉 token，三阶段（预训练→SFT→GRPO）训练。

## 基本信息

| 项 | 值 |
|---|---|
| slug | vla-world |
| arxiv | 2604.09059 |
| 日期 | 2026-04-10 |
| 机构 | 上海交通大学（MoE Key Lab of AI）· 华为中央研究院 |
| 类别 | Cascaded WAM |
| 作者 | Guoqing Wang, Pin Tang, Xiangxuan Ren 等 6 人 |

## 问题

**要解决什么**：自动驾驶里 VLA 直接从观测出动作，缺对其它动态 agent 的时空建模，没前瞻；世界模型能生成未来但只能"模拟"不能"推理评估"想象出来的未来。VLA-World 要把世界模型的预测想象和 VLA 的反思推理合并：先想象未来帧，再对自己想象出来的未来做反思，修正轨迹。

**为什么 prior work 不够**：VLA 方法（ELM/OmniDrive/DriveMoE 等）直接映射观测到轨迹，缺显式时空动态建模。世界模型（DriveDreamer/DrivingWorld/OccWorld 等）能生成未来但缺反思推理。FSDrive 虽引入未来帧作 CoT 中间步，但只生成前视、直接回归 waypoint 不评估物理可行性，且缺 RL 强化反思链。

## 输入 / 输出

### 输入

| 名称 | 类型 | 说明 |
|---|---|---|
| multi-view observations o_t | image | nuScenes 6 个相机视角图像；经 Qwen2-VL 视觉编码器编码 |
| ego status S_t | vector | 自车速度、加速度、yaw rate 等 CAN 信号 |
| mission goal g | text/discrete | 导航指令 left/right/forward |

### 输出

| 名称 | 类型 | 说明 |
|---|---|---|
| perception results | text | 检测周围动态 agent、3D 位置、运动路径、路肩距离、可行驶区域 |
| short-term prediction | text+trajectory | 下 0.5s 的 waypoint 和行驶方向，作生成未来帧的条件 |
| imagined future frame x̂_{t+1} | image | VQGAN 自回归生成的下一帧视觉 token 序列（任一视角），128×192 |
| reflective reasoning | text | 对想象未来帧的反思：重要 agent、潜在风险、安全裕度 |
| final trajectory τ̃_{t:t+H} | trajectory | 修正后的 3s horizon ego 轨迹，0.5s 间隔，6 个 waypoint（BEV）|

**控制频率**：推理 step 0.5s；轨迹 horizon 3s（6 个 waypoint）；nuScenes 2Hz 采样；8×80GB GPU 训练。

### 输入拼接 protocol

```
<bos_obs>[ Qwen2VL(I^1_t...I^6_t, S_t) ]<eos_obs>
<bos_goal>[ g ]<eos_goal>
<bos_perception>[ 检测+3D位置+路肩距离 文本 ]<eos_perception>
<bos_prediction>[ 0.5s waypoint + direction 文本 ]<eos_prediction>
<bos_visual>[ VQGAN token 序列 q^k_1...q^k_N (生成未来帧) ]<eos_visual>
<bos_think>[ 反思文本:重要agent+风险 ]<eos_think>
<bos_action>[ 高层动作 ]<eos_action>
<bos_answer>[ 3s 轨迹 waypoint 列表 ]<eos_answer>
```

- `<bos_obs>...<eos_obs>`：6 相机视角 + ego status，Qwen2-VL 编码。多视角 token 序列拼接，不是通道拼接。
- `<bos_goal>...<eos_goal>`：mission goal（left/right/forward），作自回归上下文。
- `<bos_perception>...<eos_perception>`：感知输出文本，结构化场景描述，scene-level world state。
- `<bos_prediction>...<eos_prediction>`：短期预测，0.5s waypoint + 方向，作生成条件。由 ego 历史 + goal 经物理动力学预测。
- `<bos_visual>...<eos_visual>`：想象未来帧——VQGAN codebook 离散 token 序列，自回归生成。条件是 o_{1:t} + 短期预测 τ̂_{t+1}。可指定任一视角（如 CAM_FRONT_LEFT 0.5s later）。
- `<bos_think>...<eos_think>`：反思推理，分析想象未来帧里重要 agent、风险、安全裕度。
- `<bos_action>...<eos_action>`：高层动作决策。
- `<bos_answer>...<eos_answer>`：最终修正轨迹，3s horizon，6 个 BEV waypoint。

**三个坑**：
1. language（goal/perception/prediction/think/action）全部是自回归 token 序列拼接——VLA-World 走纯 LLM 自回归范式，所有模态（包括视觉生成）都是 next-token prediction。视觉未来帧是 VQGAN 离散 token，不是连续 latent diffusion。
2. 多视角是序列拼接（6 视角 token 都在序列里），生成时可指定任一视角——多视角一致性靠 Stage1 预训练显式约束（不同于 FSDrive 只生前视）。
3. 训练和推理序列结构一致，但三阶段侧重点不同：Stage1 只激活视觉生成；Stage2 SFT 全 pipeline 监督；Stage3 RL 用 GRPO 采样打分。

## 数据集

| 数据 | 规模 | 备注 |
|---|---|---|
| nuScenes-GR-20K（自建）| 20K 样本 | 从 nuScenes 衍生，专为生成未来帧+条件推理设计，SFT 和 RL 阶段用 |
| nuScenes（主评测）| 1000 scenes | 6 相机，2Hz，带 3D 标注；L2 误差和 collision rate 评测 |
| 图像-指令预训练数据 | 480k | Stage1 视觉生成激活，继承自 FSDrive 对齐策略 |

## 架构（摘要）

| 项 | 值 |
|---|---|
| backbone | Qwen2-VL-2B（初始化自 FSDrive 对齐策略）+ VQGAN（视觉 tokenizer）|
| 总参数 | 2B（Qwen2-VL-2B）；VQGAN tokenizer 离散化图像 |
| 类型 | 自回归 next-token VLM + VQGAN 离散视觉生成 + 三阶段训练 |

**关键组件**：
- Perception 模块：6 相机 + ego status → 检测动态 agent、3D 位置、路肩距离、可行驶区域
- Short-term Prediction 模块：ego 历史 + goal → 0.5s waypoint + 方向（惯性+意图加速度融合）
- Condition-guided Generation 模块：VQGAN 自回归生成未来帧 token，条件 o_{1:t} + τ̂_{t+1}，可指定视角
- Thinking with Visual Tokens 模块：对想象未来帧做反思，识别重要 agent、风险、安全裕度
- Action & Trajectory Planning 模块：高层动作 + 3s 修正轨迹（6 个 BEV waypoint）
- GRPO 强化学习：value-free，组内归一化优势，5 个 rule-based reward

**为什么这样设计**：用纯自回归 next-token 范式统一感知、生成、推理、规划——所有模态（包括视觉未来帧）都是 token。短期轨迹引导生成未来帧，把高维未来采样到合理可信空间，再对具体未来做反思。三阶段训练对应三种知识：Stage1 生成知识，Stage2 概念知识，Stage3 推理知识。VQGAN 离散视觉 token 天然兼容 LLM 自回归。

→ 详见 `architecture.md`。

### 数值 sense

| 项 | 值 |
|---|---|
| DiT 规格 | Qwen2-VL-2B 主干（2B 参数 VLM transformer）；VQGAN tokenizer 离散 codebook |
| 分辨率 | 生成图像 128×192（FSDrive 同款）；输入 6 相机 nuScenes 原生分辨率 |
| VAE | VQGAN（非 VAE）：图像离散化为 codebook token 序列；无连续 latent 压缩比概念，token 数 = N |
| 每帧 latent 维 | VQGAN codebook token 序列，128×192 图像典型 ~256-1024 token 量级（N 论文未明示）|
| Chunk | 推理 step 0.5s；轨迹 horizon 3s = 6 个 waypoint；nuScenes 2Hz 采样 |
| 上下文 | o_{1:t} 历史观测 + ego status + goal；具体历史长度未明示 |
| 动作 | 轨迹 waypoint（BEV 2D）；高层动作离散 manoeuvre（forward/left/right + keep/acc/dec/stop）|
| 训练 | Stage1 预训练：30 epochs, AdamW lr 5e-4, per-device batch 16, 8×80GB, 480k。Stage2 SFT：12 epochs, lr 1e-4, nuScenes-GR-20K。Stage3 GRPO：1 epoch, lr 1e-6, global batch 16, 8 候选/prompt |

## 关键结果

| 指标 | 值 | 最强 baseline | setup |
|---|---|---|---|
| nuScenes L2 avg (ST-P3, 带 ego-state) | **0.26m** | FSDrive* 0.28m / OmniDrive* 0.33m / EMMA* 0.32m | 1s/2s/3s 平均, autoregressive |
| nuScenes Collision avg (ST-P3, 带 ego-state) | **0.08%** | FSDrive* 0.10% / OmniDrive* 0.30% | 1s/2s/3s 平均 |
| nuScenes L2 avg (UniAD, 带 ego-state) | **0.42m** | FSDrive* 0.45m / OccWorld 1.40m | UniAD 协议 |
| nuScenes Collision avg (UniAD, 带 ego-state) | **0.12%** | FSDrive* 0.16% / OccWorld 0.87% | UniAD 协议 |
| 未来帧生成 FID | **9.8** | FSDrive 10.1 / GEM 10.5 / Doe-1 15.9 | 128×192, autoregressive |
| 动作预测 F1 forward | **95.88%** | Qwen2-VL-2B† 92.60% / Qwen2-VL-2B 62.43% | lateral 动作 |
| 动作预测 F1 left/right | **74.22% / 75.06%** | Qwen2-VL-2B† 61.78% / 66.52% | 转向动作，提升最大 |
| 消融 w/o Reasoning L2 avg | 0.85m（去掉）/ 0.30m（full）| — | ST-P3，反思模块贡献最大（+0.55m）|

## 关键技术

1. **想象+反思闭环**——action-derived trajectory → 生成未来帧 → 反思修正。反思贡献最大（消融 +0.55m）。
2. **多视角 goal-conditioned 视觉生成**——扩展 FSDrive 只生前视，可按预测方向请求对应视角。
3. **三阶段训练**——生成知识（预训练）→ 概念知识（SFT）→ 推理知识（GRPO）。SFT 比 RL 重要。
4. **GRPO + 5 个 rule-based reward**——value-free RL，组内归一化优势，避免 reward hacking。
5. **物理动力学短期轨迹预测**——惯性加速度 + 意图加速度融合，a_eff = (1-λ)·a_hist + λ·a_goal。

## Insights

- 想象+反思闭环的关键在于把高维未来采样到"合理可信空间"再反思——短期轨迹预测起到约束采样到物理可行区域的作用。
- 反思模块贡献最大：消融 w/o Reasoning L2 从 0.30 涨到 0.85（+0.55m），远超 w/o Perception（+0.45）和 w/o Generation（+0.38）。"对想象未来做推理"比"生成未来"本身更重要。
- SFT 比 RL 重要：w/o SFT L2 0.85 vs w/o RL L2 0.71。RL 没有 SFT 冷启动无法导航大搜索空间。
- 纯自回归 next-token 范式能统一感知/生成/推理/规划——所有模态（包括视觉未来帧）都是 token，VQGAN 离散视觉 token 让视觉生成天然兼容 LLM 自回归。
- 多视角 goal-conditioned 生成是 VLA-World 对 FSDrive 的关键扩展：左转请求左视，想象更贴合驾驶意图。

## vs 同类工作

- **vs FSDrive**：FSDrive 引入未来帧作 CoT 中间步，但只生成前视、直接回归 waypoint 不评估物理可行性；VLA-World 多视角生成 + 反思修正 + GRPO。是 FSDrive 的直接升级版（同 Qwen2-VL-2B 初始化）。
- **vs DreamZero/BagelVLA（机器人 WAM）**：机器人 WAM 是 video-action 联合或 cascaded 生成动作；VLA-World 是自动驾驶，轨迹规划而非关节控制，且用 VQGAN 离散 token 而非连续 latent diffusion，纯 LLM 自回归范式。
- **vs DriveDreamer/DrivingWorld/OccWorld（驾驶世界模型）**：这些只生成未来不推理；VLA-World 在生成后加反思闭环，且用 VLM 主干而非纯视频 diffusion。
- **vs OmniDrive/ELM/DriveMoE（驾驶 VLA）**：这些直接映射观测到动作缺时空动态建模；VLA-World 显式建模未来演化并反思。
- **vs DreamVLA/WorldVLA（统一 VLA+世界模型）**：VLA-World 更聚焦驾驶场景的多视角+轨迹规划，且三阶段训练（含 GRPO）更系统。

## 局限

- 论文未显式列 Limitations 章节，以下为我们读出。
- 生成分辨率仅 128×192，远低于实际驾驶需求（1080P+），低分辨率未来帧能否支持精细风险识别（远距离行人）未讨论。
- 短期预测仅 0.5s、长程轨迹 3s，高速场景 3s horizon 可能不够；0.5s 生成一帧（2Hz），对快速动态场景（突然切入）响应可能滞后。
- GRPO 的 5 个 reward 都是 rule-based，设计依赖人工先验；reward 权重 λ 未给具体值；复杂城市场景（施工区、非标准车道）是否够用未讨论。
- VQGAN 离散视觉 token 虽兼容 LLM，但生成质量受 codebook 限制（FID 9.8 仅略胜 GEM 10.5）；离散化损失高频细节，对交通灯颜色等安全关键信息可能不利。
- 6 视角全部进序列，token 数量大，推理延迟未报告；实时驾驶需要 10Hz+ 决策，2B 模型 + 完整 pipeline 自回归一次的延迟是否达标论文未给。
- 消融只在 nuScenes 单一 benchmark，未验证跨数据集泛化（Waymo、Argoverse）；nuScenes 2Hz 采样特性可能让方法过拟合到低帧率场景。

## 可复现性

| 项 | 值 |
|---|---|
| code | https://vlaworld.github.io |
| weights | Qwen2-VL-2B 公开底座；VLA-World 自身权重开源情况未明确；nuScenes-GR-20K 声明将发布 |
| sim benchmark | nuScenes（ST-P3 + UniAD 协议，L2 + collision rate + FID）|
| real eval | 无实车部署，纯 nuScenes 离线评测 |

## 标签

`autonomous driving` `VLA world model` `reflective reasoning` `GRPO` `multi-view generation` `VQGAN` `Qwen2-VL` `nuScenes` `trajectory planning` `SJTU` `Huawei`
