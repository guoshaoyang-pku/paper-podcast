# Motus — 结构化卡片

> 论文：Motus: A Unified Latent Action World Model（Tsinghua / PKU / Horizon Robotics, arXiv 2512.13030, 2025-12-25）
> 配套：`card.json`（真相源）｜`architecture.md`｜`figures.md`｜`podcast.md`

## 一句话定位

用一个 Mixture-of-Transformers 把 VLM（理解）、Wan2.2（视频生成）、Action 专家三路并到一个模型里，靠 UniDiffuser 式调度在 VLA/WM/IDM/VGM/Video-Action 联合预测五种模式间切换；并用光流编码的 latent action 让无动作标签的海量视频也能预训练 action 专家。

## 基本信息

| 项 | 值 |
|---|---|
| slug | motus |
| arxiv | 2512.13030 |
| 日期 | 2025-12-25 |
| 机构 | 清华（THBI/BNRist, Tsinghua-Bosch JMLC）· 北大 · 地平线 |
| 类别 | Joint WAM |
| 作者 | Hongzhe Bi, Hengkai Tan, Shenghao Xie 等 16 人 |

## 问题

**要解决什么**：具身智能被切成五块独立建模——VLA、世界模型、逆向动力学、视频生成、Video-Action 联合预测——各训各的，既无法共享先验又吃不了异构数据。Motus 要把五块统一进一个生成式模型，同时让无动作标签的互联网视频/人类视频也能预训练 action 专家。

**为什么 prior work 不够**：UWM 是理论原型，但要么从头训、要么用小底座，缺 VLM/VGM 先验；F1 把 VLA 和 IDM 拼起来但排除 WM/VGM；latent action 一脉（LAPA/AdaWorld）用 RGB 重建或 β-VAE，引入任务无关外观噪声，且没法直接对齐真实控制信号。

## 输入 / 输出

### 输入

| 名称 | 类型 | 说明 |
|---|---|---|
| current observation o_t | image/video | 条件帧，Wan2.2-VAE 编码到 latent |
| future observations o_{t+1:t+k} | video latents | 要预测/去噪的未来视频帧 |
| future actions a_{t+1:t+k} | latent action / continuous | Stage2 是 14 维 latent action，Stage3 是真实动作 |
| language instruction ℓ | text | VLM 理解专家（Qwen3-VL-2B 末层 token）处理 |
| proprioception p | vector | joint position 控制下与 action 同空间 |

### 输出

| 名称 | 类型 | 说明 |
|---|---|---|
| future video latents | latent | VGM/WM/Joint 模式产出 |
| action chunk | continuous | RoboTwin chunk=16；真机 chunk=48 @30Hz（1.6s）|

**控制频率**：动作 30Hz，chunk=48 步（1.6s）；视频采样 5Hz、每 chunk 8 帧（action-dense video-sparse，视频率是动作率 1/6）；推理 10 步 flow matching。

### 输入拼接 protocol

```
<bos_cond>[ VAE(o_t) ]<eos_cond>
<bos_lang>[ VLM_last_layer(ℓ) ]<eos_lang>
<bos_video>[ noisy o^τo_{t+1:t+k} (8帧@5Hz) ]<eos_video>
<bos_action>[ noisy a^τa_{t+1:t+k} (48步@30Hz, 或 latent action 14维) ]<eos_action>
```

- `<bos_cond>...<eos_cond>`：条件帧 o_t，Wan2.2-VAE 编码；五模式共用。
- `<bos_lang>...<eos_lang>`：语言指令不进自回归序列，走 VLM 理解专家 → 经 Tri-model Joint Attention（共享多头自注意力层）注入。是 cross-expert attention，不是 token 拼接。
- `<bos_video>...<eos_video>`：未来视频帧 8 帧 @5Hz。训练时与 action 共同采样 rectified-flow timestep τo；推理时按模式切换（VGM/Joint 要去噪，IDM/VLA 保持噪声只作条件）。
- `<bos_action>...<eos_action>`：动作 chunk 48 步 @30Hz。Stage2 是 14 维 latent action（光流 VAE），Stage3 是真实动作。

**三个坑**：
1. language 不进序列拼接，是 VLM 专家 → 共享 attention 注入（和 DreamZero cross-attention 思路一致，但 Motus 走 MoT 共享自注意力层，不是单 DiT 加 cross-attn）。
2. video 和 action 在同一条去噪序列里，但分配**不同 timestep**（τo、τa 独立从 U(0,Tτ) 采）——五模式靠 (τo, τa) 起止值切换。
3. 训练/推理序列结构相同，模式不同时哪些 token 真正被去噪、哪些只做条件不同。多视角未显式处理（真机单视角双臂）。

## 数据集

| 数据 | 规模 | 备注 |
|---|---|---|
| Egodex（ego-centric human）| 230,949 段 | Level2 人类第一视角，无动作标签 |
| Agibot World Colosseo | 728,209 段 | Level5 Genie-1 机器人轨迹 |
| RDT 数据 | 6,083 段 | Level5 Aloha |
| RoboMind Franka | 9,589 段 | Level5 Franka |
| RoboMind Aloha | 7,272 段 | Level5 Aloha |
| RoboTwin 2.0 | 27,500 段 | Level3 合成，强 randomization，50 任务 |
| AnyPos task-agnostic | 1,000 段 | Level4 Curobo 随机采样，锚定 latent action |
| In-house target-robot | 2,000 段 | Level6 真机，Stage3 SFT，每任务 100 轨迹 |

## 架构（摘要）

| 项 | 值 |
|---|---|
| backbone | Wan2.2-TI2V-5B（VGM）+ Qwen3-VL-2B（VLM）+ Action Expert（641.5M）+ Und. projection（253.5M）|
| 总参数 | ~8B |
| 类型 | Mixture-of-Transformers + UniDiffuser-style rectified flow 调度 |

**关键组件**：
- Video Generation Expert（Wan2.2-TI2V-5B，自带 Wan2.2-VAE）
- Understanding Expert（Qwen3-VL-2B 末层 token → 30 层 Transformer hidden 512/24 heads）
- Action Expert（30 层 Transformer hidden 1024/24 heads，与 Wan 同深度，含 AdaLN/FFN/Tri-model Joint Attention）
- Tri-model Joint Attention（三专家共享多头自注意力层，cross-modal 融合）
- UniDiffuser-style scheduler（(τo, τa) 起止值切换五模式）
- Latent Action VAE（DPFlow→RGB→DC-AE→4×512→14 维）

**为什么这样设计**：不把 observation token 和 action token 串进单一 transformer（UWM 那样会从头训丢先验），而是让三专家各自保留独立 Transformer 模块（保住预训练先验不被互相稀释），只在多头自注意力层共享拼接做融合。UniDiffuser 调度让一个模型覆盖五种分布，不需要五个 head。latent action 用光流而不是 RGB 重建，是为了剥掉任务无关外观、把"像素级 delta action"作为跨本体通用语言。

→ 详见 `architecture.md`。

### 数值 sense

| 项 | 值 |
|---|---|
| DiT 规格 | Action Expert hidden=1024, 30 layers, 24 heads, GELU；Und. projection hidden=512, 30 layers, 24 heads；VGM=Wan2.2-5B dense DiT |
| 分辨率 | Wan2.2 原生 720P@24fps；Motus 训练把视频降到 5Hz、每 chunk 8 帧 |
| VAE | Wan2.2-VAE：空间 16× 下采样，时间 4×（压缩比 4×16×16），latent channel=16；加 patchification 后总 4×32×32 |
| 每帧 latent 维 | Wan2.2-VAE 单帧通道 16；720P 输入下空间维 ~40×22 → 每帧 ~1.4e4 维（远小于 Wan2.1 14B 的 1e5，因 Wan2.2-VAE 空间压缩更狠 16× vs 8×）|
| Chunk | Video 8 帧 @5Hz = 1.6s；Action 48 步 @30Hz = 1.6s。视频率压成动作率 1/6，平衡 token 数 |
| 上下文 | 仅条件当前帧 o_t（不显式建模长历史 chunk，区别于 DreamZero 的 6.6s）|
| 动作 | Stage2: 14 维 latent action（光流 VAE，跨本体通用）；Stage3: 真机动作（AC-One / Agilex-Aloha-2 双臂）；RoboTwin chunk=16，真机 chunk=48 |
| 训练 | Stage1 ~8000 GPU·h（只 VGM）；Stage2 ~10000 GPU·h（三专家，VLM 冻结）；Stage3 ~400 GPU·h（SFT）。batch 256, AdamW, lr 8e-5→5e-5→1~5e-5。推理 10 步 flow matching，Logit Normal 采样。总 ~18400 GPU·h |

## 关键结果

| 指标 | 值 | 最强 baseline | setup |
|---|---|---|---|
| RoboTwin Randomized 50-task avg | **87.02%** | π0.5 43.84% / X-VLA 72.84% / from-scratch 77.00% | 50 任务多任务联合训，40k steps |
| RoboTwin Clean 50-task avg | **88.66%** | π0.5 42.98% / X-VLA 72.80% / from-scratch 77.56% | Clean 场景 |
| LIBERO-Long avg | **97.6** | X-VLA 97.6 / OpenVLA-OFT 94.5 / π0 85.2 | 10 长程任务 |
| AC-One 真机 9 任务 avg | **63.22%** | π0.5 14.79% / w/o-pretrain 25.86% | 双臂，partial success rate |
| Agilex-Aloha-2 真机 5 任务 avg | **59.30%** | π0.5 48.60% / w/o-pretrain 26.60% | 双臂 |
| IDM mode action MSE | **0.014** | ResNet18+MLP 0.044 / DINOv2+MLP 0.122 | RoboTwin, chunk=16, 100 样本 |
| VLA mode vs Joint mode | VLA 83.90% / Joint 87.02% | — | 同权重切模式 |
| World Model 生成质量 | FID 9.46 / FVD 49.28 / SSIM 0.886 | AC-One FID 12.96 / FVD 73.13 | Agilex 真机，给定动作预测视频 |

## 关键技术

1. **Mixture-of-Transformers + Tri-model Joint Attention**——三专家独立 Transformer 模块 + 共享多头自注意力层，保先验又跨模态融合。
2. **UniDiffuser-style 调度**——(τo, τa) 起止值切换 VLA/WM/IDM/VGM/Joint 五模式，一个权重五种推理。
3. **光流 latent action**——DPFlow→RGB→DC-AE→14 维，剥外观留运动，跨本体通用，让无标签视频也能预训练 action 专家。
4. **Action-Dense Video-Sparse 预测**——视频率压成动作率 1/6，平衡 token 数，防 video 主导 attention。
5. **三阶段渐进训练 + 六层数据金字塔**——Stage1 视频 → Stage2 latent action → Stage3 真动作，逐层落地。

## Insights

- 五模式统一指给两个模态各分配独立 timestep，用 (τo, τa) 起止值切换——UniDiffuser 思想在机器人上的落地。
- action 专家终于能预训练了。光流 latent action 让海量无标签视频也能喂 action 专家，这是相对 VLA 路线的根本增量。
- action-dense video-sparse 反直觉但关键：压低 video 帧率不是为省算力，是为让 action token 不被淹没在 attention 里。
- 统一模型不一定输专门模型：IDM 模式 action MSE 0.014 打败专门训的 ResNet18+MLP（0.044）和 DINOv2+MLP（0.122）。
- Joint 模式（87.02%）略胜 VLA 模式（83.90%）：把 video 也一起生成反而让 action 更准，给 WAM 联合生成论断添正面证据。

## vs 同类工作

- **vs UWM**：UWM 单 transformer 串 token、从头训丢先验；Motus MoT 三专家保先验、只共享 attention，且能用互联网视频预训练 action 专家。
- **vs DreamZero**：都做 video-action 联合生成，但 DreamZero 单一 AR DiT 共享 timestep + KV-cache 闭环重实时（7Hz）；Motus 三专家 MoT + 独立 timestep + UniDiffuser 调度重统一性（五模式）和跨本体数据。
- **vs F1**：F1 拼接 VLA+IDM 但排除 WM/VGM；Motus 五种全统一且引入 latent action。
- **vs LAPA/AdaWorld**：LAPA 用 RGB 重建引入外观噪声，AdaWorld 用 β-VAE；Motus 用光流（天然剥外观）+ 14 维对齐机器人动作尺度。
- **vs Bagel**：Bagel 两专家（理解+生成）MoT；Motus 扩展到机器人加 Action 专家变三专家。

## 局限

- 论文自承：仍是 System 1，长程推理受限于上下文；未来要从互联网规模通用视频学 latent action（目前还没做到）。
- 上下文建模弱：只条件当前帧 o_t，不像 DreamZero 有 6.6s 历史记忆；长程"记忆"靠 VLM 隐式承担，未量化。
- 真机只比 π0.5，没和 DreamZero/GR00T N1.6/X-VLA 真机直接对照；Motus 在 Put Bread into Oven（Agilex 34% vs π0.5 36%）甚至略输，泛化收益不均匀。
- latent action 14 维是经验设定，不同本体动作维度差异大，是否够用未充分讨论。
- Stage2→Stage3 的"latent action → real action"迁移跳跃未单独消融。
- 五种模式实验密度不均——VGM 只有可视化无定量，"每种模式都强"的证据强度不齐。

## 可复现性

| 项 | 值 |
|---|---|
| code | https://motus-robotics.github.io/motus |
| weights | Wan2.2-TI2V-5B / Qwen3-VL-2B 公开底座；Motus 自身权重开源时间未明确 |
| sim benchmark | RoboTwin 2.0、LIBERO-Long、VLABench |
| real eval | AC-One 双臂 + Agilex-Aloha-2 双臂，9+5 长程任务 |

## 标签

`unified model` `Mixture-of-Transformers` `UniDiffuser` `latent action` `optical flow` `rectified flow` `cross-embodiment` `Wan2.2` `VLA` `world model` `IDM` `Tsinghua`
