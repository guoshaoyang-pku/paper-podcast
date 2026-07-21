# DreamZero — 结构化卡片

> 论文：**World Action Models are Zero-shot Policies**（NVIDIA, arXiv 2602.15922, 2026-02-19）
> 配套：`card.json`(机器可读) · `architecture.md`(架构图) · `podcast.md`(口播稿)

## 一句话定位

把 14B video diffusion 改造成同时生成未来视频和动作的 World Action Model，用 inverse-dynamics 视角让 video prior 直接成为 policy，并通过系统优化把它压到 7Hz 实时闭环。

## 它要解决的问题

**问题**：现有 VLA 只能在语义层泛化（换物体、换指令能行），但换一个没训练过的动作/技能（如解鞋带）就失败。

**为什么 prior work 不够**：VLA 从 VLM 初始化，继承了语义先验，但没有继承物理动态先验——它学 observation→action 映射，没有"动作会让世界怎么变"的建模，所以必须靠大量重复示教覆盖每个技能。之前的 WAM（BagelVLA/Cosmos-Policy 等）要么 cascaded、要么没解决实时推理、要么没系统验证跨本体迁移。

## 输入 / 输出

| 方向 | 名称 | 类型 | 说明 |
|---|---|---|---|
| 输入 | 视觉上下文 | image/video latent | 当前+历史；多视角通道维拼接；VAE 编码 |
| 输入 | 语言指令 | text | 冻结 text encoder |
| 输入 | 本体感受 | vector | 新增 state encoder |
| 输出 | 未来视频 latent | latent | 每 chunk K=2 帧 |
| 输出 | 动作 chunk | continuous | AgiBot H=48@30Hz / DROID H=24@15Hz；每 chunk 1.6s |

最大上下文：8 latent 帧 = 33 raw 帧 = 6.6s。

### 输入拼接 protocol

```
<bos_context>[(C_1, A_1), ..., (C_{k-1}, A_{k-1})]<eos_context>
<bos_lang>TextEnc(c)<eos_lang>
<bos_state>StateEnc(q_k)<eos_state>
<bos_chunk>[noisy z_k^t, noisy a_k^t]<eos_chunk>
```

逐 token 解释：

- `<bos_context>...<eos_context>`：历史 chunk 序列。训练时是 clean teacher-forcing context {(z_j, a_j)}_{j<k}，每 chunk 含 K=2 video latent 帧 + 对应 action latent。推理时前若干 chunk 的 KV-cache 已被真实观测替换。
- `<bos_lang>...<eos_lang>`：语言指令 c 经冻结 text encoder。注意是 **cross-attention 注入每个 DiT block**，不是序列拼接。
- `<bos_state>...<eos_state>`：当前 chunk k 的 proprioceptive state，经新增 state encoder。
- `<bos_chunk>...<eos_chunk>`：当前要联合去噪的 [video latent z_k^t, action latent a_k^t]。z 是 Wan-VAE latent（空间 104×60，channel 16，时间 K=2），a 是归一化连续动作。两者共享 timestep t_k，联合预测 velocity v_k = [z_k^1, a_k^1] - [z_k^0, a_k^0]。

**关键**：多视角在图像层就在通道维拼成单帧再过 VAE，不是序列拼接；text 是 cross-attention 而非 token 拼接；真正做序列拼接的是 chunk 级的 [video, action] 对。

## 数据集

| 数据 | 规模 | 备注 |
|---|---|---|
| AgiBot G1 自采（主） | ~500h | 7193 episodes，22 个真实环境，平均 4.4min/42 subtasks，强调多样非重复 |
| DROID | 公开 | Franka 单臂，验证在开源异质数据上同样有效 |
| YAM robot video-only | 20 min | robot-to-robot 跨本体迁移，只用视频 |
| Human egocentric video-only | 12 min | human-to-robot 跨本体迁移，只用视频 |
| Post-training | 33h+12h+40h | 叠衣服/水果打包/收餐桌 |

## 架构（详见 architecture.md）

- **主干**：Wan2.1-I2V-14B-480P（image-to-video diffusion，14B）
- **冻结**：VAE、text encoder、image encoder
- **新增（最小参数）**：state encoder、action encoder/decoder
- **结构**：autoregressive DiT + flow matching，chunk-wise，teacher forcing
- **多视角**：通道维拼成单帧，不改 backbone

**为什么这样设计**：保留 video diffusion 物理先验；AR 结构保原生帧率、避免双向 subsample 错位；chunk-wise + KV-cache 让长上下文一次前向；闭环用 GT 观测替换 KV-cache 里预测帧，消除 AR 误差累积。

### 数值 sense

| 项 | 值 |
|---|---|
| DiT 规格 | 14B；hidden_dim=5120；40 层 × 40 heads；ffn=13824 |
| 分辨率 | 480P（面积固定，宽高比随输入，典型 832×480） |
| VAE | Wan 3D causal VAE；空间 8× 下采样（832×480→104×60）；时间 4×（4 raw→1 latent）；latent channel=16 |
| 每帧 latent 维 | ~104×60×16 ≈ 10 万 |
| Chunk | K=2 latent 帧 = 8 raw 帧；5FPS → 1.6s；AgiBot H=48@30Hz，DROID H=24@15Hz |
| 上下文 | M=4 chunks = 8 latent 帧 = 33 raw 帧 = 6.6s |
| 动作 | 连续相对关节位置；AgiBot G1 双臂移动平台（DOF ~十几到二十几），DROID Franka 7-DOF+夹爪 |
| 训练 | 100K steps × global batch 128（AgiBot 与 DROID 都是）；全参数更新（LoRA 试过不行）；冻结 text/image encoder + VAE |

**给听众的标尺**：14B DiT、16 通道 latent、K=2/M=4、1.6 秒 chunk、6.6 秒上下文、100K steps × batch 128——这是非常大的训练量，后面听其它 WAM 论文时可以拿它做参照。

## 关键技术

1. **Joint video-action flow matching** — 单模型对 [z_t, a_t] 联合预测 velocity，共享 timestep，chunk 间 teacher forcing。证据：失败主要来自 video 预测错，不是 action 提取错——说明对齐很紧。
2. **AR + KV-cache 闭环** — 每 chunk 执行后用 GT 观测替换预测 KV，消除 compounding error；这是 WAM 作为 policy 比纯视频生成强的地方。
3. **DreamZero-Flash** — video timestep 偏向高噪声（Beta(7,1)，E[t]=0.125），action 仍 uniform，让模型学"video 很脏时也输出干净 action"，匹配 1-step 推理条件。1 步从 52%→74%。
4. **系统+实现加速栈** — CFG 双卡并行 / DiT caching / torch.compile+CUDA Graphs / NVFP4 / cuDNN，叠加 38×，5.7s→150ms。
5. **Cross-embodiment** — 跨本体数据只用 video objective（无 action），与主数据 1:1 共训；few-shot 适配用 30min play data post-train。

## 关键结果

| 指标 | DreamZero | 最强 baseline | setup |
|---|---|---|---|
| Seen tasks avg progress | 62.2% | 27.4% (pretrained VLA) | AgiBot G1, unseen env+obj |
| Unseen tasks avg progress | 39.5% | 16.3% (pretrained VLA) | AgiBot G1 |
| Unseen tasks success | 22.5% | 12.5% (GR00T N1.6) | DROID-Franka |
| Robot2Robot transfer | 55.4% (from 38.3%) | — | 20min YAM video-only |
| Human2Robot transfer | 54.3% (from 38.3%) | — | 12min human video-only |
| Few-shot 新本体 | 30min play data → YAM，保留 zero-shot | — | 55 轨迹/11 任务 |
| 推理延迟 | 150ms (7Hz) | 5.7s (naive) | GB200 + 全优化 + Flash |
| Flash 1-step | 74% | 83%(4-step)/52%(naive 1-step) | table bussing |

## Insights

1. WAM 把 policy 学习从 observation→action 模仿，转成 inverse dynamics——这是它能用异质非重复数据而 VLA 不能的根本原因（p3）。
2. **失败主要来自 video 预测错，不是 action 提取错**——意味着改进 video backbone 直接转化成 policy 改进（p13）。这是 WAM 路线最有力的 scaling 论证。
3. 数据多样性 > 重复性：同 500h，diverse vs repetitive 在 PnP Easy 是 50% vs 33%（p17）。
4. WAM 对模型规模更敏感：14B vs 5B 是 50% vs 21%；VLA 放到 14B 在 diverse data 上仍是 0%（p17）。
5. 闭环 GT 观测替换 KV-cache，消除 AR 误差累积——这是 WAM 作为 policy 比纯视频生成强的地方（p6）。

## vs 同类工作

- **vs Cascaded WAM（BagelVLA）**：BagelVLA 先关键帧再 action，DreamZero 是 joint 端到端；BagelVLA RFG 1.23s/chunk，DreamZero 150ms。
- **vs Modular MPC（Cosmos-Policy）**：Cosmos 用 world rollout + value scoring + MPC 做 test-time planning，DreamZero 不做 search 直接出 action，所以能 7Hz。
- **vs latent world model（V-JEPA2/Dreamer）**：latent 建模 p(s_{t+1}|s_t,a_t) 前向动力学，部署需 goal-conditioned planning 或搜索；WAM 直接建模 p(o,a|history)，无 test-time 优化，能实时。

## 局限

**论文自承**：
- 仍 System 1，视觉记忆 6.6s，长程需 System 2 或更长上下文（p18）。
- 高精度任务（亚厘米）仍是 BC 通病，diverse pretraining 稀释密集示教（p18）。
- 7Hz 依赖 2×GB200，比消费级 VLA（20Hz+）贵（p18）。
- 跨本体迁移成功率仍 moderate，是 early signal（p16）。

**我们读出**：
- 跨本体实验 YAM/AgiBot 都是双臂 parallel gripper，形态接近，收益部分来自此。
- seen/unseen task 边界依赖人工定义（动作+物体类型），有主观空间。
- DiT caching、NVFP4 不是数学等价，论文未给完整精度对照。

## 可复现性

- 代码：https://github.com/dreamzero0/dreamzero
- 权重：开源（论文声明）
- 仿真基准：PolaRiS / Genie Sim 3.0
- 真机评测：AgiBot G1（4 台）+ Franka（DROID）

## 论文重要图（详见 `figures.md`）

| 图 | 页 | 重要性 | 一句话 |
|---|---|---|---|
| [Figure 4](../../extracted/dreamzero/pages/p06.png) | 6 | key | 模型架构图，全文最该看，三路输入→AR DiT→video/action 两路输出 |
| [Figure 1](../../extracted/dreamzero/pages/p01.png) | 1 | key | 门面图，WAM 四大卖点（异质数据/zero-shot/跨本体/few-shot） |
| [Figure 5](../../extracted/dreamzero/pages/p09.png) | 9 | key | Flash 解耦 noise schedule，Beta(7,1) 让 1 步推理 52%→74% |
| [Figure 8](../../extracted/dreamzero/pages/p13.png) | 13 | key | Seen Task 结果，DreamZero 62.2% vs 27.4% |
| [Figure 9](../../extracted/dreamzero/pages/p14.png) | 14 | key | Unseen Task 结果，DreamZero 39.5% vs 16.3% |
| [Figure 11](../../extracted/dreamzero/pages/p15.png) | 15 | key | 跨本体迁移，20min/12min video-only 涨到 54%+ |
| [Figure 12](../../extracted/dreamzero/pages/p16.png) | 16 | key | Few-shot 30min 适配新本体 |
| [Figure 13](../../extracted/dreamzero/pages/p20.png) | 20 | key | 为什么选 AR 不选 bidirectional |
| [Figure 16](../../extracted/dreamzero/pages/p29.png) | 29 | key | 失败也对齐——改进 policy 等价于改进 video backbone |
| [Figure 2](../../extracted/dreamzero/pages/p03.png) | 3 | key | Video-Action 对齐成功案例 |
| Figure 3/6/7/10/14/15 | — | supportive | 见 figures.md |

## 标签

`WAM` `video diffusion` `autoregressive` `flow matching` `cross-embodiment` `KV-cache` `NVIDIA` `joint world-action`
