# HarmoWAM — 协调泛化与精度的自适应世界动作模型

> 配套 `card.json`(真相源)。下表与卡片内容一一对应。

## 元信息

| 字段 | 值 |
|---|---|
| slug | harmowam |
| title | HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Models |
| authors | Qiuxuan Feng, Jiale Yu, Jiaming Liu, Yueru Jia, et al. |
| affiliation | Peking University · Simplexity Robotics · CUHK |
| arxiv | 2605.10942 |
| date | 2026-05-11 |
| venue | arXiv preprint (PKU) |
| category | Hybrid WAM |

**一句话**:把 WAM 的两条旧路线(Imagine-then-Execute 泛化好但不准 / Joint Modeling 准但 OOD 探索崩)用一个共享世界模型 + 两个互补 action expert + 一个过程自适应门控统一进来,transit 用 reactive expert 借世界模型泛化、interaction 用 predictive expert 借隐含 latent 精准对齐。

**tags**:WAM · hybrid expert · process-adaptive gating · Wan2.2 · diffusion policy · inverse dynamics · Franka · OOD generalization · PKU

## 问题

- **要解决什么**:现有 WAM 分两条路,Imagine-then-Execute transit 泛化强但 interaction 精度差;Joint Modeling interaction 精度高但 OOD transit 探索空间被 SFT 分布锁死。如何在一个端到端 WAM 同时拿泛化 transit 和精准 interaction。
- **为什么 prior work 不够**:Table 1 motivation 实验证明这是结构性 trade-off。Imagine-then-Execute 的 IDM 只看像素缺接触级精度(OOD interaction 掉 55%);Joint Modeling 把 action 绑死 SFT 分布(OOD transit 掉到 32%),即使初始化到目标附近 interaction 还能 95%——证明瓶颈是探索不是精度。两条路各自最优解都在对方最弱处失效。

## 输入 / 输出

### 输入

| 名称 | 类型 | 说明 |
|---|---|---|
| visual observation | image | 当前帧 RGB,3 视角(单臂 1 第三人称 + 2 腕部 / 双臂 1 全局 + 2 腕部),RealSense 640×480 |
| language instruction | text | 任务指令,经冻结 T5 编码 |
| proprioceptive state | vector | 单臂 7-DoF / 双臂 14-DoF(3 位移 + 3 Euler + 1 夹爪) |

### 输出

| 名称 | 类型 | 说明 |
|---|---|---|
| predicted video V_{t:t+H} | video | 13 帧 256×320 未来视频,5 步去噪 |
| action chunk | continuous | H=12 步,单臂 R^7 / 双臂 R^14,48Hz |

**控制频率**:action 生成 48Hz,chunk H=12,世界模型 5 步去噪。

### 输入拼接 protocol

```
<bos_ctx>[I_t (3-view RGB)]<eos_ctx>
<bos_lang>T5(l_t)<eos_lang>
<bos_wm>NoisyLatent x_ξ (256×320×C, 13 帧)<eos_wm>
<bos_cond>F_V_t ∈ R^{B×80×3072} (世界模型当前步隐含 latent)<eos_cond>
<bos_gate>F_img_t (SigLIP patch) → s_t ∈[0,1]<eos_gate>
<bos_chunk>
  [Predictive: noisy a_{t+1:t+12} 经 28-block DiT cross-attn(条件 F_img_t, F_text, F_V_t)]
  OR
  [Reactive: [DINOv2(V_s) patch; pool(F_V_s)] → OrientationDecoder → â_s]
<eos_chunk>
```

**逐 token 解释**:
- `<bos_ctx>...<eos_ctx>`:多视角不是序列拼接。世界模型只用其中一路(单臂第三人称 / 双臂全局)生成 13 帧未来视频;另一路喂 predictive expert 的 SigLIP 提供实时闭环视觉。
- `<bos_lang>...<eos_lang>`:T5 编码后 cross-attention 注入世界模型和 predictive DiT,不是 token 拼接。
- `<bos_wm>...<eos_wm>`:对 noisy latent 做 5 步 flow matching 去噪,产出 13 帧未来视频 + 隐含 latent F_V∈R^{B×80×3072}。80 token = Wan2.2-VAE 把 256×320 压到 16×20 latent 再 patch(2×2)。
- `<bos_cond>...<eos_cond>`:当前步隐含 latent F_V_t 是共享"隐式"条件,cross-attn 喂 predictive、token 维 concat 喂 reactive。
- `<bos_gate>...<eos_gate>`:SigLIP patch → MLP → s_t∈[0,1]。推理 s_t>0.5 走 predictive(interaction),否则走 reactive(transit)。
- `<bos_chunk>...<eos_chunk>`:两条 expert 由门控二选一,不并行。Predictive 用 diffusion loss,Reactive 用 Smooth L1。

**三个坑**:(1) language 是 cross-attention 不是序列拼接;(2) 多视角按模块分工不在序列层处理;(3) 训练时未来视频用 GT teacher-force 世界模型、推理用世界模型生成;门控训练用 keyframe pipeline 自动标 y。

## 数据集

| 数据 | 规模 | 备注 |
|---|---|---|
| 自采真实示教(主) | 6 任务 × 100 轨迹 | Franka FR3+UMI gripper;4 单臂 + 2 双臂;平均 280-400 步,多 stage 顺序评测 |
| DROID | 201,119 轨迹 | 公开 Franka Panda,世界模型预训练 |
| AgiBot | 3,017 轨迹 | AgiBot G1,世界模型预训练 |
| RoboMIND | 1,721,985 轨迹 | Franka/UR/Ark/Agilex/TienKung 多本体,世界模型预训练 |
| 世界模型预训练总计 | ~1.9M 轨迹 | 上述公开 + 闭源机器人数据 |

## 架构(摘要)

| 字段 | 值 |
|---|---|
| backbone | Wan2.2-TI2V-5B(5B MoE)+ Action DiT(1B)+ SigLIP + DINOv2-base + T5 + 门控 MLP |
| params | 世界模型 5B(stage1 全参 finetune,stage2 冻结);Predictive DiT 1B(28 blocks);Reactive=DINOv2-base+卷积;Gate=轻量 MLP |
| type | Hybrid WAM:共享世界模型 + 双 expert + 过程自适应门控 |

**关键组件**:
- World Model(Wan2.2-TI2V-5B):5 步去噪生成 13 帧 256×320 未来视频 + 隐含 latent
- Predictive Action Expert(1B DiT,28 blocks):隐含 latent cross-attn,diffusion 出 H=12 动作(interaction)
- Reactive Action Expert(DINOv2-base + OrientationDecoder):未来视频 patch 几何特征 + latent concat,卷积出动作(transit)
- Process-Adaptive Gating(MLP):SigLIP patch → s_t,硬路由
- 双 expert 都从同一世界模型取条件(显式视频 + 隐含 latent),物理先验共享

**为什么这样设计**:动机实验(Table 1)证明两条旧 WAM 是结构性 trade-off。HarmoWAM 不是折中而是分工——世界模型同时给显式未来(reactive 做泛化 transit)和隐含 latent(predictive 做精准 interaction),门控按阶段硬路由。

### 数值 sense

| 项 | 值 |
|---|---|
| DiT 世界模型 | Wan2.2-TI2V-5B,5B(MoE);VAE 压缩比 16×16×4;hidden 3072 |
| DiT predictive | 1B 参数,28 Transformer blocks,diffusion 去噪 |
| 分辨率 | 世界模型 256×320(VAE→16×20 latent);摄像头 640×480 |
| VAE | Wan2.2-VAE:空间 16×、时间 4×、latent 通道 16 |
| 每帧 latent 维 | 16×20×16 ≈ 5120;patch(2×2)→ 80 token × 3072(论文显式 R^{B×80×3072}) |
| Chunk | 13 帧未来视频;action H=12;去噪 5 步(3步80%→5步85%→50步87%,Table 9) |
| 上下文 | 当前 1 帧 + 13 帧未来;无长历史,靠预测 horizon 提供"前瞻" |
| 动作 | 连续相对位姿:单臂 7-DoF / 双臂 14-DoF;H=12,生成 48Hz |
| 训练 | 8×H20;两阶段:stage1 世界模型全参 flow matching finetune,stage2 冻结世界模型训 expert+gate;λ_react=0.1, λ_gate=0.05;门控离线 96.95% |

→ 详见 **Architecture** tab。

## 关键结果

| 指标 | 值 | 最强 baseline | setup |
|---|---|---|---|
| ID 平均成功率 | 89% | 78% (Cosmos-Policy) / 74% (π0.5) | 6 任务 × 20 episodes |
| OOD 平均成功率 | 82% | 53% (Wan+AnyPos) / 49% (π0.5) | 3 档 OOD × 6 任务 |
| OOD 相对最强提升 | +33% vs VLA / +29% vs WAM | — | abstract 声称 |
| ID 单臂 stage-wise | 91% | 81% (Cosmos-Policy) | 4 单臂 stage-wise |
| ID 双臂 stage-wise | 85% | 73% (Cosmos-Policy) | 2 双臂 stage-wise |
| Unseen Position OOD | 80% | 32% (π0.5) / 26% (Cosmos) | 空间不相交区域(最难档) |
| Action 生成频率 | 48Hz (H=12) | — | 世界模型 5 步去噪 |
| 门控准确率 | 96.95% | — | 1637 测试帧对 offline |
| 去噪步数消融 | 85% @5步 | 80% @3步 / 87% @50步 | Put Flowers,频率 5步4Hz/50步3Hz |

## Insights

- WAM 两条旧路线是结构性 trade-off(Table 1),不是工程优劣。统一需分工不是折中——隐含 latent 管精度、显式未来管泛化。
- 门控必须阶段感知硬路由,不能平均(Figure 5b):Averaging 在 position OOD 掉 46%,平均让 expert 互相干扰。
- 世界模型隐含 latent 比显式视频更关键(Figure 5c):去掉 latent,predictive ID 从 95% 掉到 62%,reactive 掉到 65%。
- OOD 瓶颈是 transit 不是 interaction(Table 1):Joint Modeling OOD transit 掉 32% 但 interaction(初始化到目标)能 95%。
- 5 步去噪是 quality-speed 甜点(Table 9):3步80%→5步85%→50步87%,边际递减明显,视频质量"够用"即可。

## vs 同类工作

- **vs DreamZero**(单 expert joint WAM):DreamZero 单一 AR DiT 联合 video+action,靠对齐拿精度;HarmoWAM 拆双 expert+门控,承认单 expert 难同时拿泛化和精度。DreamZero 强在 6.6s KV-cache 长上下文,HarmoWAM 强在阶段感知分工。
- **vs Cosmos-Policy**(joint latent WAM):Cosmos 把 action 编码成 latent 帧和视频一起去噪,OOD transit 掉到 26%。HarmoWAM reactive expert 借世界模型显式预测跳出 SFT 分布,是针对 Cosmos 弱点的直接补丁。
- **vs Wan+AnyPos**(imagine-then-execute):Wan2.2 生成视频+AnyPos 做 IDM,interaction OOD 掉 55%。HarmoWAM predictive expert 借隐含 latent 而非纯像素,拿到时序相干精度。
- **vs VPP**(video feature condition):单 expert Joint Modeling 变体,单臂 HarmoWAM 89% vs VPP 73%,双臂 VPP 不可比。

## 局限

- 论文自承:世界模型固定 13 帧 horizon,下游任务必须保同样 horizon 才能对齐(p25)。
- 论文自承:pixel 级未来生成有冗余开销,未来探索 latent 级预测表征(p25)。
- 论文自承:三类失败(Tilted stacking / Insertion misalignment / Zipper grasp slip)是精度/硬件极限,第三人称视角对前后位移不敏感、UMI gripper 摩擦力不足(p23-24)。
- 我们读出:门控硬二选一(s_t>0.5)无平滑过渡,阶段边界附近可能抖动。
- 我们读出:世界模型只用一路视角,多视角融合在特征层而非世界模型层,可能丢空间信息。
- 我们读出:baseline 预训练基础不完全对齐(AnyPos from scratch,其它用预训练 checkpoint),公平性有细微偏差。
- 我们读出:96.95% 门控准确率是离线评测,OOD 观测退化时门控可能误判阶段,论文未给 OOD 下门控准确率。

## 可复现性

| 字段 | 值 |
|---|---|
| code | https://elbb-yu.github.io/HarmoWAM/(project page) |
| weights | 未明确开源权重 |
| hardware | Franka FR3 单/双臂 + UMI gripper + 3× Intel RealSense |
| training | 8× NVIDIA H20 GPU |
| real_eval | 6 真实任务 × 20 episodes,3 档 OOD |
