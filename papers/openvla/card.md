# OpenVLA — 开源视觉-语言-动作模型

> 配套 `card.json`(真相源)。下表与卡片内容一一对应。

## 元信息

| 字段 | 值 |
|---|---|
| slug | openvla |
| title | OpenVLA: An Open-Source Vision-Language-Action Model |
| authors | Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, et al. |
| affiliation | Stanford · UC Berkeley · TRI · Google DeepMind · Physical Intelligence · MIT |
| arxiv | 2406.09246 |
| date | 2024-09-05 |
| venue | arXiv preprint (Stanford/Berkeley/TRI) |
| category | VLA |

**一句话**:把 Prismatic-7B VLM(DINOv2+SigLIP 融合视觉编码器 + Llama 2 7B)在 970k Open X-Embodiment 机器人轨迹上 fine-tune,把 7 维连续动作离散成 256 个 token 塞进 LLM 词表用 next-token 预测训,一个 7B 开源 VLA 在 29 任务上超 55B 闭源 RT-2-X 16.5 个点,且支持 LoRA(1.4% 参数)和 4bit 量化在消费级 GPU 上 fine-tune 和推理。

**tags**:VLA · open-source · Llama 2 · DINOv2 · SigLIP · Prismatic · action token · Open X-Embodiment · LoRA · quantization · consumer GPU · Stanford · Berkeley

## 问题

- **要解决什么**:VLA(如 RT-2-X)展现通用机器人策略潜力,但两个硬伤:1)现有 VLA 全闭源,架构/训练/数据不公开;2)prior work 没研究如何高效 fine-tune VLA 到新机器人/任务,尤其消费级硬件。做一个完全开源(数据+权重+代码)的 VLA,并系统研究 fine-tune 和量化部署。
- **为什么 prior work 不够**:RT-2-X 55B 闭源且不支持 fine-tune(只给推理 API);RT-1/Octo 部分开源但 from-scratch 训练,stitch 预训练组件没充分利用 VLM Internet-scale 先验;prior VLA 都没研究 LoRA/量化。OpenVLA 直接 fine-tune 强开源 VLM(Prismatic-7B,融合 DINOv2 空间+SigLIP 语义)做 action token 预测,简单可扩展,把 LLM 生态 LoRA/量化搬过来。

## 输入 / 输出

### 输入

| 名称 | 类型 | 说明 |
|---|---|---|
| image observation | image | 单张 224×224 RGB(第三人称);无多视角无历史(局限) |
| language instruction | text | 模板 "What should the robot do to {task}? A:" |
| proprioceptive state | vector | 默认不用(视觉已含足够信息) |

### 输出

| 名称 | 类型 | 说明 |
|---|---|---|
| robot action | discrete→continuous | 7 维单步(Δx/y/z + Δroll/pitch/yaw + gripper);每维 256 bin(1%/99% 分位),映射 Llama 词表 256 token;de-tokenize 回连续;无 chunk |

**控制频率**:推理 ~6Hz(RTX 4090,bfloat16);4bit 量化 A5000 3Hz;训练 BridgeData 5Hz、DROID 15Hz;无 action chunk 每步都要推理。

### 输入拼接 protocol

```
<bos_img>[DINOv2(patch) ⊕ SigLIP(patch)] → MLP Projector → visual tokens<eos_img>
<bos_lang>What should the robot do to {task}? A:<eos_lang>
<bos_action>[a_1...a_7] (7 离散 action token, 每个∈[0,255])<eos_action>
```

**逐 token 解释**:
- `<bos_img>...<eos_img>`:224×224 图像分别过 DINOv2(低层空间)+SigLIP(高层语义),patch 特征 channel-wise concat,经 2 层 MLP 投到 Llama 词嵌入。融合视觉编码器是关键——比单 SigLIP/CLIP 在多物体 grounding 强约 10%。
- `<bos_lang>...<eos_lang>`:语言经 Llama tokenizer,模板 "What should the robot do to {task}? A:"。标准 VLM prompt 格式,动作预测被框成"回答问题"。
- `<bos_action>...<eos_action>`:7 个离散 action token,每维 256 bin。bin 宽用 1%/99% 分位定(抗 outlier)。256 token 覆写 Llama 词表最后 256 个最少用 token(因 Llama 只留 100 special token)。next-token CE loss 只在 action token 算。

**三个坑**:(1) language 和 action 都是 LLM 序列 token,统一自回归——这点和 π0.7/Qwen-VLA 不同(后者 action 是 DiT 连续张量);(2) 多视角不支持,单图输入(自承局限);(3) 训练和推理序列一致,但训练 GT teacher forcing,推理自回归生成 7 token 再 de-tokenize;无 action chunk 意味每步完整前向,6Hz 是单步延迟。

## 数据集

| 数据 | 规模 | 备注 |
|---|---|---|
| Open X-Embodiment(主) | 970k 轨迹 | 70+ 数据集筛选:单臂末端控制+至少 1 第三人称相机;Octo mixture weight 平衡;DROID 试 10% 但 accuracy 低后移除 |
| Prismatic-7B 预训练(VLM 阶段) | LLaVA 1.5 ~1M | image-text+text-only 开源;DINOv2/SigLIP/Llama 2 各自 Internet 预训练(闭源) |

## 架构(摘要)

| 字段 | 值 |
|---|---|
| backbone | Prismatic-7B VLM = DINOv2+SigLIP 融合视觉编码器(600M)+ 2 层 MLP projector + Llama 2 7B |
| params | 7B(Llama 2 7B 主导);视觉 600M;projector 小 |
| type | VLA:VLM fine-tune + 离散 action token next-token 预测(无 action chunk,无 DiT,无 flow matching) |

**关键组件**:
- 视觉编码器(600M,DINOv2+SigLIP):224×224 分别过 DINOv2(patch 14 空间)+SigLIP(patch 14 语义),concat
- MLP Projector(2 层):融合视觉特征→Llama 词嵌入空间
- Llama 2 7B LLM:处理 visual+语言 token,自回归预测 7 action token
- Action discretization:7 维每维 256 bin,1%/99% 分位,覆写词表最后 256 token
- Action de-tokenizer:7 离散 token→连续 7 维
- 无 action expert/DiT/flow matching:action 走 LLM token 通道

**为什么这样设计**:三个理由选 VLM fine-tune+action token:(1)Internet-scale VL 对齐视觉语言继承 web 先验;(2)通用 VLM 架构借 LLM 训练基础设施和生态(LoRA/量化/HuggingFace),最小代码 scale 到 billion;(3)直接受益于 VLM 快速改进。选 Prismatic 因 DINOv2+SigLIP 融合在多物体 grounding 比单 SigLIP/CLIP 强约 10%,代码模块化。离散 action token 而非连续 head 因简单——直接 next-token CE 无需额外组件。这是"最简单可扩展 VLA"哲学。

### 数值 sense

| 项 | 值 |
|---|---|
| DiT VLM | Llama 2 7B:32 layers × 32 heads,hidden 4096,ffn 11008;标准 transformer decoder |
| DiT action | 无独立 action expert;action 走 LLM token 通道,7 token/步 |
| vision encoder | DINOv2-base(patch 14,embed 768)+SigLIP(patch 14,embed 1152)→concat→MLP 投 4096;共 600M |
| 分辨率 | 224×224(试 384 无性能差但慢 3×,故选 224) |
| VAE | 无 VAE;视觉编码器直接出 patch token |
| 每帧 latent 维 | 224²/14²→~256 patch token/视角(DINOv2+SigLIP 各一);concat 后 MLP 投 4096 |
| Chunk | 无 action chunk;单步 7 token 预测;无 receding horizon(自承局限,Diffusion Policy 用 T=16/X=8) |
| 上下文 | 单图+单语言;无历史帧无 episodic memory(未来工作) |
| 动作 | 7 维单步:Δx/y/z+Δroll/pitch/yaw+gripper;每维 256 bin,1%/99% 分位;WidowX/Google Robot/Franka 7-DoF 单臂末端控制 |
| 训练 | 64×A100×14 天=21,500 A100-hours;batch 2048;27 epochs;lr 2e-5 固定无 warmup;action token accuracy 95%+;fine-tune vision encoder 至关重要(冻结掉性能);FSDP+AMP+FlashAttention |

→ 详见 **Architecture** tab。

## 关键结果

| 指标 | 值 | 最强 baseline | setup |
|---|---|---|---|
| BridgeData V2 WidowX 17 任务 | 70.6% | 50.6%(RT-2-X 55B)/20.0%(Octo)/18.5%(RT-1-X) | 170 rollouts,5 类泛化+grounding |
| Google Robot 12 任务 in-distribution | 85.0% | 88.0%(RT-2-X)/33.3%(RT-1-X)/44.0%(Octo) | 60 rollouts |
| Google Robot OOD | 78.3% | 72.0%(RT-2-X)/26.7%(RT-1-X)/32.0%(Octo) | 未见物体/任务/背景/概念 |
| 29 任务总平均超 RT-2-X | +16.5% absolute | 55B RT-2-X(闭源) | WidowX+Google Robot,7B vs 55B |
| Franka fine-tune 7 任务 aggregate | OpenVLA 最高,所有 ≥50% | Diffusion Policy(窄强)/Octo | 10-150 demonstrations |
| OpenVLA vs Diffusion Policy(多物体 grounding) | +20.4% | Diffusion Policy(from scratch) | Franka 多样多指令任务 |
| LoRA fine-tune(rank=32) | 68.2% | 69.7%(full FT,统计持平) | Franka-Tabletop,1.4% 参数,单 A100 10-15h |
| 4bit 量化推理 | 71.9% / 7.0GB | 71.3% / 16.8GB(bfloat16) | BridgeData V2 8 任务,80 rollouts |
| 推理频率 | ~6Hz(RTX 4090,bfloat16) | — | 无编译/推测解码 |
| 训练算力 | 21,500 A100-hours(64×A100×14 天) | — | batch 2048,27 epochs,970k 轨迹 |

## Insights

- 简单设计+更多数据胜过堆参数:7B OpenVLA 在 29 任务超 55B RT-2-X 16.5 个点,因 970k vs 350k 数据 + DINOv2/SigLIP 融合 + 更仔细数据清洗。VLA 不是越大越好。
- VLA 训练反 LLM 直觉:27 epoch(LLM/VLM 通常 1-2)、fine-tune vision encoder(VLM 习惯冻结)、lr 2e-5 无 warmup。给后续 VLA 基线参照。
- 离散 action token 路线简单可扩展:action 走 LLM token 通道用 next-token CE,无需 DiT/flow matching,直接借 LLM 生态。代价是无 action chunk 每步 6Hz。
- DINOv2 空间 + SigLIP 语义融合是多物体 grounding 关键:比单 SigLIP/CLIP 强约 10%。
- LoRA+4bit 量化让 VLA 民主化:1.4% 参数匹配 full FT,7GB 显存性能不掉。VLA 从服务器走向消费级 GPU。
- OpenVLA 是 downstream fine-tune 强默认:唯一所有 Franka 任务 ≥50%,尤其多样语言指令;窄高灵巧任务 Diffusion Policy 的 action chunk 仍优。

## vs 同类工作

- **vs RT-2-X**(闭源 55B):OpenVLA 7B 超 16.5 个点,7× 少参数,且开源支持 fine-tune。RT-2-X 在 semantic 泛化仍领先(Internet 预训练更大+co-fine-tune)。
- **vs Octo**(开源 93M from-scratch):OpenVLA 用 VLM fine-tune 继承 Internet 先验,BridgeData V2 70.6% vs Octo 20.0%——VLM 先验碾压。
- **vs Diffusion Policy**(from-scratch IL):DP 窄单指令任务强(action chunk+temporal smoothing),OpenVLA 多样多指令强(语言 grounding)。OpenVLA 更通用默认,DP 适合窄高灵巧。
- **vs π0.7/Qwen-VLA**(后续 VLA):后者用 flow-matching action expert(连续)+多维 context。OpenVLA 走离散 action token 简单路线,无 action expert 解耦——更简单但牺牲 action chunk 和多模态动作分布。后续工作多继承 OpenVLA 的 VLM fine-tune 框架但换连续 action expert。
- **vs RT-2**(原 VLA):RT-2 用 PaLI 55B 闭源,OpenVLA 用 Prismatic-7B 开源+融合视觉编码器。OpenVLA 把 RT-2 的"VLM fine-tune 做 action token"路线开源化并加 fine-tune/量化研究。

## 局限

- 论文自承:只支持单图观测,无多视角/proprio/历史帧,异质传感器输入未支持(p11)。
- 论文自承:推理吞吐不足以支撑高频控制(如 ALOHA 50Hz),action chunk 或推测解码是潜在解(p11)。
- 论文自承:可靠性仍未达高(典型 <90% 成功率),性能有提升空间(p11)。
- 论文自承:因算力限制,许多 VLA 设计问题未充分探索——base VLM 规模影响、VL+action co-training、最佳视觉特征(p11)。
- 我们读出:无 action chunk 每步完整前向,6Hz 在高频任务受限;Diffusion Policy 用 chunk 在窄灵巧任务更平滑,OpenVLA 未集成。
- 我们读出:离散 action token(256 bin)有量化损失,且无 action 多模态建模(同一状态多种合理动作被平均);后续 π0/Qwen-VLA 用 flow matching 正是补此。
- 我们读出:970k OpenX 虽多样但仍是机器人示教,没像 π0.7 混 autonomous/失败/人类/web 数据,数据多样性有上限。
- 我们读出:DROID 试 10% weight 但 accuracy 低被移除——暗示 OpenVLA 对高多样性大数据集拟合不足,可能需更大模型或更高 weight。

## 可复现性

| 字段 | 值 |
|---|---|
| code | https://openvla.github.io + https://github.com/openvla/openvla |
| weights | HuggingFace 开源 |
| data | 970k Open X-Embodiment 子集(mixture weight 开源) |
| training | 64×A100×14 天=21,500 A100-hours;PyTorch+FSDP+AMP+FlashAttention;支持 LoRA/量化 |
| fine_tune | LoRA 单 A100 10-15h;full FT 8×A100 5-15h |
| inference | bfloat16 15GB(~6Hz RTX 4090);int4 7GB;remote inference server 开源 |
| real_eval | WidowX(BridgeData V2)+ Google Robot + Franka-Tabletop + Franka-DROID |
