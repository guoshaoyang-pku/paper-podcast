# BagelVLA — 结构化卡片

> 论文：BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation（清华 / ByteDance Seed，arXiv 2602.09849，2026-02-12）
> 配套：`card.json`（真相源）｜`architecture.md`｜`figures.md`｜`podcast.md`

## 一句话定位

在 Bagel 统一理解+生成 MoT 底座上接一个 2B action expert，把"语言规划→关键帧预测→动作生成"三步显式交错进一条序列，并用 Residual Flow Guidance（以当前帧为初始噪声做单步去噪）把视觉前瞻的延迟从 6 秒压到 1.23 秒。

## 基本信息

| 项 | 值 |
|---|---|
| slug | bagel-vla |
| arxiv | 2602.09849 |
| 日期 | 2026-02-12 |
| 机构 | 清华 · ByteDance Seed |
| 类别 | Cascaded WAM |
| 作者 | Yucheng Hu, Jianke Zhang, Yuanfei Luo 等 12 人 |

## 问题

**要解决什么**：长程操作任务（如按指定顺序叠积木、算术式摆积木）里，全局指令隐含一串子阶段。现有 VLA 把它当黑盒映射，要么只做语言规划缺视觉前瞻，要么只做视觉预测缺逻辑推理。BagelVLA 要让模型显式交错地"想下一步文字→想象下一个关键帧→出动作"。

**为什么 prior work 不够**：VPP/Cosmos Policy 这类用视频预测做 policy 的方法缺专门 VLM backbone，指令跟随差；RT-2/OpenVLA 这类纯 VLA 没视觉前瞻。统一理解生成模型（Bagel/Chameleon/Show-o）有"边想边画"能力但不是为具身控制设计的，且把视觉生成塞进 action 循环会带来 6 秒级延迟（Complete Denoise）。F1 拼了 VLA+IDM 但没做显式交错规划和视觉生成。

## 输入 / 输出

### 输入

| 名称 | 类型 | 说明 |
|---|---|---|
| multi-view observations v_t | image | Calvin 2 视角预测第 3；RoboTwin/真机 3 视角（主+左右腕）；SigLIP2 编码理解特征 + FLUX VAE 编码生成特征 |
| global instruction L | text | 全局指令，如"按红→黄→蓝→绿顺序叠积木" |
| proprioception | vector | 本体感受；Calvin 不用，RoboTwin/真机用；进 action expert |

### 输出

| 名称 | 类型 | 说明 |
|---|---|---|
| subtask l_t | text | 当前子任务文字规划（如"下一个抓红色积木"），理解专家自回归生成 |
| future keyframe v_{t+k} | image | 预测的下一个关键帧图像，生成专家 flow matching 去噪 |
| action chunk a_t | continuous | Calvin chunk=10；RoboTwin chunk=16（effective horizon 48）；真机 chunk=24；双臂 14-DOF |

**控制频率**：推理 1.23s/chunk（单 RTX 5090）；真机 action 频率 40Hz（chunk=48）；异步执行下 72Hz；action expert 2B 单独激活可 KV-cache。

### 输入拼接 protocol

```
<bos_obs>[ SigLIP2(v_t), VAE(v_t) ]<eos_obs>
<bos_lang>[ L ]<eos_lang>
<bos_plan>[ l_t (自回归) ]<eos_plan>
<bos_keyframe>[ noisy v^τ_{t+k} (FLUX VAE latent) ]<eos_keyframe>
<bos_action>[ noisy a^τ_t (action chunk) ]<eos_action>
```

- `<bos_obs>...<eos_obs>`：多视角观测，同时经两个 encoder：SigLIP2 出理解特征给 LLM expert，FLUX VAE 出生成特征给 generation/action expert。多视角是序列拼接，不是通道拼接。
- `<bos_lang>...<eos_lang>`：全局指令 L，作 LLM expert 的自回归上下文。
- `<bos_plan>...<eos_plan>`：子任务规划 l_t，由 LLM expert 自回归生成（CE loss）。推理时先出这段文字，再出关键帧和动作。
- `<bos_keyframe>...<eos_keyframe>`：关键帧 noisy latent，经 FLUX VAE，generation expert 用 flow matching 去噪。timestep 服从 LogitNormal(0,1)。RFG 模式下初始噪声是 N(v_t, I)（以当前帧为中心的残差噪声）。
- `<bos_action>...<eos_action>`：动作 chunk noisy latent，action expert 用 flow matching 去噪，timestep 服从 Beta(1.5,1)（偏向低噪声，加快收敛）。

**三个坑**：
1. language 和 plan 都是序列拼接进自回归（这点和 DreamZero/Motus 的 cross-attention 注入不同）——BagelVLA 把文本规划当成可生成的 token，显式写在序列里。视觉条件通过 MoT 共享自注意力注入，不是 token 拼接。
2. 多视角在序列层拼接（每个视角一套 SigLIP2+VAE token），不是通道拼接。Calvin 预测第 3 视角，RoboTwin/真机预测主视角。
3. 训练和推理序列结构一致，但 dual flow-matching 交互方式有三种（Complete/Joint/Single-step），默认 Single-step RFG：推理时 action expert 只 attend 关键帧第 1 步去噪的 KV-cache，不跑完整 N1 步图像去噪。

## 数据集

| 数据 | 规模 | 备注 |
|---|---|---|
| General VQA（LLaVA+FineVision） | 2.56M QA pairs | Stage1 语言共训练，保通用语言能力 |
| EgoDex（人类手部视频） | 310k episodes | Stage1 视觉动态，只预测最终帧 |
| Open-source Robot（Bridge/Galaxea/RoboTwin/Agibot/GR） | ~640k episodes | Seed-1.5-VL-thinking 合成子任务标签 |
| Self-collected Aloha 真机 | 4.5k 预训 + 3k basic + 1.5k long-horizon | Stage2 finetune，人工标注子任务和关键帧 |
| Calvin ABC-D | ABC split | 仿真评测，1000 任务长度 5 |
| RoboTwin 2.0 | 2.5k episodes（50×50） | Clean+Randomized，unseen instructions |

## 架构（摘要）

| 项 | 值 |
|---|---|
| backbone | Bagel（Qwen2.5-LLM-7B 统一理解+生成 MoT，7B active/14B total）+ Action Expert 2B |
| 总参数 | ~16B（Und 7B + Gen 7B + Action 2B，共享自注意力）|
| 类型 | MoT 三专家 + 双 Flow Matching + Interleaved Planning |

**关键组件**：
- Understanding Expert（7B, hidden 3584, 28 layers）：自回归出子任务文字 l_t，CE loss
- Generation Expert（7B, hidden 3584, 28 layers）：flow matching 去噪关键帧，timestep LogitNormal(0,1)
- Action Expert（2B, hidden 3584, 28 layers, MLP intermediate 3584=1/5 of 18944）：flow matching 去噪动作 chunk，timestep Beta(1.5,1)
- Tri-model Joint Attention（三专家共享多头自注意力，MoT 耦合）
- Dual Flow Matching 交互（Complete/Joint/Single-step，默认 Single-step RFG）
- Residual Flow Guidance（RFG）：关键帧初始噪声用 N(v_t, I) 而非 N(0, I)
- Asynchronous Execution：训练随机用前一帧替换，推理只更新 proprio KV，40Hz→72Hz

**为什么这样设计**：用 Bagel 统一底座继承互联网级多模态推理和生成先验（理解专家保语言推理，生成专家保图像生成），这是 Cosmos Policy 纯视频模型给不了的。交错规划是把长程任务全局指令显式分解成因果链。RFG 解决"视觉生成塞进 action 循环"的 6 秒延迟——不从头生成关键帧，而是以当前帧为残差起点做单步去噪。Action expert 用 2B 而非 7B 是为了高频跑（40Hz）并支持 KV-cache 异步。

→ 详见 `architecture.md`。

### 数值 sense

| 项 | 值 |
|---|---|
| DiT 规格 | 三专家均 hidden=3584, 28 layers（同 Qwen2.5-LLM-7B）；Und/Gen intermediate=18944，Action intermediate=3584（缩到 1/5）。总 ~16B |
| 分辨率 | 图像 256×256（VAE 输入）；Calvin 2 视角预测第 3，RoboTwin/真机 3 视角（主+左右腕）预测主视角 |
| VAE | FLUX VAE 编码图像（256×256→latent）；SigLIP2 作理解侧 visual encoder。两套独立 visual encoder |
| 每帧 latent 维 | FLUX VAE 典型空间 8× 下采样，256×256→32×32，channel 16 → 每帧 ~1.6e4 维 |
| Chunk | Calvin chunk=10；RoboTwin chunk=16（effective horizon 48）；真机 chunk=24。真机 40Hz@chunk=48 → 1.2s/chunk；异步 72Hz |
| 上下文 | 当前帧 v_t + 全局指令 L + 历史子任务（隐式在自回归里）；不显式维护长历史 KV |
| 动作 | 双臂 14-DOF；Calvin 不用 proprio，RoboTwin/真机用；动作 chunk 连续，flow matching 去噪 |
| 训练 | Stage1：64×A800, batch~1600, 20k steps, lr 1e-5, FSDP, 只训 Und+Gen。Stage2：Calvin 8×A800 batch192 30k steps；RoboTwin 8×A800 batch128 60k steps；真机 32×A800 batch512 50k steps。image FM 50 步，action FM 10 步；Single-step 1 步图像+10 步动作 |

## 关键结果

| 指标 | 值 | 最强 baseline | setup |
|---|---|---|---|
| Calvin ABC-D mean completion | **4.405** | VPP 4.329 / UP-VLA 4.078 / π0 3.648 | 1000 任务长度 5，D-split |
| RoboTwin Clean 50-task avg | **75.26%** | w/o-keyframe 56.72% / UP-VLA 52.92% / π0 46.42% | 50 任务，unseen instructions |
| RoboTwin Randomized 50-task avg | **20.87%** | w/o-textual 19.20% / UP-VLA 15.16% / π0 16.34% | 强 randomization |
| 真机 Basic Task 9 类 avg | **75.5%** | π0 65.0% / VPP 59.5% | Aloha-AgileX 双臂，每任务 20 次 |
| 真机 Stack Cubes（长程）avg | **73.3%** | w/o-keyframe 53.3% / π0 40.0% / VPP 25.0% | Easy/Middle/Hard 三档 |
| 真机 Calculate Symbols（长程）avg | **63.3%** | w/o-keyframe 50.0% / π0 31.7% / VPP 23.3% | 需 CoT 推理 |
| 推理延迟（单 chunk） | **1.23s**（Single-step RFG） | Complete 6.04s / Joint 2.90s | 单 A800, Calvin single-view |
| 真机 action 频率 | **40Hz（同步）/ 72Hz（异步）** | — | 单 RTX 5090, chunk=48 |

## 关键技术

1. **Interleaved Planning**——文字→关键帧→动作显式因果链，每步显式监督。
2. **Dual Flow Matching + Single-step Denoise**——action 只看图像第 1 步 KV，1.23s 且最准（OOD 鲁棒）。
3. **Residual Flow Guidance（RFG）**——初始噪声 N(v_t, I) 而非 N(0, I)，聚焦残差变化，10 步出高质量关键帧。
4. **MoT 三专家 + Bagel 底座**——继承 Bagel 语言推理+图像生成先验，Action 2B 高频跑。
5. **Asynchronous Execution**——训练随机用前一帧替换，推理只更新 proprio KV，40Hz→72Hz。

## Insights

- Single-step denoise 不仅最快还最准——Complete/Joint 在 OOD 测试时图像 FM 中间态会进入 OOD 污染 action，Single-step 只取第 1 步 KV 反而更鲁棒。反直觉：少去噪比多去噪好。
- RFG 本质是把"生成未来帧"变成"生成相对当前帧的残差"，让模型聚焦动态区域而非重建静态背景。
- 长程任务规划准确率（近 90%）远高于任务成功率（73.3%），说明模型"想对了但手没跟上"——action mapping 精度是瓶颈。
- 语言规划在 RoboTwin 上贡献 +21%（54%→75%），这是显式交错规划最强证据。
- 预训练只训语言规划+视觉动态（不训 action），就能在 action finetune 后提升 OOD pick&place——通用多模态先验能跨模态迁移到控制。

## vs 同类工作

- **vs Cosmos Policy**：Cosmos 直接 fine-tune 大视频模型当 policy，缺专门 VLM backbone，指令跟随差；BagelVLA 用 Bagel 统一底座继承语言推理，能做算术 CoT 任务，且用交错规划显式分解长程。
- **vs VPP**：VPP 用视频预测做辅助但没显式语言规划；BagelVLA 把文字规划也塞进序列，长程 +21%。
- **vs DreamZero**：DreamZero joint 端到端 AR DiT 共享 timestep 实时闭环（7Hz）；BagelVLA cascaded（文字→关键帧→动作显式三步）MoT，RFG 把视觉前瞻压到 1.23s 但仍比 DreamZero 慢。DreamZero 重实时，BagelVLA 重显式规划。
- **vs Motus**：都用 MoT 三专家 + flow matching。但 Motus 用 UniDiffuser 调度切换五模式、用光流 latent action 预训练 action 专家；BagelVLA 用 interleaved planning 显式三步、用 RFG 解决视觉前瞻延迟。Motus 重统一性，BagelVLA 重长程规划。
- **vs F1**：F1 拼 VLA+IDM 但没做显式交错规划和视觉生成；BagelVLA 把三步全显式交错。

## 局限

- 论文自承：长程任务规划准确率（近 90%）与任务成功率（63-73%）有 gap，action mapping 精度是瓶颈，模型"想对了但手没跟上"。
- 论文自承：RoboTwin Randomized 场景绝对分仍低（20.87%），强 randomization 下所有方法都 struggle。
- RFG 的 N(v_t, I) 假设"未来帧和当前帧结构相似"，对快速运动或视角剧烈变化可能不成立；未给 RFG 失效场景分析。
- 异步执行（72Hz）隐含假设"子任务和关键帧不每 chunk 变"，对高动态任务会掉点，未量化异步精度损失。
- Single-step 只取图像第 1 步 KV，等于放弃完整关键帧信息；精细对位任务可能不够，未对比精细任务上 Single-step vs Complete。
- 真机只比 π0 和 VPP，没和 DreamZero/Motus/X-VLA 真机直接对照；长程优势部分来自 baseline 本身没显式规划。
- 三专家总 16B，单 RTX 5090 跑 1.23s/chunk 依赖异步和 KV-cache，部署门槛高于纯 VLA；Action expert 2B 容量是否够复杂任务未讨论。

## 可复现性

| 项 | 值 |
|---|---|
| code | https://cladernyjorn.github.io/BagelVLA.github.io |
| weights | Bagel-7B-MoT 公开底座；BagelVLA 自身权重开源情况未明确 |
| sim benchmark | Calvin ABC-D、RoboTwin 2.0（50 任务 Clean+Randomized） |
| real eval | Aloha-AgileX 14-DOF 双臂，9 类 basic + 2 类 long-horizon，每任务 20 次 |

## 标签

`interleaved planning` `Mixture-of-Transformers` `dual flow matching` `Residual Flow Guidance` `long-horizon manipulation` `Bagel` `keyframe prediction` `Tsinghua` `ByteDance`
