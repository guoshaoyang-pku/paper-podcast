# π0.7 — 可引导的通用机器人基础模型

> 配套 `card.json`(真相源)。下表与卡片内容一一对应。

## 元信息

| 字段 | 值 |
|---|---|
| slug | pi-series |
| title | π0.7: A Steerable Generalist Robotic Foundation Model with Emergent Capabilities |
| authors | Physical Intelligence, Bo Ai, Ali Amin, et al. |
| affiliation | Physical Intelligence |
| arxiv | 2604.15483 |
| date | 2026-04-24 |
| venue | arXiv preprint (Physical Intelligence) |
| category | VLA |

**一句话**:一个 5B VLA(4B Gemma3 VLM + 860M flow-matching action expert),靠"把每条轨迹的 episode metadata(速度/质量/是否犯错)+ subtask 指令 + 世界模型生成的 subgoal 图像"塞进 prompt 做条件,从而能把混合质量(含失败、含 RL specialist 蒸馏)的异质数据训成一个会组合泛化的通用 policy,零样本迁移到新任务/新本体/逆数据偏差指令。

**tags**:VLA · flow matching · context conditioning · episode metadata · subgoal image · BAGEL · Gemma3 · cross-embodiment · compositional generalization · Physical Intelligence · coaching

## 问题

- **要解决什么**:现有 VLA 用一条语言指令做条件,只能用高质量示教。一旦混入失败、低质量、跨策略、跨本体、人类视频、web 多模态这些异质数据,naive 训练会把不同模式平均掉产出次优行为。如何让通用 VLA 真正利用异质混合数据,同时组合泛化到没见过的任务和本体。
- **为什么 prior work 不够**:经典 VLA 只吃语言条件数据多样性受限;跨本体/人类视频工作偏表征学习没解决"同任务多策略"歧义;subgoal conditioning 工作没系统化成训练多维 context。核心矛盾:数据要多样,但多样数据若不加标注会训坏——π0.7(no metadata)在更大混合质量数据上反而变差(Fig.18 left)。

## 输入 / 输出

### 输入

| 名称 | 类型 | 说明 |
|---|---|---|
| multi-view observation | image | 最多 4 路(前视+2 腕部+可选后视),每路最多 6 历史帧(stride 1s),448×448,经 MEM vision encoder 时空压缩 |
| proprioceptive state | vector | 关节状态(含历史),linear projection 映射到 backbone dim |
| task instruction ℓ | text | 总任务描述,如 "clean the kitchen" |
| subtask instruction ℓ̂ | text | 下一语义子任务,来自 high-level policy 或人类 coaching |
| subgoal images g | image | 多视角近未来期望图(最多 3 路),BAGEL 14B 世界模型生成 |
| episode metadata m | structured | speed(每 500 步分桶)/quality(1-5)/mistake(布尔) |
| control mode c | text token | joint 或 ee |

### 输出

| 名称 | 类型 | 说明 |
|---|---|---|
| action chunk | continuous | 50 步 action token,860M expert flow matching 去噪(5 步),执行 Ĥ∈{15,25} |

**控制频率**:UR5e 20Hz,其它 50Hz;50 步/chunk,5 步去噪;最小变体 38ms(3 视角),最坏 127ms(+MEM+subgoal);世界模型 subgoal 1.25s/张,异步执行。

### 输入拼接 protocol

```
<bos_obs>[I1_t...I4_t (448×448, MEM ViT)]...[历史帧 stride 1s]<eos_obs>
<bos_subgoal>[G1_t...G3_t (生成或GT, 448×448)]<eos_subgoal>
<bos_lang>Task: ℓ. Subtask: ℓ̂. Speed: s. Quality: q. Mistake: b. Control Mode: c.<eos_lang>
<bos_proprio>[q_t, q_{t-T}...(linear proj)]<eos_proprio>
<bos_action>[noisy a_{t:t+50}^k, 50 tokens, 860M expert flow matching]<eos_action>
```

**逐 token 解释**:
- `<bos_obs>...<eos_obs>`:多视角+历史。block-causal:观测内部 bidirectional,subgoal 可 attend 观测。最多 4×6 帧,stride 1s,历史 30% drop、后视 30% drop。MEM 压成单帧 token 数。
- `<bos_subgoal>...<eos_subgoal>`:subgoal 训练时 25% batch 加(action 加了变 inverse dynamics 太简单);30% 同时 drop subtask 文本。来源:25% segment 末帧、75% 0-4s 前未来真实帧、加 BAGEL 生成假图。
- `<bos_lang>...<eos_lang>`:文本走 Gemma3 文本通道,causal。metadata 15% 全 drop,单字段 5% drop。control mode 不 drop。这是 π0.7 核心"context 丰富化"。
- `<bos_proprio>...<eos_proprio>`:proprio linear projection(不离散化),每历史帧一 token;历史帧 drop 时对应 state token 也 mask。
- `<bos_action>...<eos_action>`:860M action expert 50 token,flow matching 5 步,adaptive RMSNorm 注 timestep。内部 bidirectional + attend VLM 全 activation(knowledge insulation:梯度不回传 VLM,VLM 只被 FAST token CE 训)。

**三个坑**:(1) language/subtask 是 causal 序列 token,subgoal/观测是 block-causal bidirectional,不是统一自回归;(2) 多视角在观测 block,subgoal 也是多视角(最多 3),两者进同一 vision encoder,但 subgoal 可 attend 观测反之不行;(3) 训练每组件随机 drop 推理可任意组合;CFG 可对任意 prompt 部分做(主对 metadata,β∈{1.3,1.7,2.2})。

## 数据集

| 数据 | 规模 | 备注 |
|---|---|---|
| Demonstration data | 大规模(未给具体小时数) | 多任务多机器人(静态/移动、单臂/双臂)、多样环境 |
| Autonomous data | 大规模 | 策略评测收集,含 π0.6 RL rollout(specialist 蒸馏来源);排除 generalization eval 任务防泄漏 |
| Human interventions | — | policy rollout 中人工干预段 |
| Open-source robot datasets | — | 公开跨本体数据 |
| Egocentric human video | — | 人类第一视角视频 |
| Web multimodal | — | object localization/attribute prediction/VQA/text-only/video captioning |
| Subgoal 训练子集 Dg | 子集 | 高质量 subtask 标注 segment,末帧作 GT subgoal |

## 架构(摘要)

| 字段 | 值 |
|---|---|
| backbone | Gemma3 4B VLM(初始化)+ MEM vision encoder(400M)+ 860M flow-matching action expert + BAGEL 14B 世界模型 |
| params | 约 5B(4B VLM + 400M vision + 860M action);世界模型 14B 独立推理 |
| type | VLA(flow-matching action expert + knowledge insulation)+ multimodal context conditioning + 轻量世界模型 subgoal 生成 |

**关键组件**:
- VLM backbone(Gemma3 4B):处理多视角观测+历史+文本+subgoal+proprio;FAST token 离散 CE 训(knowledge insulation)
- MEM vision history encoder(400M):历史帧时空压缩,输出固定 token 数
- Action expert(860M):flow matching,50 action token,adaptive RMSNorm 注 timestep;bidirectional 内部 + attend VLM;梯度不回传 VLM
- World model(BAGEL 14B MoT):SuSIE 风格,初始化自 web-scale 图像生成/编辑;flow matching 生成多视角 subgoal;3 路 CFG(±text × ±img)
- High-level language policy:同 Gemma3 4B,把 coaching 数据训成自动产 subtask 指令的策略
- Block-causal attention:观测/subgoal bidirectional,subgoal 可 attend 观测;文本 causal;action bidirectional + attend backbone

**为什么这样设计**:核心论点"naive 用异质混合数据会平均掉不同模式产出次优行为"。解法不是改架构而是改 prompt——把 episode metadata + subtask + subgoal 塞进 context 让模型区分"高质量快速"vs"失败演示",从而混合质量数据变可学习且推理时 metadata prompting 选模式。knowledge insulation 让 VLM 稳定 CE 训、action expert flow matching 训解耦。subgoal 让 action 预测退化成 inverse dynamics 加速训练(故只 25% batch 加)。杠杆点:context 丰富化使数据 scaling 从"越大数据越差"变"越大数据越好"(Fig.18)。

### 数值 sense

| 项 | 值 |
|---|---|
| DiT VLM | Gemma3 4B(初始化);FAST token 离散监督 |
| DiT action | 860M;50 action token;adaptive RMSNorm 注 timestep;5 步 flow matching |
| vision encoder | MEM ~400M;448×448;时空压缩输出固定 token 数 |
| 分辨率 | 观测/subgoal 448×448;世界模型 ViT 448×336、VAE 512×384(patch 14 vs 16) |
| VAE | BAGEL 世界模型 VAE(patch 16);VLM 观测用 ViT(patch 14) |
| 每帧 latent 维 | 448×448/patch14² → ~1024 patch token/view(估算);6 历史帧×4 视角经 MEM 压到单帧 token 数 |
| Chunk | action H=50 token;执行 Ĥ∈{15,25};5 步去噪;训练 RTC 模拟 0-12 步延迟(最大 240ms @50Hz) |
| 上下文 | 最多 4 视角×6 历史帧(stride 1s)+ 3 subgoal;block-causal;历史 30% drop、后视 30% drop、subgoal 25% batch、subtask 30% drop、metadata 15% 全+单字段 5% drop |
| 动作 | joint 或 ee;UR5e 20Hz、其它 50Hz;双臂移动 2×6DoF+1夹爪+1-2升降+3全向底盘;BiPi 2×6DoF+1夹爪;UR5e 2×6DoF+Robotiq |
| 训练 | 从 Gemma3 4B 初始化;knowledge insulation;CFG on metadata β∈{1.3,1.7,2.2};推理最小 38ms/最坏 127ms;世界模型 subgoal 1.25s/张(25 步,4×H100 tensor parallel+8bit+SageAttention) |

→ 详见 **Architecture** tab。

## 关键结果

| 指标 | 值 | 最强 baseline | setup |
|---|---|---|---|
| Out-of-the-box 灵巧任务 | 匹配 π*0.6 specialist,laundry/box throughput 超 | π*0.6 RL specialist | Laundry/Espresso/Box + 6 灵巧任务 |
| 未见环境指令跟随 | 显著超 π0.5/π0.6,高绝对成功率 | π0.5/π0.6 | 14 场景 × 3-6 步,4 未见厨房+2 未见卧室 |
| 复杂指代指令 | π0.7(GC)显著领先 complex | π0.5/π0.6/π0.7(无 GC) | Office Desk 重排 |
| 打破数据偏差(Reverse Fridge) | π0.7(GC)成功,无 GC 失败 | π0.5/π0.6(均失败) | 反向任务,subgoal 是关键 |
| 跨本体叠衣(UR5e 零样本) | task progress 85.6%/success 80% | 人类 top2% 90.9%/80.6%(首次) | UR5e 双臂,从未训练叠衣 |
| 跨本体重排(最大 gap) | π0.7 显著超 prior | π0.5(崩)/π0.6 | 静态双臂数据→单臂 UR5e |
| Scaling with metadata | π0.7(metadata)数据越大越好(即使质量降) | π0.7(no metadata)越大越差 | 4 桶 top30/50/80/all |
| 任务多样性消融 | 去最高多样性 20% 显著掉 | 去随机 20%(不掉) | 未见短任务 |
| 推理延迟 | 最小 38ms(3 视角)/最坏 127ms | — | 单 H100,5 步去噪+RTC |
| 世界模型 subgoal 延迟 | 1.25s/张(25 步) | — | 4×H100+8bit+SageAttention,异步 |

## Insights

- naive 用异质混合数据会平均掉不同模式产出次优行为——π0.7(no metadata)越大越差,π0.7(metadata)越大越好(Fig.18 left)。context 丰富化使数据 scaling 成立。
- 数据多样性 > 数据质量:去最高多样性 20% 在未见短任务显著掉,去随机 20% 不掉(Fig.18 right)。组合泛化驱动力是任务多样性。
- subgoal image 把世界模型语义/物理先验注入 policy:复杂指代、打破偏差、跨本体都提升,Reverse Fridge GC 是成功关键(Fig.10/11/12)。
- 跨本体涌现策略适应:π0.7 在 UR5e 发现合适策略(单臂 pick-and-place 代替双臂扶袋、垂直抓取代替倾斜)——真组合泛化不是模仿(Fig.13)。
- 通用模型可蒸馏 specialist:π0.7 靠 autonomous eval data(含 RL rollout)+ metadata 区分质量,out-of-the-box 匹配 RL specialist,throughput 甚至超(Fig.6/7)。metadata 让蒸馏不污染通用性。
- 语言 coaching 是新任务数据高效路径:π0.7 可被人分步指令教新长程任务,coaching 数据训 high-level policy 实现全自主(Fig.14/16)。数据获取从示教降到语言。

## vs 同类工作

- **vs π0.5/π0.6**(自家前作):只吃语言条件只能用高质量示教;π0.7 加 metadata+subtask+subgoal 三路 context,能用混合质量+autonomous+人类视频+web 数据。复杂指代、跨本体、打破偏差上显著超。
- **vs DreamZero**(WAM):DreamZero 靠 video-action joint 拿物理先验,π0.7 用轻量 BAGEL 世界模型只生成 subgoal image 作条件。DreamZero 强实时闭环(7Hz),π0.7 强组合泛化和 coaching。
- **vs HarmoWAM**(双 expert WAM):HarmoWAM 用双 expert+门控分工;π0.7 用单 action expert+metadata 路由模式(prompt-level vs model-level)。π0.7 更轻量但依赖数据标注质量。
- **vs SuSIE**(subgoal conditioning):SuSIE 用单独模型生成 goal image chain-of-thought;π0.7 把 subgoal 系统化成训练 context 一部分,加 dropout 学任意子集,世界模型用 BAGEL 14B 更强。
- **vs OpenVLA**(开源 VLA):OpenVLA 7B 单 VLM+离散 action token;π0.7 5B VLM+860M flow-matching action expert(连续)+多维 context。π0.7 灵巧和组合泛化远超,但 OpenVLA 完全开源可复现。

## 局限

- 论文自承:零样本泛化成功率(60-80%)低于 in-distribution(>90%),seen vs unseen 任务/本体组合有差距(p15)。
- 论文自承:大且多样数据上很难确定哪些任务真正 seen vs unseen,可能含相关技能(不同标签或作其他任务一部分),generalization 可能主要是 remixing 已见技能(p15)。
- 论文自承:context 丰富化依赖 metadata 标注质量(speed GT 但 quality/mistake 人工粗标),标注成本未量化。
- 我们读出:π0.7 实际是三模型系统(4B VLA+14B 世界模型+4B high-level policy),非端到端;部署需多卡(世界模型 4×H100)。
- 我们读出:subgoal 只在 25% batch 加训练,推理 always 用 subgoal 可能与训练分布 mismatch;未给 always-subgoal 消融。
- 我们读出:跨本体叠衣匹配人类是单任务单本体(UR5e)且人类零样本首次尝试(非最佳),泛化到更多本体/更难任务是否仍成立未证。
- 我们读出:coaching 范式依赖人类实时分步指令,high-level policy 需 coaching 数据训练——完全自主长程新任务仍需先收集 coaching 数据,非完全 zero-data。

## 可复现性

| 字段 | 值 |
|---|---|
| code | https://pi.website/pi07(project page,未明确开源代码) |
| weights | 未开源(商业模型) |
| hardware | 双臂移动平台 / BiPi 静态双臂 / UR5e 双臂+Robotiq;推理单 H100(VLA),世界模型 4×H100 |
| real_eval | 灵巧任务+指令跟随+跨本体+组合泛化+人类被试对比(10 操作员) |
| training | 从 Gemma3 4B 初始化,knowledge insulation,RTC 训练时模拟延迟 |
