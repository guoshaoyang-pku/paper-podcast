# Qwen-VLA — 统一视觉-语言-动作建模

> 配套 `card.json`(真相源)。下表与卡片内容一一对应。

## 元信息

| 字段 | 值 |
|---|---|
| slug | qwen-vla |
| title | Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments |
| authors | Qwen Team |
| affiliation | Alibaba Qwen Team |
| arxiv | 2605.30280 |
| date | 2026-05-28 |
| venue | arXiv preprint (Alibaba Qwen) |
| category | VLA |

**一句话**:把 Qwen3.5-4B VLM 接一个 1.15B DiT flow-matching action expert,用 embodiment-aware prompt(文本描述本体/控制频率/horizon)统一操控、导航、轨迹预测、人类第一视角动作为同一 action-and-trajectory 预测问题,靠四阶段训练(T2A 纯文本→动作解压缩→CPT 多模态→SFT→RL)解决 VLM 已预训练而 DiT 随机初始化的不对称,一个通用模型跨多本体多任务。

**tags**:VLA · unified model · embodiment-aware prompt · flow matching · DiT action expert · Qwen3.5 · cross-embodiment · manipulation · navigation · T2A pretraining · Qwen · Alibaba

## 问题

- **要解决什么**:现有具身模型碎片化——操控模型只管桌面/灵巧,导航模型只管 waypoint,各自专属本体。能否把操控、视觉语言导航、轨迹预测、人类第一视角动作建模统一进单个 VLA,跨任务/环境/本体泛化。
- **为什么 prior work 不够**:三个根本困难——任务异质表面(操控末端/关节/夹爪/灵巧手,导航 waypoint,人类 MANO,观测/频率/horizon/维度都不同);VLM 已预训练而 DiT 随机初始化,naive 联合训练浪费且扰动 VLM;跨本体要不同动作维度却不能每本体单独 head。prior 工作要么单任务专属,要么没解决 VLM-DiT 不对称。

## 输入 / 输出

### 输入

| 名称 | 类型 | 说明 |
|---|---|---|
| visual observation | image/video | 单帧或多帧/历史窗;多视角用 <tag_start><image><tag_end> 包裹 |
| language instruction x | text | 任务指令,粗或细(fine-grained 13 维 caption) |
| embodiment prompt e | text | 模板描述本体/臂/控制频率/horizon,唯一本体接口 |
| task identifier z | text | 可选,标识任务族 |

### 输出

| 名称 | 类型 | 说明 |
|---|---|---|
| action/trajectory chunk Y | continuous | H×K 张量;实际 c≤K 通道,余零填充+mask;操控=末端/关节/夹爪,导航=(Δx,Δy,Δθ),人类=腕 SE(3)+10 eigengrasps |

**控制频率**:按数据集原生(导航 2FPS、合成 50Hz);SFT 操控 H=16/chunk,导航 H=8/chunk;action expert 少步 Euler 积分低延迟。

### 输入拼接 protocol

```
<bos_emb>The robot is {robot_tag} with {arms}[, waist][, and mobile base]. The control frequency is {FPS} Hz. Please predict the next {chunk_size} control actions to execute the following task: {ori_instruction}.<eos_emb>
<bos_view><|tag_start|><image_ego><|tag_end|><|tag_start|><image_left_wrist><|tag_end|><|tag_start|><image_right_wrist><|tag_end|><eos_view>
<bos_lang>{instruction x}<eos_lang>
<bos_hidden>[VLM hidden states, linear 投影到 DiT channel]<eos_hidden>
<bos_action>[noisy Y_τ ∈ R^{H×K}, 与 VLM hidden 拼接进 DiT joint self-attn]<eos_action>
```

**逐 token 解释**:
- `<bos_emb>...<eos_emb>`:embodiment-aware prompt,模板化文本。是模型知道"当前控制什么机器人"的唯一接口。VLN 类似写导航约定和 waypoint horizon。由 VLM 处理,hidden state 拼进 DiT。
- `<bos_view>...<eos_view>`:多视角用 <tag_start>/<tag_end> 包裹,tag 标识相机源。VLM view-aware 表征,action expert 选择性 attend。
- `<bos_lang>...<eos_lang>`:任务指令。fine-grained caption 13 维由两阶段 VLM 标注+人工校对,约 48k 对。
- `<bos_hidden>...<eos_hidden>`:VLM 处理完 prompt+图像的 hidden states,经 3.9M linear 投到 DiT channel。
- `<bos_action>...<eos_action>`:noisy action chunk 与 VLM hidden 拼接成一条序列,进 DiT joint self-attention + AdaLN timestep + multi-section RoPE。16 blocks 1.13B。flow matching 预测 velocity,推理少步 Euler。

**三个坑**:(1) language/embodiment 是 VLM causal token,但 action 是 DiT 连续张量经 self-attention,不是统一自回归——action expert 与 VLM 通过 hidden state 拼接+cross 注意;(2) 多视角在 VLM 内用 view tag,不在序列层拼接;(3) 训练阶段 context 不同:T2A 完全无图像(只文本+embodiment→action),CPT/SFT/RL 才加图;每 dataset 保留原生动作格式靠 prompt 告知。

## 数据集

| 数据 | 规模 | 备注 |
|---|---|---|
| Robot Manipulation | 74.2% | 公开(RobotSet/Galaxea/AgiBot/RoboMIND/DROID/BridgeData V2 等)+自研 1000h+合成;>10,000 小时 |
| Human Egocentric | 6.0% | Ego4D/EPIC-KITCHENS/EgoDex(829h)/EgoVerse(1300h)/Xperience;动作=腕 SE(3)+10 eigengrasps/手=32 维/步 |
| Navigation | 7.5% | 指令跟随 4.3%+物体搜索 2.3%+目标跟踪 1.0%;3-DoF 移动机器人,2FPS |
| Synthetic Simulation(自研) | 3.7% | IsaacLab+cuRobo;VLA 359,848 轨迹;LA 7.2M 轨迹/14,000 小时(用于 T2A) |
| General VL | 3.4% | captioning/VQA/OCR/interleaved/grounding;防遗忘 |
| Spatial Grounding | 2.5% | 2D bbox grounding |
| Autonomous Driving VQA | 2.4% | LingoQA/DriveAction/nuScenes-QA/DriveLM 等 |
| Fine-Grained Action Caption | 0.2% | ~48k 对,13 维细粒度,两阶段 VLM 标注+人工校对 |

## 架构(摘要)

| 字段 | 值 |
|---|---|
| backbone | Qwen3.5-4B(原生多模态 VLM,hybrid attention)+ DiT flow-matching action expert(1.15B,16 blocks) |
| params | VLM 4B;Action expert 1.15B(16×70.8M=1.13B + projection 4.9M + VLM→DiT 3.9M + timestep 2.8M + AdaLN 4.7M) |
| type | Unified VLA:VLM backbone + DiT action expert + embodiment prompt + 四阶段训练 |

**关键组件**:
- VLM backbone(Qwen3.5-4B):早期视觉语言融合,ViT 空间合并视觉 token interleave 进文本流;hybrid attention
- Action expert(1.15B DiT):VLM hidden + noisy action 拼接,joint self-attn + AdaLN + multi-section RoPE;16 blocks;flow matching
- Embodiment-aware prompt:文本模板描述本体/频率/horizon,唯一本体接口;配合零填充+mask 统一 action
- Unified action representation:Y∈R^{H×K},c≤K 通道+零填充,per-channel mask 排除填充
- View tagging:多视角 <tag_start><image><tag_end>,VLM view-aware
- Value head(RL):mean-pool VLM hidden→scalar,stop-grad

**为什么这样设计**:核心论点——操控/导航/轨迹预测表面异质但共享计算结构,可统一。三个支柱:(1)embodiment prompt+统一 action(零填充+mask)让单一 DiT 跨本体无需 per-embodiment head;(2)DiT 与 VLM 解耦,专家专注连续动作多模态高频;(3)四阶段训练解决 VLM-DiT 不对称——T2A 纯文本→动作建先验,CPT 视觉 grounding,SFT 专精,RL 闭环。把"动作学习是结构化解压缩"工程化。

### 数值 sense

| 项 | 值 |
|---|---|
| DiT VLM | Qwen3.5-4B,原生多模态;hybrid attention(gated linear 多数+grouped-query softmax 周期);ViT 空间合并 token interleave |
| DiT action | 1.15B;16 blocks × 70.8M=1.13B;projection 4.9M;VLM→DiT 3.9M;timestep 2.8M;AdaLN 4.7M |
| 分辨率 | 图像经 ViT 空间合并(具体未明,典型 448 级);导航 2FPS;合成 50Hz |
| VAE | 无独立 VAE(非 latent diffusion);action expert 直接 raw action 空间 flow matching;per-dataset quantile 归一化[-1,1] |
| 每帧 latent 维 | VLM hidden→DiT channel 经 3.9M linear;action chunk Y∈R^{H×K} |
| Chunk | SFT 操控 H=16;导航 H=8 waypoint;合成 50Hz;RL H=16,τ=1.0 训/0.6 评测 |
| 上下文 | 多视角+embodiment prompt+instruction;无长历史 episodic memory(未来工作);navigation 有历史窗 |
| 动作 | 操控:ΔEEF+Euler/quaternion+abs joint+gripper+dexterous hand;导航:(Δx,Δy,Δθ);人类:腕 SE(3)+10 eigengrasps=32 维/步;每 dataset quantile 归一化 |
| 训练 | 四阶段:T2A(冻结 VLM,2000 步,Sig-Norm)→CPT(解冻,异质混合)→SFT(双轨,VL 0.1/action 1.0)→RL(PPO+GAE γ=0.99 λ=0.95 ε=0.2,SimplerEnv,128 并行,8192 transition/iter,value head lr 1e-4/actor 5e-6);log-prob 用 ODE→SDE 转换 |

→ 详见 **Architecture** tab。

## 关键结果

| 指标 | 值 | 最强 baseline | setup |
|---|---|---|---|
| LIBERO | 97.9% (Instruct) | 98.6% (ABot-M0) / 97.6% (π0.5) | 4 splits 通用 vs specialist |
| RoboCasa-GR1 | 56.7% (Instruct) | 58.3% (ABot-M0) / 53.3% (Being-H0.5) | 24 atomic kitchen |
| Simpler-WidowX | 73.7% (Instruct) | 64.6% (StarVLA-OFT) / 63.2% (GR00T) | WidowX 单臂 |
| RoboTwin-E/H | 86.1%/87.2% (Instruct) | 86.0%/85.0% (ABot-M0) | 50 双臂任务 |
| ALOHA in-domain 平均 | 83.6% (w/ pretrain) | 71.6% (π0.5) / 48.5% (w/o pretrain 同架构) | 6 任务双臂 |
| ALOHA OOD 平均 | 76.9% (w/ pretrain) | 41.5% (π0.5) / 36.2% (w/o pretrain) | 5 类 OOD |
| VLN R2R SR/OSR | 57.5%/69.0% (Instruct) | 56.9%/64.2% (StreamVLN) | VLN-CE |
| VLN RxR SR/SPL | 59.6%/47.8% (Instruct) | 52.9%/46.0% (StreamVLN) | VLN-CE 更难 |
| SimplerEnv-OOD 静态 | 32.0% (Instruct) | 12.6% (π0.5) | 6 OOD 任务,仅 pick-and-place 训 |
| DOMINO 动态(零样本)SR/MS | 26.6%/39.5% (Instruct) | 24.1%/36.1% (LingBot-VA) / 17.2%/35.0% (PUMA fine-tuned) | 35 suite,无动态数据,仅当前帧 |
| RL 累积增益 | Simpler +2.9pp / DOMINO SR +0.9pp | SFT only | RL 仅 SimplerEnv,增益泛化 |

## Insights

- 动作学习是结构化解压缩:语言+embodiment 几 token 编码意图,动作可能数百高维值。T2A 先纯文本→动作建语言索引先验,CPT 才加视觉——解 VLM-DiT 不对称的根本解法(Fig.2/6)。
- T2A 必须无视觉且不能训太久:加图 -2.87pp(走视觉捷径);40000 步过拟合掉到 60.42%。2000 步+Sig-Norm 是甜点(Fig.6)。
- embodiment prompt 是唯一本体接口:共享 latent 空间建好后,projection 设计影响<1.2pp,Zero-Pad 参数最少故默认(Table 10)。多本体从架构问题变 prompt 问题。
- VL+VLA 共训练互益非互斥:简单任务持平无干扰,难任务(需细粒度识别+组合指令)明显赢 RoboCasa +4.9pp/RoboTwin +4.6pp(Fig.7a)。统一 VLA 直接证据。
- RL 增益不限于训练环境:SFT→RL SimplerEnv +2.9pp,DOMINO 零样本动态(无动态数据)SR 也 25.7→26.6——task-success 优化的"decisive 执行+误差恢复"泛化(Table 11)。
- 预训练先验可零样本迁移:Qwen-VLA-Base(未 SFT)在 ALOHA 抓未见物体、组合 clean up、未见背景开笔帽——VL 预训练物体词汇和视觉多样性迁移到操控(Fig.5)。

## vs 同类工作

- **vs π0.5/π0.7**(PI VLA):π0.7 用 metadata+subgoal+coaching 解异质数据,Qwen-VLA 用 embodiment prompt+统一 action 解多本体。π0.7 强 coaching 和组合泛化,Qwen-VLA 强跨本体统一和操控+导航+VL 三合一。
- **vs DreamZero/HarmoWAM**(WAM):WAM 靠 video-action joint 建物理先验,Qwen-VLA 不做 video 预测,直接 VLM→action expert。DOMINO 零样本 26.6% 超 WAM 风格 LingBot-VA 24.1%,但 WAM 物理先验 Qwen-VLA 没继承。
- **vs OpenVLA**(开源):7B 单 VLM+离散 action token,Qwen-VLA 4B+1.15B flow-matching(连续)+embodiment prompt 跨本体。Qwen-VLA 多任务统一,OpenVLA 专注操控。DOMINO 零样本 Qwen-VLA 26.6% vs OpenVLA-OFT 6.7%。
- **vs GR00T N1.6**(NVIDIA):专精操控 specialist,Qwen-VLA 通用模型在 RoboCasa 56.7% vs GR00T 49.9%,Simpler 73.7% vs 63.2%——通用反超 specialist,支持"统一训练不牺牲性能"。
- **vs NaVid/StreamVLN**(导航 specialist):Qwen-VLA 在 R2R/RxR 都超 specialist 导航模型,且同时是操控模型——操控+导航共训练互益。

## 局限

- 论文自承:具身动作数据规模多样性远小于 VL 预训练数据,限制长尾物体/环境/本体/接触密集交互鲁棒性(p26)。
- 论文自承:VL+导航+动作联合训练引入优化 trade-off,action 训练让部分纯 VL 和导航评测小幅回退,需更好平衡/课程/模块专精(p26)。
- 论文自承:当前评测仍主要短程 benchmark 驱动,长时程易失败真实部署仍是开放挑战(p26)。
- 我们读出:T2A 完全无视觉建语言索引先验,对视觉歧义大(同语言多视觉场景)任务可能不够,需 CPT/SFT 补;论文未给 T2A 先验在视觉歧义任务的消融。
- 我们读出:state conditioning 消融(Table 12)proprio 收益 marginal(≤1.3pp)故默认不用,但这是多视角观测充分前提下;腕部不可见/遮挡场景可能损失,论文未给此边界分析。
- 我们读出:RL 只在 SimplerEnv 单环境,泛化增益虽在但幅度小(多数<1pp),且依赖稀疏二值奖励语义;对奖励难二值定义的任务(操控质量)RL 路线未证。
- 我们读出:统一 action 用零填充+mask,但 K(固定通道)和 H(固定 horizon)选择论文未详述——对本体 horizon 差异大(导航 2FPS vs 操控 50Hz),固定 H 可能造成某些本体 chunk 过长或过短。

## 可复现性

| 字段 | 值 |
|---|---|
| code | https://github.com/QwenLM/Qwen-VLA |
| weights | 开源(GitHub) |
| project_page | https://qwen.ai/blog?id=qwenvla |
| sim_benchmark | LIBERO/Simpler/RoboCasa-GR1/RoboTwin 2.0/SimplerEnv-OOD/DOMINO/VLN-CE |
| real_eval | ALOHA 双臂(6 in-domain + 5 类 OOD) |
| training | 四阶段 T2A→CPT→SFT→RL;RL 用 RLinf,128 并行环境 |
