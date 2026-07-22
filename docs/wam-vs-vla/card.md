# WAM vs VLA 鲁棒性研究 — 结构化卡片

> 论文:**Do World Action Models Generalize Better than VLAs? A Robustness Study**(Huawei Technologies · University of Toronto, arXiv 2603.22078, 2026-04-30)
> 配套:`card.json`(机器可读) · `architecture.md`(架构图) · `podcast.md`(口播稿)

## 一句话定位

在 LIBERO-Plus 和自建 RoboTwin 2.0-Plus 两个扰动基准上系统对比 VLA/WAM/混合三类策略,发现 WAM 在视觉扰动(噪声/光照/布局)上更鲁棒,但相机视角和机器人初始状态仍是硬伤;推理比 π0.5 慢至少 4.8 倍是 WAM 部署的主要障碍。

## 它要解决的问题

**问题**:WAM(用视频生成模型做 policy)被宣称因继承视频预训练的时空先验而比 VLA 更鲁棒,但缺系统对照。要在统一扰动协议下,把 SOTA VLA、混合方法、WAM 放一起比,回答:WAM 真的比 VLA 泛化更好吗?鲁棒性来自视频先验还是数据多样性?代价是什么?

**为什么 prior work 不够**:之前 WAM 论文各自报告优于 VLA,但评测协议不统一、扰动维度不全、baseline 自选。VLA 阵营的 π0.5 也宣称强泛化,但靠大量多样化数据。两条路线的"鲁棒性来源"被混在一起,无法判断是视频先验有效,还是数据多样性有效,还是两者必需。需要一个统一基准把变量拆开。

## 输入 / 输出(评测既有模型的 protocol)

| 方向 | 名称 | 类型 | 说明 |
|---|---|---|---|
| 输入 | third-person camera | image | LIBERO-Plus: 256×256;RoboTwin 2.0-Plus: head 相机 320×240 |
| 输入 | wrist camera(s) | image | LIBERO-Plus: 1 张;RoboTwin 2.0-Plus: 2 张(每臂一张) |
| 输入 | language | text | RoboTwin: 2500 条变体(50 任务×50);R1 干扰/R2 共识改写/R3 推理链 |
| 输入 | proprio | vector | delta joint/absolute joint/delta EEF 等(随模型) |
| 输出 | action chunk | continuous | chunk size 随模型:π0.5=50, X-VLA=30, Fast-WAM/GE-Act/Cosmos/MOTUS=16, LingBot-VA=32 |

### 评测的 protocol 差异

本研究不提新 protocol,而是评测既有 VLA/WAM。三类家族差异:

**VLA 家族**(π0.5/OpenVLA-OFT/X-VLA):`<bos_vision>[img_3rd ⊕ img_wrist]<eos_vision><bos_lang>TokLLM(instruction)<eos_lang><bos_state>proprio<eos_state><bos_action>diffusion action expert → a1:H<eos_action>`。backbone 是 VLM(图像-文本预训练)。

**WAM 家族**(LingBot-VA/Cosmos-Policy/GE-Act/Fast-WAM/DreamZero):`<bos_vision>[img history]<eos_vision><bos_lang>instruction<eos_lang><bos_state>proprio<eos_state><bos_wam>[video latent z_t, action latent a_t] → 视频扩散去噪 → (未来视觉状态 z_{t+1}, 动作 a)<eos_wam>`。backbone 是视频生成模型(web 视频预训练)。

**三个关键差异**:
1. **backbone 不同**:VLA 用 VLM(静态图像-文本),WAM 用视频扩散(动态视频)。这决定是否继承时空先验。
2. **因果预测方向不同**:①IDM 式(LingBot-VA/GE-Act)先预测未来视觉状态再条件化动作;②联合去噪(Cosmos/DreamZero/Fast-WAM)共享 timestep 联合预测;③动作优先(GigaWorld-Policy)先动作再条件化视频。论文发现 IDM 式对训练数据多样性依赖更低。
3. **测试时是否生成视频**:LingBot-VA/GE-Act/Cosmos 测试时必须生成未来视觉状态再解码动作(慢);Fast-WAM/GigaWorld-Policy 测试时可跳过视频生成直接出动作(快但仍比 π0.5 慢 3×)。

## 数据集

| 数据 | 规模 | 备注 |
|---|---|---|
| LIBERO-Plus | 单臂 7-DoF Franka | Fei et al. 2025,7 类扰动;2 相机 256×256;评单臂灵巧性 |
| RoboTwin 2.0-Plus(自建) | 双臂 Aloha-Agilex, 50 任务 | 本文扩展;7 维度 21 子维度;3 相机 320×240;评双臂协调鲁棒性 |
| π0.5 RoboTwin finetune | 27.5k 训练数据 | JAX,60k steps,AdamW,cosine lr 2.5e-5→2.5e-6,batch 64,delta joint |
| 评测模型集合 | 10+ 模型 | VLA: π0/π0-FAST/π0.5/OpenVLA-OFT/UniVLA/RIPT-VLA/X-VLA/HoloBrain0-GD/ABot-M0;Hybrid: MOTUS/VLA-JEPA;WAM: GE-Act/Cosmos-Policy/LingBot-VA/Fast-WAM |

## 架构(详见 architecture.md)

- **不提模型**:对比研究,评测既有 VLA/WAM
- **评测模型跨度**:VLA 2-7B;WAM 1.5-14B(VPP 1.5B / GE-Act 2.2B / Cosmos-Policy 2B / LingBot-VA 5.3B / Fast-WAM 6B / DreamZero 14B)
- **类型**:systematic comparison: VLA vs Hybrid(VLA+WM) vs WAM, across 7 perturbation dimensions
- **核心组件**:统一扰动协议 / WAM 分类轴(5 维) / Fast-WAM 天然实验 / 三阶段训练数据对照 / 推理延迟拆解

**为什么这样设计**:核心是把"视频先验有效"和"数据多样性有效"两个混淆变量拆开。Fast-WAM 提供天然实验:同架构同 backbone,RoboTwin 用 clean+domain-randomized 训(数据多样),LIBERO 用 clean only 训(数据单一)——结果 RoboTwin 上 72.7%、LIBERO 上崩到 51.5%。这隔离出"视频先验必要但不充分,数据多样性仍是关键杠杆"。同时 IDM 式 vs 联合去噪式对比发现:IDM 式因显式状态-动作因果耦合,对训练数据多样性依赖更低。

### 数值 sense

| 项 | 值 |
|---|---|
| benchmarks | 2:LIBERO-Plus(单臂 Franka,2 相机 256×256)+ RoboTwin 2.0-Plus(双臂 Aloha-Agilex,3 相机 320×240) |
| 任务 | RoboTwin 2.0-Plus: 50 双臂任务;LIBERO-Plus: 标准 LIBERO 任务集 |
| 扰动维度 | 7 维度 21 子维度:Camera(C1/C3 active,C2 off)/Robot/Light(L1-L4)/Background(B1/B2)/Noise(N1-N5)/Layout(O1/O2)/Language(R1/R2/R3) |
| episodes | 50 episodes/task/config;8 configs/task(1 clean + 7 扰动分支) |
| Robot 扰动 | 关节 Gaussian noise std=0.1 rad(clip ±0.225 rad);gripper 极端(0.05/0.95) p=0.25 |
| Language 变体 | 2500 条预生成(50 任务×50 变体);R1 干扰~30%/R2 共识改写~50%/R3 推理链~20% |
| 模型参数跨度 | VPP 1.5B / GE-Act 2.2B / Cosmos-Policy 2B / LingBot-VA 5.3B / Fast-WAM 6B / DreamZero 14B;VLA 多 2-7B |
| action chunk | π0.5=50 / X-VLA=30 / Fast-WAM=GE-Act=Cosmos=MOTUS=16 / LingBot-VA=32 |
| 推理延迟 | π0.5 63ms(基线) / X-VLA 195ms(3.1×) / Fast-WAM 190ms(3.0×) / GE-Act 300ms(4.8×) / Cosmos-Policy 390ms(6.2×) / LingBot-VA(RW) 480ms(7.6×) / MOTUS 1175ms(18.6×) / LingBot-VA(RT) 5230ms(83×) |
| denoising steps | GE-Act: 1 state + 10 action;Cosmos: 5+5;MOTUS: 10+10;LingBot-VA(RW): 3+5;LingBot-VA(RT): 25+50(state denoising 主导) |

## 关键技术(评测方法论)

1. **统一扰动协议 LIBERO-Plus / RoboTwin 2.0-Plus** — 7 维度 21 子维度,每维度独立激活(每 config 只开一个),50 episodes/task/config。隔离每个扰动效果,同协议同硬件让 10+ 模型数字可比。
2. **WAM 分类轴(5 维)** — backbone/MOT/PretrainFree/CausalPred/ARGen。分类后能解释:Fast-WAM 是唯一 Pretrain Free 的(只 60h task 数据达 72.7%),证明视频 backbone 先验足够;IDM 式 vs 联合去诺式对数据多样性依赖不同。
3. **Fast-WAM 天然实验隔离"视频先验 vs 数据多样性"** — 同架构同 backbone,RoboTwin(数据多样)72.7% vs LIBERO(数据单一)51.5%,差 21 点。证明视频先验必要但不充分,task-specific 数据多样性仍是关键杠杆。
4. **训练数据三阶段对照** — Table 2 拆 embodied pre-training/post-training/task-specific finetune 三阶段,标注每阶段数据类型。π0.5 用 7 类数据混合达 85.7%;Cosmos-Policy 185 条轨迹无 embodied pre-training 达 82.2%——两条路殊途同归到鲁棒。
5. **推理延迟拆解** — Table 5 同硬件测各模型,拆 state/action denoising steps。state denoising 主导 runtime;Fast-WAM 跳过 state 生成最快(190ms)但仍 3× 慢于 π0.5,且数据单一时鲁棒性崩——速度和鲁棒性有 trade-off。

## 关键发现

| 发现 | 证据 | 对照 |
|---|---|---|
| WAM 整体比 VLA 鲁棒,尤其视觉扰动 | LingBot-VA 74.2% / Fast-WAM 72.7%(RoboTwin total);Cosmos 82.2% / GE-Act 80.3%(LIBERO total) | π0.5 58.6%(RoboTwin)/ 85.7%(LIBERO,靠极复杂数据) |
| WAM 鲁棒性集中在噪声/光照/布局,相机/初始状态是硬伤 | LingBot-VA: light 89.0% / noise 80.9% / layout 87.9% 强;但 camera 28.9% / robot 36.2% 弱 | 视频先验对几何配置改变帮助有限 |
| VLA 也能鲁棒,但需大量多样数据补偿视频先验缺失 | π0.5 LIBERO-Plus 85.7% 总分最高,训练用 7 类数据 | WAM 如 Cosmos-Policy 185 条轨迹无 embodied pre-training 就 82.2% |
| 视频先验必要但不充分,数据多样性仍是关键杠杆 | Fast-WAM 同架构:RoboTwin(clean+domain-rand)72.7% vs LIBERO(clean only)51.5%,差 21 点 | 唯一变量是训练数据多样性 |
| IDM 式因果耦合比联合去噪式对数据多样性依赖更低 | LingBot-VA(IDM 式)数据相对单一时仍稳;Fast-WAM(联合去噪)数据单一时崩 | 架构选择影响鲁棒性,不只是 backbone |
| 混合方法鲁棒性介于纯 VLA 和 WAM 之间 | MOTUS 71.5%(RoboTwin,第 3)/ VLA-JEPA 77.9%(LIBERO) | 部分引入视频先验能提升,但不如原生 WAM |
| WAM 推理比 π0.5 慢至少 4.8×,部署障碍 | π0.5 63ms;Fast-WAM 190ms(3×)/GE-Act 300ms(4.8×)/LingBot-VA(RT) 5230ms(83×) | state denoising steps 主导 runtime |

## Insights

1. WAM 不是万能鲁棒:它在噪声/光照/布局上强(视频先验帮忙),但在相机视角和机器人初始状态上和 VLA 一样弱(p9 Table 3)。视频先验对"视觉外观变化"有效,对"几何配置变化"无效——这是视频先验的边界(p12)。
2. π0.5 85.7% 总分最高证明:VLA 不靠视频先验也能鲁棒,但代价是训练数据极其多样复杂(7 类数据混合)。WAM 鲁棒更省事(Cosmos-Policy 185 条轨迹无 embodied pre-training 就 82.2%)。两条路殊途同归到鲁棒,但 WAM 训练更简单(p10 RQ3)。
3. **Fast-WAM 天然实验是最干净的因果论证**:同架构同 backbone,只改训练数据多样性,鲁棒性差 21 点。这把社区从"视频先验万能"的幻觉里拉回(p10 RQ2)。
4. 架构选择影响鲁棒性,不只是 backbone:IDM 式(LingBot-VA)对训练数据多样性依赖低于联合去噪式(Fast-WAM)。暗示显式因果建模提供额外架构先验(p10 RQ2)。
5. WAM 推理慢的根因是 state denoising,不是 action:GE-Act 1 state step 最快(300ms),LingBot-VA(RT) 25 state steps 最慢(5230ms)。Fast-WAM 跳过测试时 state 生成是最快 WAM,但数据单一时鲁棒性崩——速度和鲁棒性有 trade-off(p11 RQ4)。
6. 评测协议的"独立激活"设计是本研究方法论亮点:每 config 只开一个扰动维度,能隔离每个维度效果——但代价是低估了多扰动叠加时的交互效应(论文未做叠加评测)。

## vs 同类工作

- **vs 各 WAM 原论文(LingBot-VA/Cosmos-Policy/DreamZero)**:那些论文各自报告优于 VLA,但评测协议不统一、baseline 自选;本研究用统一 LIBERO-Plus/RoboTwin 2.0-Plus 协议把 10+ 模型放一起比,数字才可比。
- **vs LIBERO-Plus 原论文(Fei et al. 2025)**:LIBERO-Plus 只评 VLA,本研究扩展到 WAM 并自建 RoboTwin 2.0-Plus 双臂 benchmark,补了 WAM 在双臂协调鲁棒性上的评测空白。
- **vs DreamZero 论文**:DreamZero 因 14B Wan2.1 太贵且推理需 15min warmup 被本研究排除——这本身是个发现:大 WAM 在 benchmark-scale 评测上不实际,暗示 WAM 路线要落地推理效率是硬约束。
- **vs 各 VLA scaling 论文(π0.5/OpenVLA-OFT)**:那些论文强调数据 scaling 提泛化,本研究把"数据多样性"和"视频先验"两个变量拆开,证明两者都能到鲁棒但代价不同——这是对 scaling 路线的补充视角。

## 局限

**论文自承**:
- DreamZero 和 GigaWorld-Policy 因 checkpoint 不开源/太贵被排除,评测集合不完整,最强 WAM 可能未参评(p8)。
- Fast-WAM 在 LIBERO 上用 clean only 训是"有意的自然实验",但两个 checkpoint 的 task 不同(RoboTwin 双臂 vs LIBERO 单臂),所以"数据多样性"和"task 类型"两个变量未完全隔离(p8)。
- WAM 推理慢是部署障碍,本研究只诊断未提出加速方案(p11 RQ4)。

**我们读出**:
- 扰动评测是"独立激活"每维度,未做多扰动叠加评测——真实世界往往是多扰动同时发生,独立激活可能高估或低估实际鲁棒性。
- π0.5 在 RoboTwin 上是作者自训(JAX,60k steps),可能未达 π0.5 在 PI 原生训练的最优水平;π0.5 LIBERO-Plus 85.7% 是其论文数字,两个 setup 不完全可比。
- 7 维度扰动里 Language 只测了指令改写,没测多语言/跨文化指令;Camera 只测了视角小扰动(±10°),没测极端视角变化——扰动空间还有扩展余地。
- 所有评测都在仿真(LIBERO/RoboTwin 都是 sim),真机鲁棒性未验证;sim 里的"光照/噪声"和真机传感器噪声分布不同,结论外推到真机需谨慎。

## 可复现性

- 代码:RoboTwin 2.0-Plus benchmark 协议公开(论文 Appendix A);各模型用官方 checkpoint
- 权重:评测的模型 checkpoint 大多公开;π0.5 RoboTwin 版是作者自训(JAX);DreamZero/GigaWorld-Policy 未开源
- 仿真基准:LIBERO-Plus(公开)+ RoboTwin 2.0-Plus(自建,协议公开)
- 真机评测:无(全仿真)

## 论文重要图(详见 `figures.md`)

| 图 | 页 | 重要性 | 一句话 |
|---|---|---|---|
| [Figure 1](../../extracted/wam-vs-vla/pages/p07.png) | 7 | key | 7 类扰动可视化,定义评测口径 |
| [Figure 2](../../extracted/wam-vs-vla/pages/p09.png) | 9 | key | π0.5 失败 vs LingBot-VA 成功,WAM 视觉扰动鲁棒的定性证据 |
| [Figure 3](../../extracted/wam-vs-vla/pages/p11.png) | 11 | key | Cosmos-Policy 预测图像,denoise 能力 vs OOD 背景崩的机制证据 |

## 标签

`WAM` `VLA` `robustness study` `benchmark` `LIBERO-Plus` `RoboTwin` `perturbation` `video prior` `Huawei` `comparison`
