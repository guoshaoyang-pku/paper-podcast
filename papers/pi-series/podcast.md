# π0.7: A Steerable Generalist Robotic Foundation Model with Emergent Capabilities

2026-04-24，Physical Intelligence 的 Physical Intelligence 等人发布的论文，标题是 π0.7: A Steerable Generalist Robotic Foundation Model with Emergent Capabilities。

## 开场:这篇论文真正要解决什么

这篇论文要解决的核心问题非常具体:**怎么让一个通用 VLA 真正用上异质混合数据——含失败、含跨策略、含 RL specialist 蒸馏、含人类视频、含 web 多模态——而不是只能用精心筛选的高质量示教。**

为什么这件事难。因为 naive 训练会把不同模式平均掉。比如同一个叠衣任务,数据里有快速高质量的、有慢速犯错的、有 RL agent 跑出来的。如果你只用一条语言指令"fold the shirt"做条件,模型学到的就是这些模式的平均——产出次优行为。这就是为什么 prior VLA 必须精心筛选数据,只留高质量示教,数据多样性被锁死。

π0.7 的核心论点是:**如果你给每条轨迹的 prompt 加上足够丰富的 context——不只是任务描述,还有 episode metadata(这条数据多快、质量几分、有没有犯错)、subtask 指令(下一语义子任务)、subgoal 图像(世界模型生成的近未来期望场景)——模型就能区分"这条是高质量快速执行"还是"这条是失败演示",从而把混合质量数据变成可学习的、且能在推理时通过 metadata prompting 选最优模式。**

它走通了,顺带做到了一件很惊人的事:同一个通用 π0.7 模型,out-of-the-box 匹配任务专属的 RL specialist,在叠衣和搭盒子上 throughput 甚至超过 specialist;还能零样本迁移到从未训练过的 UR5e 工业臂上叠衣,匹配 top 2% 经验人类操作员的首次尝试。

## 它的输入和输出到底是什么

你需要在脑子里先建一个模块图。

输入有六路:**多视角观测**,最多 4 个相机(前视、2 腕部、可选后视),每路最多 6 个历史帧,stride 1 秒,resize 到 448×448,经 MEM vision encoder 做时空压缩;**proprio**,机器人关节状态含历史,linear projection 映射到 backbone 维度;**task 指令 ℓ**,总任务如 "clean the kitchen";**subtask 指令 ℓ̂**,下一语义子任务如 "open the fridge door";**subgoal 图像**,最多 3 路近未来期望图,由 BAGEL 14B 世界模型生成;**episode metadata**,speed(每 500 步分桶)、quality(1-5 分)、mistake(布尔);**control mode**,joint 或 ee。

输出就一路:**action chunk**,50 步连续动作,由 860M 的 action expert 用 flow matching 去噪生成,5 步去噪,实际执行 15 或 25 步。

### 输入到底怎么拼成一条序列

光说六路输入还不够具体,我把它拆成一条显式的拼接序列,你在脑子里能建出来。

序列大致长这样:开头是 `<bos_obs>`,里面是多视角观测加历史帧,经 MEM vision encoder 时空压缩;然后 `<bos_subgoal>` 包 subgoal 图像(最多 3 路);接着 `<bos_lang>` 是文本块,包括 task、subtask、speed、quality、mistake、control mode;再 `<bos_proprio>` 是 proprio state 经 linear projection;最后 `<bos_action>` 是 action expert 要去噪的 50 个 noisy action token。

这里有几个坑要特别说清,不然容易误解。

第一,language 和 subtask 是 causal 序列 token——后面的文本 attend 前面的,这是标准 LLM 那套。但 subgoal 图像和观测是 block-causal bidirectional——它们内部全互注意,subgoal 还能 attend 观测,但观测不 attend subgoal(未来不泄漏到当前)。所以这不是统一自回归,是混合 attention 模式。action token 也是 bidirectional 内部,而且能 attend VLM backbone 的全部 activation。

第二,多视角在观测 block 内,subgoal 也是多视角(最多 3 路,不含后视)。两者都进同一个 vision encoder,但 subgoal 可 attend 观测、反之不行。这是说 subgoal 是"对未来期望的描述",要看着当前观测来生成和解读,但不能反过来让当前观测看未来。

第三,训练和推理时这条序列长得不一样——具体是每个组件在训练时随机 drop。历史帧 30% 概率全 drop、后视 30% drop、subgoal 只在 25% 的 batch 里加(因为加了之后 action 预测退化成 inverse dynamics 太简单,多了模型反而依赖)、subtask 在有 subgoal 时 30% drop(因 visual subgoal 可替代文本)、metadata 15% 全 drop 加单字段各 5% drop。control mode 不 drop。这样模型学到的是任意 prompt 子集都能工作,推理时可灵活组合。

这个 dropout 设计是 π0.7 的隐藏杠杆——它让模型不依赖任何单一 context 组件,推理时可以根据任务选最合适的组合,比如 UR5e 叠衣用 subgoal 加 metadata。

## 架构:5B VLA + 860M action expert + 14B 世界模型

主干是 Gemma3 4B VLM,从预训练初始化。π0.7 的本体是 4B VLM + 400M MEM vision encoder + 860M action expert,加起来约 5B 参数。

这里有个关键设计叫 knowledge insulation。VLM backbone 用 FAST token 做离散 cross-entropy 监督——这是稳定的离散 loss。860M 的 action expert 用 flow matching 学连续动作——这是捕获动作多模态的生成式 loss。两者 loss 性质不同,直接耦合会互相干扰。所以 action expert attend VLM 的全部 activation,但梯度不回传 VLM。VLM 学稳定语义表征,action expert 在固定 backbone 上学连续动作分布。

action expert 是单独的 860M transformer,不是 VLM 的 head。它有 50 个 action token,用 adaptive RMSNorm 注入 flow matching 的 timestep,5 步去噪。内部 bidirectional 互注意,且能 attend VLM 全 activation。

推理时还有两个辅助模型。一个是 high-level language policy,同样是 Gemma3 4B 架构,负责产 subtask 指令——这个可以用 coaching 数据训成,实现全自主长程任务。另一个是 BAGEL 14B 世界模型,负责生成 subgoal 图像——SuSIE 风格,初始化自 web-scale 图像生成和编辑模型,输入当前观测加 subtask 加 metadata,flow matching 生成多视角 subgoal,3 路 CFG。

所以严格说,π0.7 不是单一端到端模型,是三模型系统:5B VLA + 14B 世界模型 + 4B high-level policy。部署时 VLA 在单 H100 上,世界模型需要 4 张 H100 做 tensor parallel。

## 给你一个数值 sense:这套模型到底多大

光说 5B 加 860M 可能没感觉,我把维度念细一点。

VLM backbone 是 Gemma3 4B,从预训练初始化,用 FAST token 做离散监督。MEM vision encoder 约 400M 参数,448×448 输入,做时空压缩——能把任意历史帧数压成单帧 token 数,这是它继承自 π0.6-MEM 的能力。action expert 860M,50 个 action token,adaptive RMSNorm 注 timestep,5 步 flow matching 去噪。

分辨率上,观测和 subgoal 都 resize 到 448×448。世界模型那边更复杂:ViT 走 448×336(patch 14),VAE 走 512×384(patch 16),因为 ViT 和 VAE 的 patch size 不同。粗算每视角约 1024 个 patch token,6 个历史帧 × 4 视角经 MEM 压到单帧 token 数。

action chunk 是 50 个 token,执行 15 或 25 步。训练时用 training-time RTC(real-time chunking)模拟 0 到 12 步延迟,对应 50Hz 机器人上最大 240ms 推理延迟——这是为了保证轨迹平滑,即使推理有延迟也不卡顿。

控制频率上,UR5e 跑 20Hz,其它机器人 50Hz。机器人形态多样:双臂移动平台是 2×6 自由度臂 + 1 夹爪 + 1-2 升降机构 + 3 自由度全向底盘;BiPi 静态双臂是 2×6 自由度 + 1 夹爪;UR5e 双臂是 2×6 自由度 + Robotiq 夹爪。UR5e 比训练用的臂更长更重、形态不同、桌面侧置,是跨本体迁移的最大挑战。

推理速度,最小变体 38ms——3 个相机、5 步去噪、training-time RTC(无额外推理开销)。开启 MEM vision encoder 加 subgoal image 到 context,最坏 127ms。世界模型生成一张 subgoal 要 1.25 秒,25 步去噪,在 4 张 H100 上做 tensor parallel,矩阵乘法 8bit 量化,attention 用 SageAttention 优化。但这个 1.25 秒不阻塞 VLA,异步执行——VLA 继续跑,世界模型在后台生成下一个 subgoal。

CFG 用在 metadata 上,β 取 1.3、1.7 或 2.2,引导动作朝高质量、快速、无错误的模式走。

## 真正让 π0.7 成立的:context 丰富化使数据 scaling 成立

这是论文最核心的论点,我必须单独讲。

π0.7 做了一个决定性实验(Fig.18 left)。它把叠衣数据按质量和速度分 4 桶:top 30%、top 50%、top 80%、全部数据。然后分别训 π0.7 with metadata 和 π0.7 without metadata。

结果非常打脸 naive 直觉。π0.7 with metadata 随数据增大持续提升,即使平均质量在下降——因为 metadata 让模型能区分好坏模式,坏数据也能学。π0.7 without metadata 反而越大数据越差——因为没标注的坏数据被平均进来,污染了模型。

这意味着什么。意味着 metadata 不是装饰,是异质数据 scaling 的钥匙。没有它,数据越多越烂;有它,数据越多越好,即使质量下降。这把数据获取从"精心筛选高质量示教"变成"来者不拒靠 metadata 标注",成本数量级下降。

第二个实验(Fig.18 right)进一步证明数据多样性 > 数据质量。去掉最高任务多样性的 20% 数据,π0.7 在未见短任务上显著掉;而去掉随机的 20% 不掉。这证明组合泛化的驱动力是任务多样性,不是单纯数据量。

这两个实验合起来就是 π0.7 的核心论点:context 丰富化 + 数据多样性协同,才能让通用 VLA 真正泛化。缺一不可。

## 一个最值得记住的 insight:通用模型可蒸馏 specialist

π0.7 做了一件很反直觉的事。它把 autonomous eval data——就是策略评测时收集的数据,包含 π0.6 RL 训练 agent 的 rollout 和失败 episode——大量混进训练。靠 episode metadata 区分质量。

结果是 π0.7 out-of-the-box 匹配 RL specialist π*0.6,在叠衣和搭盒子上 throughput 甚至超过 specialist(Fig.6)。Fig.7 消融证明:π0.7 no eval data 在 throughput 上显著掉——说明 specialist 性能确实来自这些 autonomous rollout 的蒸馏。

这个 insight 分量很重。它意味着通用模型可以蒸馏多个 specialist 进自身,且 metadata 让蒸馏不污染通用性。你不需要为每个灵巧任务单独 RL 训一个 specialist,一个 π0.7 通用模型 out-of-the-box 就行。这是通用 VLA 路线对"每任务 RL"范式的直接挑战。

## 跨本体迁移:不只是迁移,是涌现新策略

π0.7 最惊人的结果是跨本体叠衣。叠衣数据全部在静态双臂上收集,从未在 UR5e 上训过。但 π0.7 在 UR5e 双臂上零样本叠衣,task progress 85.6%、success 80%——匹配 10 个 top 2% 经验人类操作员在 UR5e 上的首次尝试(90.9% / 80.6%)。

但更深的 insight 在 Fig.13。π0.7 不是复制源机器人的行为。源机器人是人类用双臂——一臂扶袋一臂插入;π0.7 在 UR5e 上用单臂 pick-and-place,因为 UR5e 臂长够。源机器人人类用倾斜末端接近布料;π0.7 在 UR5e 用垂直抓取,更适合大臂的运动学。

这说明 π0.7 学到的是"理解任务目标后为目标本体发现合适策略",而非"模仿源轨迹"。这是真组合泛化,区别于模仿。这个结果指向一个实用价值:灵巧技能可以从轻量低成本平台(易遥操)迁移到高负载工业臂(难遥操),大幅降低示教成本。

## 语言 coaching:用语言而非示教教新任务

π0.7 的语言跟随能力强到一个新程度——可以被人分步指令 coaching 完成未见长程任务。比如 Air Fryer 装红薯,人一步步说"抓把手""开盖""拿红薯""放进去""关上",π0.7 跟着做。

更妙的是,这些 coaching 数据可以训一个 high-level language policy(同 Gemma3 4B 架构),把"观测加历史 subtask→新 subtask"学下来。推理时这个 high-level policy 自动产 subtask 指令喂给 π0.7,实现全自主。Fig.16 证明 π0.7 autonomous 紧跟 π0.7 coaching 表现,无需额外遥操数据。

这开启一个范式转变:新任务的数据获取从"收集示教轨迹"降到"人用语言分步指挥几次"。成本数量级下降。prior VLA 因为语言跟随弱,根本无法用这个范式——Fig.15 显示 π0.5/π0.6 在 coaching 任务上几乎全失败。

## 关键结果:挑几个有对照的数字

我挑几个有对照的。

**Out-of-the-box 灵巧任务**:同一个 π0.7 通用模型直接 out-of-the-box 匹配 RL specialist π*0.6,在 laundry 和 box building 上 throughput 甚至超。这是通用模型蒸馏 specialist 的实证。

**未见环境指令跟随**:14 个场景 × 3-6 步开放指令,跨 4 未见厨房 + 2 未见卧室。π0.7 显著超 π0.5 和 π0.6,绝对成功率高。证明语言跟随在完全未见环境成立。

**打破数据偏差**:Reverse Fridge to Microwave 任务(训练只单向,测试反向),π0.7(GC)成功,prior 全失败。subgoal image 是关键——世界模型能据反方向文本生成 subgoal。这是"语言跟随强到能对抗数据偏差"的最强证据。

**跨本体叠衣**:UR5e 零样本 task progress 85.6% / success 80%,人类 top 2% 操作员首次尝试 90.9% / 80.6%。工业臂上零样本灵巧任务达到人类水平。

**Scaling with metadata**:π0.7 with metadata 数据越大越好(即使质量降),without metadata 越大越差。metadata 是异质数据 scaling 的钥匙。

## 局限:别替它过度外推

论文自己承认的:零样本泛化成功率(60-80%)低于 in-distribution(>90%),seen 和 unseen 任务/本体组合有差距;在如此大且多样的数据上,很难确定哪些任务真正 seen vs unseen,数据里可能含相关技能(不同标签或作为其他任务一部分),generalization 可能主要是 remixing 已见技能;context 丰富化依赖 episode metadata 标注质量,speed 是 GT 但 quality 和 mistake 是人工粗标,标注成本未量化。

我们读出的:π0.7 实际是三模型系统(5B VLA + 14B 世界模型 + 4B high-level policy),不是端到端单模型,部署需多卡,世界模型要 4×H100;subgoal 只在 25% batch 加训练,推理时若 always 用 subgoal 可能与训练分布有 mismatch,论文未给 always-subgoal 消融;跨本体叠衣匹配人类是单任务单本体(UR5e)且人类也是零样本首次尝试非最佳表现,泛化到更多本体或更难任务是否仍成立未证;coaching 范式依赖人类实时分步指令,high-level policy 需 coaching 数据训练,完全自主长程新任务仍需先收集 coaching 数据,不是完全 zero-data。

所以,π0.7 是 VLA 路线组合泛化的一个强证据,但它支持的是"context 丰富化 + 数据多样性协同"这个方法论,不等于所有场景都能靠 prompt 解决——它依赖高质量 metadata 标注和大量异质数据,这是它的隐形门槛。

## 一句话收束

π0.7 是一个 5B VLA,靠把 episode metadata、subtask 指令、世界模型生成的 subgoal 图像塞进 prompt,让模型能区分混合质量数据的不同模式,从而把含失败、含 RL specialist 蒸馏的异质数据训成一个会组合泛化的通用 policy。它最有力的论证是 Fig.18,而非某个跑分——没有 metadata 数据越大越差,有 metadata 数据越大越好,即使质量下降。这把数据获取从"精心筛选"变成"来者不拒靠标注",是 VLA 数据方法论的一次实质升级。
