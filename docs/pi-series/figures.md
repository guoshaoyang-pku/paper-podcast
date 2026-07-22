# π0.7 — 论文重要图

> 配套 `card.json` / `card.md`。论文共 22 张 Figure,全部列出。每张图给出:所在页、原文 caption(精简)、我们写的"这张图到底在讲什么"。
> 页图存在 `extracted/pi-series/pages/pXX.png`,可对照查看。

## Figure 1 — Overview + 数据来源(page 1)

![Figure 1](../../extracted/pi-series/pages/p01.png)

**原文 caption**:We introduce π0.7, a steerable generalist robot foundation model that can perform dexterous tasks across many tasks, environments, and robots. π0.7 is trained with diverse prompts that contain not only the task description, but detailed language, generated subgoal images, and episode metadata.

**这张图讲什么**:门面图。左上数据来源:Robot Data(demonstration)+ Autonomous Data(含失败)+ Non-Robot Data(egocentric human + multimodal web)。右上训练:π0.7 VLA 吃 "Episode Metadata + Subgoal Images + Language Instructions" 三类 context。右下推理:High-Level Policy 产 subtask + World Model 产 subgoal image,喂给 action expert。核心信息:π0.7 不是更大模型,是用更丰富的 prompt 解锁异质数据利用 + 组合泛化。

## Figure 2 — Architecture(page 4)★ 最重要

![Figure 2](../../extracted/pi-series/pages/p04.png)

**原文 caption**:The π0.7 model is a 5B-parameter VLA consisting of a 4B VLM backbone, a MEM-style video history encoder and a 860M parameter action expert. The model's context includes multiple distinct modalities, including language commands, episode metadata that describes the data quality and strategy, and multimodal inputs such as subgoal images. At runtime, the language commands are produced by a high-level semantic policy based on the same architecture, and the subgoal images are produced by a lightweight world model based on the BAGEL image generation model.

**这张图讲什么**:全文最该看。中间 VLA:4B Gemma3 VLM backbone + MEM history encoder + 860M action expert(flow matching)。Context 输入(左上):language commands、episode metadata(quality/speed/mistake)、subgoal images。运行时(左下):high-level policy 产 subtask 指令、BAGEL 世界模型产 subgoal 图像,异步喂给 VLA。一张图同时回答:模型多大(5B)、context 有哪几路、推理时 subtask/subgoal 哪来。注意 action expert 是单独的 860M transformer,不是 VLM head。

## Figure 3 — Prompt 结构示例(page 6)

![Figure 3](../../extracted/pi-series/pages/p06.png)

**原文 caption**:π0.7 uses diverse modalities of context in the prompt, including: subtask instructions, subgoal images, and episode metadata. We train the model with dropout for each component, and then prompt the model flexibly combining modalities.

**这张图讲什么**:两个 prompt 完整示例。例1 "put food on table":subtask "push the open button on the microwave" → "pick up the plate of food in the microwave" → "put the plate with food on the dining table",配 metadata(Quality 5/5, Speed 2000, Mistake false)和 subgoal 图。例2 "fold the shirt":subtask "close the microwave" 等。强调每个组件训练时随机 drop,推理可任意组合(如 UR5e 叠衣用 subgoal+metadata)。这张图把"context 丰富化"落到具体可读的 prompt 串。

## Figure 4 — 评测机器人(page 7)

![Figure 4](../../extracted/pi-series/pages/p07.png)

**原文 caption**:Illustrations of some of the robots in our experiments. We evaluate π0.7 on a variety of robots, including bimanual mobile manipulators (left), static bimanual robots (middle), and a bimanual UR5e setup (right) that we use for cross-embodiment experiments.

**这张图讲什么**:三类机器人平台。左:双臂移动平台(2×6DoF 臂 + 1 夹爪 + 1-2 升降 + 3 全向底盘)。中:BiPi 静态双臂(2×6DoF + 1 夹爪)。右:UR5e 双臂(2×6DoF + Robotiq 夹爪,跨本体测试用)。UR5e 比训练用的臂更长更重、形态不同、桌面侧置,是跨本体迁移的最大挑战。UR5e 20Hz,其它 50Hz。

## Figure 5 — 长程任务示例(page 8)

![Figure 5](../../extracted/pi-series/pages/p08.png)

**原文 caption**:Illustration of selected evaluation tasks. We evaluate π0.7 on a number of tasks, and two of the more longer-horizon ones are visualized here.

**这张图讲什么**:两个长程任务的可视化。Take Out Trash:粗指令 "take out the trash" 自主完成全程。Toasting a Bagel:训练未见任务,靠 coaching(分步语言指令 "open toaster oven"/"grasp knob"/"pick up plate"/"put bagel on plate")完成。展示 π0.7 的两种能力:长程自主 + 语言 coaching 学新任务。

## Figure 6 — Out-of-the-box 灵巧任务(page 9)

![Figure 6](../../extracted/pi-series/pages/p09.png)

**原文 caption**:π0.7 can perform a wide range of highly dexterous tasks directly out of the box. We consider tasks from π*0.6 (top row) and a number of other dexterous tasks including ones from the "Robot Olympics" experiments (bottom row).

**这张图讲什么**:上下两行任务。上行(π*0.6 任务):Laundry(T恤短裤 / 多样最难项)、Make Espresso、Box Building。报 success rate + normalized throughput(相对 specialist)。下行:Make PB Sandwich、Shirt Inside-Out、Drive Through Door、Slice Zucchini、Peel Fruits/Veg、Take Out Trash,报 task progress。核心结论:同一个 π0.7 通用模型直接 out-of-the-box 匹配 RL specialist π*0.6,在 laundry/box building 上 throughput 甚至超过 specialist。

## Figure 7 — Metadata + Eval Data 消融(page 9)

![Figure 7](../../extracted/pi-series/pages/p09.png)

**原文 caption**:Impact of prompt composition and evaluation data on out-of-the-box performance: We compare π0.7 with two ablations: one that does not include episode metadata in the context, π0.7 (no metadata), and another that does not include data from autonomous evaluation episodes during training, π0.7 (no eval data).

**这张图讲什么**:三模型对比:π0.7 vs π0.7(no metadata) vs π0.7(no eval data),在 Laundry/Espresso/Box 上比 throughput + success。π0.7 全胜,gap 在 throughput 最明显。证明两件事:(1)metadata 是关键——去掉它无法区分混合质量数据;(2)autonomous eval data(含 RL specialist 蒸馏)是 specialist 性能来源。这是论文核心论点的直接消融证据。

## Figure 8 — 记忆任务(page 10)

![Figure 8](../../extracted/pi-series/pages/p10.png)

**原文 caption**:Tasks that require memory: π0.7 can also perform tasks that require explicitly keeping track of prior context, achieving similar or better performance compared to the specialist policies with memory fine-tuned to some of the tasks in the MEM paper.

**这张图讲什么**:四个需记忆的任务:Swap 3 Mugs、Find Object、Scoop Coffee、Window Cleaning。π0.7 out-of-the-box 匹配或超过 π0.6-MEM SFT specialist(后者是专门 finetune 的)。证明 MEM history encoder 让 π0.7 不 finetune 也有记忆能力——是 context 丰富化 + 架构继承的复合收益。

## Figure 9 — 新环境指令跟随(page 11)

![Figure 9](../../extracted/pi-series/pages/p11.png)

**原文 caption**:Broad instruction following in novel environments: We evaluate π0.7 on 14 instruction following scenarios, each of which involve following a sequence of 3-6 open-ended instructions, across 4 unseen kitchen and 2 unseen bedroom environments.

**这张图讲什么**:14 个指令跟随场景,每个 3-6 步开放指令,跨 4 未见厨房 + 2 未见卧室。报 instruction following success rate(正确跟随的指令占比)。π0.7 显著超 π0.5 和 π0.6,绝对成功率高。证明 π0.7 的语言跟随能力在完全未见环境也成立——不是过拟合训练环境。

## Figure 10 — 复杂指代指令(page 11)

![Figure 10](../../extracted/pi-series/pages/p11.png)

**原文 caption**:Following complex referential instructions: π0.7 and prior models all succeed on the simpler re-arrangement instructions, but π0.7 performs significantly better on the complex and unusual instructions.

**这张图讲什么**:Office Desk 物体重排,分 Standard("pick up the spoon")和 Complex("pick up the object I would use to eat soup"/"pick up the fruit on the largest plate")。π0.7 在 complex 显著领先,加 subgoal image(π0.7 GC)进一步拉开。证明 subgoal image 把世界模型的语义理解注入 policy,在语言指代模糊时用视觉 disambiguate。

## Figure 11 — 打破数据偏差(page 11)

![Figure 11](../../extracted/pi-series/pages/p11.png)

**原文 caption**:Breaking dataset biases by following instructions: the improved language-following performance of π0.7 enables it to break strong dataset biases.

**这张图讲什么**:两任务:Reverse Bussing(训练 bussing 是垃圾入垃圾桶+餐具入餐盘,测试要求反着来)、Reverse Fridge to Microwave(训练只单向,测试反向)。π0.7 显著超 prior,且 Reverse Fridge 任务 π0.7(GC)即 subgoal image 是成功关键——世界模型能据文本指令生成反方向 subgoal。这是"语言跟随强到能对抗数据偏差"的最强证据。

## Figure 12 — 跨本体迁移(page 12)

![Figure 12](../../extracted/pi-series/pages/p12.png)

**原文 caption**:Cross-embodiment transfer: Left: Both π0.7 and prior models achieve strong cross-embodiment transfer directly out of the box on simpler re-arrangement or repositioning style tasks. Right: for the more dexterous tasks that require folding towels and t-shirts, the embodiment gap poses an even greater challenge.

**这张图讲什么**:左:简单重排任务跨本体(Table Setting/Bag In Backpack/Organize Tupperware/Shirt Bagging),embodiment gap 递增。π0.5 在大 gap 崩,π0.6/π0.7 撑住,最大 gap(单臂 UR5e)π0.7 显著胜。右:灵巧叠衣从静态双臂迁 UR5e(从未见过),π0.7 成功且加 subgoal(π0.7 GC)更好,task progress 匹配人类 teleoperator 零样本表现。这是论文最惊人的结果。

## Figure 13 — 涌现策略适应本体(page 12)

![Figure 13](../../extracted/pi-series/pages/p12.png)

**原文 caption**:Cross-embodiment transfer produces emergent strategies adapted to the target embodiment. (a) On the source robot, human teleoperators use one arm to hold the bag open while the other performs insertion. On the UR5e target robot, π0.7 instead discovers a single-arm pick-and-place strategy. (b) Human teleoperators approach the shirt with a tilted end-effector on the source robot, while π0.7 produces vertical grasps on the UR5e.

**这张图讲什么**:两个对比案例。(a)装包:人类在源机器人用双臂(一扶袋一插入),π0.7 在 UR5e 单臂 pick-and-place(因 UR5e 臂长够)。(b)叠衣:人类在源机器人倾斜末端接近,π0.7 在 UR5e 用垂直抓取(更适合大臂)。关键 insight:π0.7 会为目标本体发现合适策略,而非复制源行为——这是真组合泛化,区别于模仿。

## Figure 14 — 语言 coaching 学新任务(page 13)

![Figure 14](../../extracted/pi-series/pages/p13.png)

**原文 caption**:Example of language coaching: We can "teach" a new task to π0.7 by providing step-by-step verbal instructions. Because of its language following ability, π0.7 can perform new tasks successfully under user instruction, and these instructions can then by used to train a high-level policy that prompts π0.7 so that it can perform the task fully autonomously.

**这张图讲什么**:Air Fryer 装红薯的 coaching 序列:"grasp handle"→"open air fryer"→"pick up sweet potato"→"put into air fryer"→"close"。π0.7 跟随分步指令完成未见任务。coaching 数据可训成 high-level policy 实现全自主(Fig.16)。这开启"用语言而非示教教机器人新任务"的范式。

## Figure 15 — Coaching 长程新任务(page 13)

![Figure 15](../../extracted/pi-series/pages/p13.png)

**原文 caption**:Coaching to perform new long-horizon tasks: Because π0.7 can follow language instructions effectively, even for unfamiliar skills, it can be "coached" to perform a number of unseen, longer horizon tasks both when conditioned on language and generated subgoal images (π0.7 (GC)).

**这张图讲什么**:三任务:Loading/Unloading Air Fryer、Toasting Bagel。π0.7 vs π0.6 vs π0.5 + π0.7(GC)。prior 模型语言跟随差无法 coaching,π0.7 显著领先,GC 进一步提升。证明 coaching 是 π0.7 独有能力,prior VLA 因语言跟随弱无法用此范式。

## Figure 16 — Coaching 转自主(page 13)

![Figure 16](../../extracted/pi-series/pages/p13.png)

**原文 caption**:Acquiring new autonomous capabilities with coaching: We can use the coaching episodes collected for a number of different unseen tasks to train a high-level policy to automatically prompt π0.7 in accordance with the coaching episodes.

**这张图讲什么**:五任务:Scoop Rice/Reverse Fridge/Loading Air Fryer/Toasting Bagel/Unloading Air Fryer。对比 π0.7(coaching 人类实时指令) vs π0.7(autonomous 用 coaching 数据训的 high-level policy)。autonomous 紧跟 coaching 表现,无需额外遥操数据。证明 coaching→high-level policy 闭环成立,是新任务数据高效获取路径。

## Figure 17 — 短程新任务 out-of-the-box(page 13)

![Figure 17](../../extracted/pi-series/pages/p13.png)

**原文 caption**:Performing new short-horizon tasks: π0.7 can perform a number of new short-horizon tasks directly out of the box, including scooping rice into a rice cooker, spinning various objects such as a gear set and desk fan, and wiping down objects with a cloth, such as a ruler and headphones, despite no data being collected for any of these tasks.

**这张图讲什么**:短程新任务:Press French Press Plunger/Scoop Rice/Wiping Office Supplies/Spinning Articulated Objects(gear/fan)。无任何该任务数据,π0.7 直接 out-of-the-box 完成,语言或 GC 表现相近。证明 π0.7 能组合已学技能做新短任务——compositional generalization 的直接证据。

## Figure 18 — Scaling 实验(核心 insight)(page 14)★ 最重要

![Figure 18](../../extracted/pi-series/pages/p14.png)

**原文 caption**:Scaling of generalization performance with diverse context and data: Left: We find that π0.7 (with metadata) can continuously improve its performance when it is trained on larger datasets, even when the average quality of the data actually decreases. By contrast, without training on rich conditioning information, π0.7 (without metadata) actually can degrade in performance as more lower quality data is introduced. Right: When π0.7 is trained without our robot data with the highest task diversity, its performance degrades substantially.

**这张图讲什么**:两张图支撑论文最核心论点。左:数据分 4 桶(top30%/50%/80%/all),π0.7(with metadata)随数据增大持续提升即使平均质量下降;π0.7(no metadata)反而变差——证明 metadata 是异质数据 scaling 的钥匙。右:去掉最高任务多样性 20% 数据,π0.7 在未见短任务上显著掉,而去掉随机 20% 不掉——证明任务多样性是组合泛化的驱动力。这张图是"context 丰富化 + 数据多样性"协同论点的决定性证据。

## Figure 19 — Attention 模式细节(page 22)

![Figure 19](../../extracted/pi-series/pages/p22.png)

**原文 caption**:The π0.7 model and its world model (for generating subgoal images) use several different nontrivial attention patterns during training and inference. From top left: in absence of image goals we use the same attention patterns as in π0.5, with global bidirectional attention between embeddings for all memory-aware image views. Note that the FAST tokens (only available at training time) and the flow actions do not attend to each other.

**这张图讲什么**:VLA 和世界模型的 attention mask 设计。VLA:无 subgoal 时同 π0.5(memory 图像 bidirectional);有 subgoal 时加一个 block-causal bidirectional block 在 text 后;CFG 推理时正负样本 pack 成 attention tree 两分支互不 attend。世界模型:3 路图(current-ViT、current-VAE、noisy goal-VAE)block-bidirectional,3 路 CFG 比 VLA 的 2 路更复杂。这张图是 Fig.2 的 attention 细节展开,解释 block-causal mask 怎么实现。

## Figure 20 — Joint vs EE 控制(page 23)

![Figure 20](../../extracted/pi-series/pages/p23.png)

**原文 caption**:Joint vs. end-effector (EE) control for prior models on cross-embodiment tasks. We compare joint-space and end-effector (EE) control for baseline policies across a range of tasks, observing no substantial difference in performance between the two control modes.

**这张图讲什么**:Table Setting/Bag In Backpack/Organize Tupperware/Shirt Bagging/Towel Folding/Shirt Folding 六任务,π0.5/0.5EE/0.6/0.6EE 对比。结论:joint 和 EE 控制无显著差异。因此主跨本体实验用 joint 控制(更简单)。这是工程选择证据,非核心结果。

## Figure 21 — 人类被试经验分布(page 23)

![Figure 21](../../extracted/pi-series/pages/p23.png)

**原文 caption**:Operator experience in the human subject study. Box plots show teleoperation experience (in hours) of the ten recruited operators across three categories: UR5e (target robot), the static bimanual robot (source robot), and all robots combined.

**这张图讲什么**:10 个 top 2% 经验操作员在 UR5e/静态双臂/全部机器人上的遥操经验箱线图。平均 ~375 小时全平台经验。关键:他们在 UR5e 叠衣是零样本(从未做过)。这是 Fig.22 人类 baseline 的可信度证据——操作员是真专家但任务零样本,和 π0.7 公平对比。

## Figure 22 — π0.7 vs 人类(叠衣)(page 23)

![Figure 22](../../extracted/pi-series/pages/p23.png)

**原文 caption**:Comparison of π0.7 (GC) and human. We find that π0.7 (GC) achieves competitive performance compared to the human operators, in the shirt folding task with the UR5e bimanual platform.

**这张图讲什么**:UR5e 双臂叠衣任务:π0.7(GC)task progress 85.6% / success 80% vs 人类 90.9% / 80.6%。π0.7 在从未训练过的本体上零样本叠衣,匹配 top 2% 经验人类操作员的首次尝试。这是论文最惊人的单点结果——工业臂上零样本灵巧任务达到人类水平。

---

## 用法

- 想看模型长什么样:Figure 2(架构)、Figure 19(attention 细节)
- 想看结果:Figure 6(灵巧)、Figure 9-11(指令跟随)、Figure 12-13(跨本体)、Figure 22(vs 人类)
- 想理解核心论点:Figure 18(scaling,最重要)、Figure 7(metadata 消融)
- 想理解 coaching 范式:Figure 14/15/16
- 想看 prompt 长啥样:Figure 3
- 想看硬件:Figure 4
