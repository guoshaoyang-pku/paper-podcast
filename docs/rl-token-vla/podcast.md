# RL Token: Bootstrapping Online RL with Vision-Language-Action Models

2026-04-30，Physical Intelligence 的 Charles Xu 等人发布的论文，标题是 RL Token: Bootstrapping Online RL with Vision-Language-Action Models。

## 开场：这篇论文真正要解决什么

RLT 是一个**微调框架**——决定"怎么在真实机器人上，用几小时数据，把一个预训练好的 VLA 精修到能做毫米级精密任务"。它不造新 VLA，也不造新 RL 算法，只解决这个"最后一毫米"问题。

问题长这样：现在的 VLA，像 π0.6，开箱能做很多操作，但它在"最后一毫米"上经常失败。动作慢、要重试、critical phase 一个小误差累积成失败。装个螺丝，对齐差一点就拧不进去；插个 Ethernet 接头，角度差一点就卡在壳上。这种精密任务靠示教数据很难覆盖——示教本身就有噪声，操作员也会失败。

自然想法是用 RL 精修——让机器人在任务上自己练，发现更快更准的策略。但真实机器人 RL 有个紧约束：每个 episode 都要时间，每次失败都耗磨损，几小时预算内必须看到改进。

而 VLA 上的 RL 一直卡在两个极端之间。一端是直接更新整个 VLA，像 RECAP 那样训整个 π*0.6——算力贵、样本低效，几小时根本不够。另一端是用 SERL、RL100 这类小模型从零做 online RL——几小时能学，但放弃了 VLA 的泛化先验，从零学一个精密任务太难。

作者要回答的问题是：**能不能既保住 VLA 的感知加行为先验，又拿到小 actor-critic 的样本效率？几小时真实数据，能不能把毫米级任务的 critical phase 提速并提升成功率？**

## 它的解法：分工哲学——VLA 当先验，小网络做精修

RLT 的核心思想是分工。冻结的 VLA 提供感知理解和行为先验，给一个好的初始策略；一个小 actor-critic 在它基础上做 local refinement，专门攻克 critical phase。

这个分工靠三件事落地。

第一，**给 VLA 训一个"RL token"**。VLA 的内部 token embedding 是高维的，2048 维乘以 M 个 token，直接喂给小 RL 网络既算力贵又样本低效。但用 ResNet 这种通用视觉编码器又丢了 VLA 学到的 manipulation 结构。所以作者在 VLA token 序列末尾 append 一个学习的特殊 embedding，过一个小 transformer，取那个位置的输出，叫它 z_rl，1×2048 维。训练这个小 transformer 时，用一个 decoder 从 z_rl 自回归重建原来的 embedding——这就逼 z_rl 成为一个瓶颈，必须保留足够信息才能重建。训完之后冻结。这个 z_rl 就是给小 actor-critic 的状态表示。

第二，**actor-critic 在 chunk 上操作，不在单步上**。这是 RLT 和之前单步 RL 方法的关键区别。VLA 原生输出 50 步的 chunk，对应 1 秒。RLT 的 actor 也输出 chunk，但短一点，C=10 步，0.2 秒。为什么 chunk 这么重要？因为这些精密任务 episode 1500 到 6000 步，稀疏奖励只在 episode 末给一次 success 或 failure。单步 RL 要在这么长 horizon 上做 credit assignment，reward 信号根本传不回去。chunk-level 把 TD horizon 缩到 10 步级别，reward 信号才能传播。论文专门对比了 HIL-SERL 和 PLD 这两个单步方法，在 Ethernet 任务上完全失败——这就是直接证据。

第三，**actor 条件化于 VLA 的参考动作，并加 KL 正则**。actor 不从零生成动作，而是看 VLA 采样的参考 chunk ã，在这个基础上输出一个高斯分布。训练 loss 是负 Q 值加一个 β 乘以动作和参考的 L2 距离。等于把 online RL 变成"在 VLA 先验附近做 local refinement"，而不是在 140 维动作空间里无约束搜索。

这三件事互相强化，让几小时真实数据精修毫米级任务成为可能。

## 输入到底怎么拼成一条序列

我把 RLT 的数据流写成一条显式序列，你在脑子里能建出来。

序列大致长这样：开头是 `<bos_vla>`，里面是图像加语言加 proprio 喂给冻结的 π0.6，产出 token embeddings z1:M，每个 2048 维。然后 `<eos_vla>`。接着 `<bos_rltok>`，在 z1:M 末尾 append 一个学习的 embedding e_rl，过 encoder transformer g_φ，取那个位置的输出 z_rl，1×2048 维，`<eos_rltok>`。然后 `<bos_rlstate>`，里面是 z_rl 拼上 proprio，`<eos_rlstate>`。然后 `<bos_ref>`，里面是 VLA 采样的参考动作 chunk ã1:H，实际取前 C=10 步，`<eos_ref>`。最后 `<bos_actor>`，小 MLP actor 输出高斯分布，采样 a1:C，10 步乘 14 维，`<eos_actor>`。

这里有几个坑要特别说清。

第一，**language 不是序列拼接到 RL 侧**。实验里每个任务固定一条指令，RL token 提取时甚至把语言 embedding drop 掉了。所以 RL 看到的状态是 z_rl 加 proprio，没有显式语言 token。这是个简化，论文承认多指令、语言条件 RL 没验证——这是 future work。

第二，**多视角在 VLA 内部已经被处理成 token 了**。两张腕部相机加一张 base 相机，喂给 π0.6，RL 侧只见 z_rl 这个压缩向量。这点和 DreamZero 在通道维拼多视角不同——RLT 完全依赖 VLA 内部处理。

第三，**训练和推理时 reference action 的处理不一样**。推理时 ã 必给，actor 看着 VLA 的建议输出动作。训练时，ã 有 50% 概率被 drop 成零再喂给 actor。这个 reference action dropout 是反直觉但必要的 trick。为什么要 drop？因为如果总给 ã，actor 会倾向于直接抄作业——尤其 critic 还没信号的时候，抄 ã 就是最低风险的局部最优。dropout 逼 actor 保一条独立的 action 生成通路，等 critic 有信号了，actor 自然学会在能提升 Q 的时候偏离 ã。

第四，**chunk 内是开环执行**。RLT 输出 C=10 步动作，0.2 秒内开环执行不接新观测，然后重新观测、重新规划。这点和 DreamZero 闭环 KV-cache 替换完全不同。0.2 秒开环在静态精密任务上够用，但如果是高频动态任务可能受限——论文没测这种场景。

记住这条序列，后面听别的 VLA-RL 论文你会反复遇到类似的拼接，区别往往在 VLA 怎么冻结、状态怎么压缩、参考动作怎么用。RLT 的答案是：VLA 全冻结、状态压成 1×2048 瓶颈、参考动作条件化加 dropout。

## 给你一个数值 sense：这套系统到底多大

光说"冻结 VLA + 小 actor-critic"可能没感觉，我把维度念细一点。

主干是 π0.6，内部是 SigLIP 400M 视觉编码器加 Gemma 4B 语言模型，再加 860M 的 diffusion action expert。总参数大概 5.3B，全部冻结。RLT 不动它一根毛。

RL token 是 1×2048 维的 bottleneck 向量。产生它的 encoder 和 decoder 是轻量 transformer，论文没给精确层数，但远小于 VLA。在任务 demo 上训 2000 到 10000 个 gradient step，之后冻结。

actor-critic 是真的小。zip tie、Ethernet、charger 三个任务用 2 层 MLP，hidden dimension 256。screw 任务因为更难，用 3 层 MLP，hidden 512。参数量在百万以下。这就是为什么几小时数据能学——要训的参数太少了。

action 这边：RL chunk C=10 步，50Hz 控制频率，对应 0.2 秒。每步 14 维动作，所以 actor 输出 140 维的 chunked action。VLA 那边原生 chunk 是 H=50 步，1 秒。RLT 用前 10 步做参考。

控制频率 50Hz。chunk 内开环执行 10 步，0.2 秒后重新规划。replay buffer 的 stride 是 2，意思是每个 chunk 存多个中间步的 transition，1 秒数据大概产 25 个 RL 样本。off-policy 让所有数据——VLA warmup rollout、RL 自主 rollout、人类 teleop 干预——都进 buffer 复用。

online RL 的训练规模：每个任务 400 到 1000 个 episode，实际机器人数据 15 分钟到 5 小时。update-to-data ratio 是 5，意思是每条数据训 5 次——这是低数据 regime 必需的。每 2 个 critic 更新对应 1 个 actor 更新。critic 用 TD3 风格的 twin Q 集成，取 min 算 target，防 value overestimation。

reward 是稀疏二值，+1 表示 success，0 表示 failure，由人标。episode 长 30 到 120 秒，对应 1500 到 6000 个 control step。critical phase 是 5 到 20 秒，250 到 1000 步——这是 RL 真正发力的地方。

记住这套数字，后面听别的 VLA-RL 论文时可以拿它当标尺：5.3B 冻结 VLA、1×2048 RL token、几层 MLP、C=10 chunk、50Hz、U2D=5、15min~5h 真实数据。最该记住的是"小"——actor-critic 在百万参数以下，所以几小时能学。

## 关键技术讲透：RL token 为什么是瓶颈

这是 RLT 最核心的设计，值得单独讲。

问题是这样：VLA 的 final-layer token embedding 高维，2048 维乘以 M 个 token。直接喂给 online RL 既算力贵又样本低效——小 MLP 没法吃这么多输入。但用 ResNet 这种通用视觉编码器，又丢了 VLA 在大规模 web 加机器人数据上学到的 manipulation-relevant 结构。

作者的解法是造一个瓶颈。在 VLA token 序列末尾 append 一个学习的 embedding，叫 e_rl。然后过一个小 transformer encoder，取 e_rl 那个位置的输出，这就是 z_rl，1×2048 维。

为什么是瓶颈？因为训练的时候，有一个 decoder 要从 z_rl 出发，自回归地重建原来的所有 embedding。重建 loss 是所有 token 的重建误差之和。z_rl 要保留足够信息才能让 decoder 重建出来——所以它被迫成为一个压缩表示，既小又信息密集。

这里有个细节叫 stop-gradient。decoder 重建的时候，原始 embedding 是 stop-gradient 的，意思是 decoder 的梯度不会回传到 VLA。φ 只学"怎么压缩"，不动 VLA 一根毛。

消融证明这个设计有效。把 RL token 换成 ImageNet 预训练的 ResNet-10 编码器，throughput 降 50%。这就直接证明 RL token 编码了 ResNet 没有的、manipulation-relevant 的结构。这是 RLT 区别于"小模型 + ResNet"路线的根本——它不丢 VLA 的知识。

## 关键技术讲透：为什么 chunk-level 这么重要

这是 RLT 区别于单步 RL 方法的关键，论文专门对比了 baseline 来证明。

50Hz 的精密任务，episode 1500 到 6000 步。稀疏奖励只在 episode 末给一次 success 或 failure。如果是单步 RL，value function 要在这 1500 到 6000 步的 horizon 上做 credit assignment——也就是说，要判断每一步动作对最终 success 贡献多少。但 reward 信号隔了几千步，根本传不回去。

RLT 的解法是 chunk-level。actor 和 critic 都在 chunk 上操作，C=10 步。critic 估的是 chunk-level 的 C-step return，意思是"这个 chunk 之后 10 步的累计 reward 加上下一个 chunk 的 value"。TD horizon 从 1500~6000 步缩到 10 步级，reward 信号才能有效传播。

论文专门对比了 HIL-SERL 和 PLD 这两个单步方法，在 Ethernet 任务上完全失败。这是直接证据：不是这些方法本身差，是单步接口在稀疏奖励长 horizon 上根本学不出来。

同时 C=10 < H=50 还有个好处：RL 策略比 VLA 更 reactive，0.2 秒重新规划一次，而 VLA 是 1 秒。在 critical phase 需要快速调整的时候，更短的 chunk 让 RL 更灵活。

## 关键技术讲透：reference action conditioning 把 RL 变成 local refinement

这是 RLT 让几小时数据够用的另一个关键。

从零学一个 140 维 action chunk 的高斯 actor，在几小时数据下根本不可行。无约束探索会陷入失败模式，actor 在巨大的动作空间里乱撞。

RLT 的解法是 actor 不从零生成，条件化于 VLA 采样的参考 chunk ã。actor 输入是状态 x 加参考 ã，输出一个高斯分布，均值是 μ_θ(x, ã)。训练 loss 是负 Q 值加一个 KL 正则项：β 乘以动作和参考的 L2 距离平方。

等于把 online RL 变成"在 VLA 先验附近做 local refinement"。actor 不能跑离 VLA 太远，否则 KL 正则惩罚；但可以在 VLA 附近找 Q 值更高的动作。几小时数据够，因为搜索空间被限制在 VLA 附近。

条件化 ã 还有个好处：VLA 的动作分布是多模态的，同一状态可能有多条合理动作。一个单模态高斯 actor 没法表达多模态。但条件化于 VLA 采样的 ã，每次 ã 是从一个模态采的，actor 就能间接利用 VLA 的多模态信息。

但条件化 ã 有个 failure mode：actor 可能直接抄 ã，不学改进。尤其在 critic 还没信号的时候，抄 ã 就是最低风险的局部最优。所以训练时 50% 概率把 ã drop 成零。这逼 actor 保一条独立的 action 生成通路——当 ë 被_drop 时，actor 只能从状态 x 自己生成动作。等 critic 有信号了，actor 自然学会在能提升 Q 的时候偏离 ë。

消融很干净：w-o BC Regularizer，就是 β=0，去掉 KL 正则，跌得最狠。因为没有 anchor，actor 在全 140 维空间裸搜，几小时数据根本不够。w-o Pass-Through，去掉参考动作输入，最终能追平但学得慢，训练过程中失败多。这两个消融共同证明：KL 正则 + 参考条件化，两个机制叠加才让几小时数据够用。

## 关键结果：不仅提成功率，还提速度，甚至超过人类 teleop

我挑几个有对照的数字。

**critical phase 提速**：四个任务的 critical phase，RLT 比 base π0.6 VLA 快最多 3 倍。这是最显眼的结果，也是论文卖点。15 分钟到 5 小时数据就能做到。

**success rate 提升**：在 VLA 弱的任务上提升大。screw installation full-task 从 base 提 40%，zip tie fastening 提 60%。在 VLA 已经不错的任务上，像 Ethernet 和 charger，RLT 维持成功率先，主要提速——throughput 上去了，成功率没掉。

**Ethernet 速度**：这个数字我特别想讲。critical phase 的 episode length 中位数：expert teleop 是 146 步，base VLA 是 228 步——VLA 比 teleop 还慢，因为它有"probing"行为，接近目标、退一点、再调整、再试，循环几次才成功。RLT 是 66 步。中位数不到 teleop 的一半！而且 RLT 一半的 episode 比 teleop 全部都快。

这点分量很重。它说明 RLT 不是在模仿示教，它学出了 teleop 数据里没有的策略——直接接近端口、压入、轻微 wiggle 利用 compliance，一次流畅插入。这个策略是 online RL 探索出来的，示教里没有。这是"RL 能超越示教数据"的最有力证据。

**vs 其它 RL 算法**：Ethernet 任务对比四个 baseline。HIL-SERL 和 PLD 这两个单步方法完全失败，学不出来。DAgger 和 DSRL 成功率追平 RLT，但 throughput 明显低。DAgger 是 imitation learning，受限于人类示教速度，提不了速。DSRL 在 diffusion noise 空间学 latent policy，强约束 VLA 动作生成，稳定但改进空间小。RLT 既高成功又高吞吐。

**sample efficiency**：Ethernet 任务上，RLT 用 5 分钟数据就超过 base policy。整个实验总时长约 40 分钟。这是"几小时甚至几分钟"那个卖点的具体数字。

**消融**：w-o RL token，换 ResNet-10 编码器，throughput 降 50%——证明 RL token 编码了 manipulation 结构。w-o Chunk，单步，credit assignment 失败。w-o BC Regularizer，最大单点跌幅——无 anchor 让 actor 全空间裸搜。w-o Pass-Through，最终能追平但学得慢+训练中失败多。四个组件每个都有贡献。

## 一个最值得记住的 insight

论文里有个观察，我建议你重点记：**RLT 学到的策略，比 expert teleop 还快，而且快的方式是 teleop 数据里没有的**。

这句话分量很重。它意味着两件事。第一，RL 不只是"在示教基础上微调"，它能发现示教者没展示过的、更优的策略。RLT 在 Ethernet 上学会压入加 wiggle 利用 compliance，这是 teleop 数据里没有的行为。第二，这给 VLA 的scaling 论证一个新方向——VLA 提供好的初始策略，RL 在它基础上探索超越示教的策略。pretraining 该做的是给 good initialization for downstream exploration，最成功的策略靠 RL 发现。

这是 RLT 对"VLA + RL"路线最有意义的论证——其含义是"RL 微调让机器人超越示教"(而非仅"让 VLA 更准")。

## 局限：别替它过度外推

论文自己承认的：仍需人在环。reward 信号、干预纠正、RL 和 base policy 的切换，都靠人。作者明说全自主 RL pipeline，用 reward model 加 progress prediction 自动化这些，是 future work。critical-phase 隔离评测降低方差但也简化了问题，full-task 评测显示 compounding error 仍让整体成功率低于 critical-phase only。只在固定语言指令的任务上验证，每任务一条指令；RL token 提取时实验性 drop 语言 embedding——多指令、语言条件 RL 没验证。

我们读出的：四个任务都是 50Hz 双臂或单臂精密插入类，形态接近。RLT 在更动态任务上，比如接球、长horizon 多阶段装配，是否仍几小时收敛，没证。actor-critic 是 Gaussian 单模态，靠 reference ë 暴露多模态；如果 VLA 本身多模态弱，RLT 多模态恢复能力受限，论文没测。RL token encoder/decoder 的精确参数量、层数论文没给，β 的具体值也没明示——复现需要从 code release 补。

所以 RLT 是"VLA + 几小时 online RL"路线的一个强证据，但不是终点。它支持"VLA 当先验、小网络做精修"这个方向，不等于全自主、长horizon、多模态任务已经解决。

## 一句话收束

RLT 把一个 5.3B 的 π0.6 VLA 完全冻结，给它训一个 1×2048 的 RL token 瓶颈表示，在它上面挂一个几层 MLP 的 actor-critic，用 chunk-level 动作加参考动作条件化加 KL 正则，几小时甚至几分钟真实机器人数据就能把毫米级精密任务的 critical phase 提速 3 倍。它最有力的论证在跑分之外,是"RL 学到的策略比 expert teleop 还快，而且快的方式是示教里没有的"——这把 VLA 的角色从"最终策略"重新定义成"探索的好初始化"。
