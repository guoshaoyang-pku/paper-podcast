# Do World Action Models Generalize Better than VLAs? A Robustness Study

2026-04-30，Huawei Technologies 的 Zhanguang Zhang 等人发布的论文，标题是 Do World Action Models Generalize Better than VLAs? A Robustness Study。

## 开场：这篇论文真正要解决什么

这份研究把现在最火的两条机器人策略路线，VLA 和 WAM，放在统一扰动协议下系统比一比，看看到底谁更鲁棒，为什么。它是一个**对照实验**，而不是新模型或新训练方法。

背景是这样。最近一年，WAM，也就是用视频生成模型做机器人策略的那一脉，像 LingBot-VA、Cosmos-Policy、DreamZero 这些，纷纷报告说自己比 VLA 鲁棒。理由听起来很顺：WAM 的 backbone 是视频生成模型，在 web 视频上预训练过，继承了时空先验，知道"动作会让世界怎么变"，所以泛化更好。而 VLA 的 backbone 是 VLM，在静态图像文本上预训练，只有语义先验，缺物理动态先验。

但问题是，这些 WAM 论文各自评测、baseline 自选、协议不统一，数字根本不可比。而且 VLA 阵营的 π0.5 也宣称自己鲁棒性强——它靠的是把 web data、多环境 tabletop、cross-embodiment、移动操作、高层规划、语言指令七类数据全混进去训，用数据多样性补偿先验缺失。

于是"鲁棒性到底来自视频先验还是数据多样性"这个问题被搅成一团浆糊。两个变量在大多数模型上耦合在一起，分不开。

作者要回答的是：**WAM 真的比 VLA 泛化更好吗？如果更好，是视频先验的功劳，还是训练数据多样性的功劳，还是两者都得有？代价是什么？**

## 它的评测框架：统一扰动协议 + 两互补 benchmark

要公平对比，第一步是统一评测协议。作者用了两个 benchmark。

一个是 LIBERO-Plus，别人已经提出的，单臂 Franka 机器人，7 类扰动，两个相机 256×256。另一个是作者自建的 RoboTwin 2.0-Plus，双臂 Aloha-Agilex 平台，50 个任务，三个相机 320×240，同样 7 类扰动但扩展到 21 个子维度。两个 benchmark 互补——LIBERO 评单臂灵巧性，RoboTwin 评双臂协调鲁棒性，结论不依赖单一 setup。

七类扰动分别是：Camera（相机视角）、Robot（机器人初始状态）、Language（语言指令改写）、Light（光照）、Background（背景纹理）、Noise（传感器噪声）、Layout（干扰物布局）。每个维度再细分，比如 Noise 有 5 种：motion blur、Gaussian blur、zoom blur、fog、glass blur。Light 有 4 种：diffuse color、direction、specular、shadows。

评测协议有个设计亮点：**独立激活**。每个 config 只开一个扰动维度，其余保持 clean。50 个 episode 每任务每 config，8 个 config 每任务——1 个 clean baseline 加 7 个扰动分支。这样能隔离每个维度的效果，避免混淆。比那种"所有扰动一起上"的评测更能定位失败原因。

然后是模型集合。作者把 10 多个模型分三类放一起比：纯 VLA（π0、π0-FAST、π0.5、OpenVLA-OFT、UniVLA、RIPT-VLA、X-VLA、HoloBrain0-GD、ABot-M0）、混合方法（MOTUS、VLA-JEPA，部分引入视频先验的 VLA）、WAM（GE-Act、Cosmos-Policy、LingBot-VA、Fast-WAM）。所有模型用同一协议、同一硬件评测，数字才可比。

这里有个诚实交代：DreamZero 和 GigaWorld-Policy 被排除。DreamZero 因为 14B 的 Wan2.1 backbone 太贵，重训不现实，而且推理需要 15 分钟 warmup，benchmark 级评测几千个 episode 根本跑不动。GigaWorld-Policy 没开源。这本身是个发现——最大的 WAM 在 benchmark 评测上不实际，暗示 WAM 路线要落地，推理效率是硬约束。

## 三类策略到底差在哪：把 protocol 讲清

光说"VLA 用 VLM backbone、WAM 用视频 backbone"还不够具体，我把三类家族的差异拆成三个关键点。

第一，**backbone 不同**。VLA 的 backbone 是 VLM，在静态图像文本上预训练，继承了语义先验。WAM 的 backbone 是视频扩散模型，在 web 视频上预训练，继承了时空先验——知道物体怎么动、物理动态怎么演化。这决定了它们能不能"预测动作会让世界怎么变"。

第二，**因果预测方向不同**。WAM 内部还分三式。IDM 式，像 LingBot-VA 和 GE-Act，先预测未来视觉状态，再条件化动作——p(h_{t+1}|h_t) 乘以 g(a|h_t, h_{t+1})，类似逆动力学模型。联合去噪式，像 Cosmos-Policy、DreamZero、Fast-WAM，共享 timestep 联合预测视觉状态和动作。动作优先式，像 GigaWorld-Policy，先预测动作，再条件化生成未来视觉状态。论文后面发现，这三个方向对鲁棒性影响很大——IDM 式对训练数据多样性依赖最低，因为显式的状态-动作因果耦合提供了额外架构先验。

第三，**测试时是否生成视频**。LingBot-VA、GE-Act、Cosmos-Policy 这些，测试时必须先生成未来视觉状态，再从状态解码动作，所以慢。Fast-WAM 和 GigaWorld-Policy 测试时可以跳过视频生成，直接出动作，所以快——但 Fast-WAM 仍比 π0.5 慢 3 倍。

记住这三个差异，后面听结果时就明白为什么不同模型表现不同。鲁棒性不只是 backbone 的事，因果方向和测试时策略都影响。

## 给你一个数值 sense：这套评测到底多大

光说"7 类扰动 21 子维度"可能没感觉，我把维度念细一点。

RoboTwin 2.0-Plus 有 50 个双臂任务，每任务 8 个 config（1 clean + 7 扰动分支），每 config 50 个 episode。所以一个模型完整评测要 50 × 8 × 50 = 2 万个 episode。10 多个模型全跑，episode 数是天文数字——这也是 DreamZero 被排除的原因之一。

扰动具体长什么样？Robot 扰动是在 home 关节角度上加高斯噪声，标准差 0.1 弧度，clip 到 ±0.225 弧度；夹爪有 25% 概率被设到极端位置（0.05 或 0.95）。Language 是 2500 条预生成的指令变体，50 任务每个 50 条，分三种：R1 干扰约 30%（把指令包在无关对话里）、R2 共识改写约 50%（用功能描述替换物体名）、R3 推理链约 20%（把祈使句改成目标状态描述）。Noise 五种，severity 在 2 到 3 之间。Camera 是 head 相机距离缩放 0.85 到 1.0，yaw-pitch-roll 各 0 到 5 度扰动。

模型参数跨度很大。VLA 多在 2 到 7B。WAM 从 VPP 的 1.5B 到 DreamZero 的 14B 都有，主流是 2 到 6B：GE-Act 2.2B、Cosmos-Policy 2B、LingBot-VA 5.3B、Fast-WAM 6B。

action chunk 大小随模型：π0.5 是 50，X-VLA 是 30，Fast-WAM、GE-Act、Cosmos、MOTUS 都是 16，LingBot-VA 是 32。这影响响应频率——chunk 大开环执行久，chunk 小更 reactive。

推理延迟是这套评测最该记住的数字。π0.5 是 63 毫秒基线。X-VLA 195 毫秒，3.1 倍。Fast-WAM 190 毫秒，3 倍，是最快的 WAM，因为它测试时跳过视频生成。GE-Act 300 毫秒，4.8 倍。Cosmos-Policy 390 毫秒，6.2 倍。LingBot-VA 真实世界配置 480 毫秒，7.6 倍。MOTUS 1175 毫秒，18.6 倍。LingBot-VA 在 RoboTwin 上用 25 步 state denoising 加 50 步 action denoising，5230 毫秒，83 倍——这是 5.2 秒一次推理，根本没法实时控制。

记住这套数字，后面听 WAM 部署讨论时可以拿它当标尺：4.8 倍是 WAM 比 VLA 慢的下限，83 倍是上限。state denoising steps 是主导 runtime 的关键。

## 主结果：WAM 确实更鲁棒，但不是全方位

我挑 RoboTwin 2.0-Plus 的主结果讲。LingBot-VA 总分 74.2% 第一，Fast-WAM 72.7% 第二，MOTUS 71.5% 第三，π0.5 58.6% 垫底，X-VLA 53.1%。

但看分项就有意思了。LingBot-VA 在 Light 89.0%、Background 91.3%、Noise 80.9%、Layout 87.9% 上都是最强——视觉扰动上 WAM 确实碾压。但在 Camera 上 LingBot-VA 只有 28.9%，Robot 上 36.2%。π0.5 在 Camera 上 45.6%、Robot 上 27.6%，差距没那么大。

这说明什么？说明 WAM 的鲁棒性集中在视觉外观扰动上——噪声、光照、背景纹理、干扰物布局，这些视频先验能帮上忙，因为 web 视频见过各种各样的视觉变化。但相机视角变化和机器人初始状态变化，本质是几何配置变化，视频先验帮不上忙——视频模型没学过"视角变了世界该怎么变"。所以 WAM 在这两项上和 VLA 一样弱。

这是这篇论文第一个重要发现：**WAM 不是万能鲁棒，它有明确的边界——视觉外观强、几何配置弱**。

## 反转：LIBERO 上 VLA 反而更强

但故事没完。在 LIBERO-Plus 上，π0.5 总分 85.7% 反而最高，超过所有 WAM。Cosmos-Policy 82.2% 第二，GE-Act 80.3% 第三。

为什么 π0.5 在 RoboTwin 上垫底，在 LIBERO 上第一？因为 π0.5 的训练数据极其多样复杂——web data、multi-env tabletop、cross-embodiment、mobile manipulation、high-level planning、verbal instruction 七类数据全混进去训。它用数据多样性补偿了视频先验缺失。

对比一下 WAM 的训练。Cosmos-Policy 只用 185 条 task 轨迹，没有 embodied pre-training，就达到 82.2%。Fast-WAM 只用 60 小时 task 数据，达到 72.7%。WAM 靠视频 backbone 先验，训练简单得多。

所以第二个重要发现：**VLA 也能鲁棒，但代价是训练数据极其多样复杂；WAM 鲁棒更省事，靠视频先验省数据**。两条路殊途同归到鲁棒，但代价不同。这是对"VLA 必然输给 WAM"这种简单论调的纠正。

## 最干净的因果论证：Fast-WAM 天然实验

前面两个发现还是有点绕——π0.5 数据多样性强、WAM 先验强，两个变量在不同模型上耦合，还是分不开到底哪个更关键。作者找了一个特别干净的天然实验：Fast-WAM。

Fast-WAM 同一个架构、同一个 Wan2.2-5B backbone，但在两个 benchmark 上用了不同训练数据。RoboTwin 上用 clean 加 domain-randomized 27.5k 数据训，数据多样。LIBERO 上只用 clean 数据训，数据单一。除此之外，架构、backbone、训练流程完全相同。

结果对比：RoboTwin 上 Original 91.2%，扰动平均 72.7%，降 18 个点。LIBERO 上 Original 97.6%，扰动平均 51.5%，降 46 个点。两个 checkpoint 的扰动鲁棒性差 21 个点。

这是整个研究最有价值的发现。**唯一变量是训练数据多样性**，结果差 21 个点。这证明：**视频先验必要但不充分，task-specific 训练数据多样性仍是关键杠杆**。即使有视频 backbone 先验，训练数据不够多样，鲁棒性照样崩。

这个发现把社区从"视频先验万能"的幻觉里拉回来。WAM 不是装上视频 backbone 就自动鲁棒，你还得给够多样的 task 数据。

## 进一步：架构选择也影响鲁棒性

还有一个更细的发现。对比 LingBot-VA 和 Fast-WAM，两者都是 WAM，但因果预测方向不同。LingBot-VA 是 IDM 式，先预测未来视觉状态再条件化动作。Fast-WAM 是联合去噪式，共享 timestep 联合预测状态和动作。

Fast-WAM 在 LIBERO 数据单一时鲁棒性崩到 51.5%。LingBot-VA 在数据相对单一时还稳。论文的解释是：IDM 式有显式的状态-动作因果耦合，提供了额外架构先验，所以对训练数据多样性依赖更低。联合去噪式没有这种显式耦合，更依赖数据多样性来学到状态-动作关系。

这说明**架构选择影响鲁棒性，不只是 backbone**。设计 WAM 时，因果方向是个重要考量。如果你担心数据不够多样，IDM 式可能更稳；如果数据够多样，联合去噪式可能更灵活。这是个实用的设计指导。

## 推理延迟：WAM 部署的硬障碍

最后讲 WAM 最大的代价——慢。前面数值 sense 里我念过：WAM 全部至少比 π0.5 慢 4.8 倍，最快的 Fast-WAM 也要 3 倍，最慢的 LingBot-VA 在 RoboTwin 上 83 倍，5.2 秒一次推理。

作者拆解了慢在哪。主要是 state denoising steps，不是 action。GE-Act 只要 1 步 state denoising 加 10 步 action，所以是 WAM 里最快的之一（300 毫秒）。LingBot-VA 在 RoboTwin 上用 25 步 state 加 50 步 action，所以 5230 毫秒。state denoising 主导 runtime，因为它要生成高维的视频 latent，比 action decoding 重得多。

Fast-WAM 的加速思路是测试时跳过 state 生成，直接出动作。这让它成为最快的 WAM（190 毫秒）。但代价是：数据单一时鲁棒性崩——因为跳过 state 生成失去了视频先验的"想象未来"能力。这是个 trade-off：速度和鲁棒性不能兼得，至少在 Fast-WAM 这种联合去噪设计上是这样。

这指明了 WAM 加速的方向：减少或跳过 state denoising。但跳过有代价，所以需要更聪明的设计——比如 DreamZero-Flash 那种解耦 noise schedule，或者像 GigaWorld-Policy 那样测试时根本不生成视频但训练时保留视频预测目标。

## 一个最值得记住的 insight

这篇研究最有价值的判断，我建议你重点记：**视频先验是必要条件，不是充分条件；数据多样性仍是关键杠杆**。

这句话分量很重。它纠正了"WAM 自动鲁棒"的简单论调。你装上视频 backbone，确实继承了时空先验，能在视觉扰动上比 VLA 强。但这只是入场券，不是终点。如果 task-specific 训练数据不够多样，鲁棒性照样崩——Fast-WAM 从 72.7% 掉到 51.5% 就是证据。

这个发现对 WAM 路线的实践意义是：你不能因为有了视频 backbone 就在 task 数据采集上偷懒。反过来对 VLA 路线也是鼓励：没有视频先验，靠数据多样性也能到鲁棒，π0.5 的 85.7% 就是证据。两条路都能到鲁棒，选择取决于你的算力预算和数据可得性。

## 局限：别替它过度外推

论文自己承认的：DreamZero 和 GigaWorld-Policy 被排除，评测集合不完整，最强 WAM 可能未参评。Fast-WAM 两个 checkpoint 的 task 不同（RoboTwin 双臂 vs LIBERO 单臂），所以"数据多样性"和"task 类型"两个变量未完全隔离——这是这个天然实验的一个瑕疵。WAM 推理慢是部署障碍，本研究只诊断未提出加速方案。

我们读出的：扰动评测是"独立激活"每维度，未做多扰动叠加评测。真实世界往往是多扰动同时发生，独立激活可能高估或低估实际鲁棒性。π0.5 在 RoboTwin 上是作者自训（JAX，60k steps），可能未达 π0.5 在 PI 原生训练的最优水平。7 维度扰动里 Language 只测了指令改写，没测多语言、跨文化指令；Camera 只测了视角小扰动（±10°），没测极端视角变化。所有评测都在仿真，真机鲁棒性未验证——sim 里的"光照/噪声"和真机传感器噪声分布不同，结论外推到真机需谨慎。

所以这份研究是 WAM vs VLA 鲁棒性对照的一个扎实基线，但不是终点。它支持"视频先验有用但有边界、数据多样性仍是关键"这个判断，不等于工业现场的鲁棒部署已经解决。

## 一句话收束

这份研究在统一扰动协议下系统对比了 10 多个 VLA、WAM、混合方法，发现 WAM 在视觉扰动（噪声、光照、背景、布局）上确实比 VLA 鲁棒，但相机视角和机器人初始状态是两类方法的共同硬伤；VLA 用数据多样性也能到鲁棒，但训练更复杂。它最有力的论证是 Fast-WAM 那个天然实验，重点不在于"WAM 更强"——同架构同 backbone，只改训练数据多样性，鲁棒性差 21 个点——这把"视频先验必要但不充分，数据多样性仍是关键杠杆"这个判断钉死了。
