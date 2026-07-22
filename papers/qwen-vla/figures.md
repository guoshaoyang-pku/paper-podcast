# Qwen-VLA — 论文重要图

> 配套 `card.json` / `card.md`。论文共 7 张 Figure,全部列出。每张图给出:所在页、原文 caption(精简)、我们写的"这张图到底在讲什么"。
> 页图存在 `extracted/qwen-vla/pages/pXX.png`,可对照查看。

## Figure 1 — Overview(page 1)

![Figure 1](../../extracted/qwen-vla/pages/p01.png)

**原文 caption**:Overview of Qwen-VLA, a unified embodied model trained on mixed manipulation, navigation, and vision-language understanding data to generate both robot actions and textual responses.

**这张图讲什么**:门面图。Qwen3.5 VLM 接 DiT action expert(Noisy Action→MLP→Diffusion Transformer N×→Clean Action)。Qwen-VLA 同时支持 VLA(操控)+VLN(导航)+VL(视觉语言理解)。输入:observed images + prompt,输出 robot actions 或 textual responses。核心信息:一个模型统一三类任务,不是三个 head。

## Figure 2 — 四阶段训练 recipe(page 6)★ 最重要

![Figure 2](../../extracted/qwen-vla/pages/p06.png)

**原文 caption**:Training recipe of Qwen-VLA. Stage I (T2A) trains the DiT action decoder to reconstruct actions from text alone, building a structured action prior without visual input. Stage II (CPT) unfreezes both modules to ground this prior in visual observations. Stage III (SFT) branches into multi-task and real-robot tracks, and Stage IV (RL) optimizes closed-loop task success via environment rewards.

**这张图讲什么**:全文最该看。四阶段:(I)T2A 冻结 VLM,DiT 仅从文本+embodiment 重建动作(无图像)→建结构化动作先验;(II)CPT 解冻两模块,视觉 grounding;(III)SFT 双轨(多任务+真实机器人);(IV)RL 环境奖励优化闭环成功。每阶段闭合前一阶段的能力缺口。这张图把"动作学习是结构化解压缩"的视角工程化——T2A 学语言→动作的解压映射,CPT 才加视觉。

## Figure 3 — ROBOINF 合成数据示例(page 11)

![Figure 3](../../extracted/qwen-vla/pages/p11.png)

**原文 caption**:Examples of data generated through ROBOINF. The top row shows a short-horizon task, "Place the two green staplers side by side." The bottom row shows a long-horizon task, "Group the drinks together and leave the cleaning sponge by itself."

**这张图讲什么**:ROBOINF 合成数据生成示例。上:短程任务(place two green staplers side by side)的 reaching→grasping→placing→completed 序列。下:长程任务(group drinks + leave sponge)分解为 pick/place 7Up→pick/place Red Bull→pick/place sponge 的子任务段。展示合成数据覆盖原子技能+组合指令,且轨迹自然分解为中间阶段做多粒度监督。

## Figure 4 — ALOHA 真实评测任务(page 17)

![Figure 4](../../extracted/qwen-vla/pages/p17.png)

**原文 caption**:Overview of real-world evaluation tasks on the ALOHA bimanual platform.

**这张图讲什么**:ALOHA 双臂真实评测任务全览。6 个 in-domain 任务(pick&place/table cleaning/bowl stacking/bowl pick&place/towel folding/fine-grained)+ 5 类 OOD(color/instance/position/background/instruction generalization)。定义了 Table 5/6 所有真实结果的评测口径——OOD 是 5 正交维度,不是简单换背景。

## Figure 5 — Qwen-VLA-Base OOD 定性(page 21)

![Figure 5](../../extracted/qwen-vla/pages/p21.png)

**原文 caption**:Qualitative out-of-distribution generalization of Qwen-VLA-Base on the ALOHA dual-arm robot. Top-left: color-conditioned grasping. Top-right: novel object manipulation and compositional clean-up. Bottom-left: unseen object interaction. Bottom-right: background robustness.

**这张图讲什么**:Qwen-VLA-Base(未 SFT)在 ALOHA 上的 4 类 OOD 定性。左上:颜色条件抓取(绿/蓝/红/黄球都正确)。右上:未见物体(西兰花/玩具鸭)+ 组合 clean up table(蓝伞/玩具鸭/瓶装酸奶顺序入筐)。左下:完全未见物体交互(墨镜/毛绒娃娃/玩具鸭,approach 动作训练里几乎没有)。右下:未见黄色背景下开笔帽放笔帽(灵巧两阶段)。证明预训练的视觉语言先验迁移到操控,即使无 in-domain 示教。

## Figure 6 — T2A 消融(核心 insight)(page 22)★ 最重要

![Figure 6](../../extracted/qwen-vla/pages/p22.png)

**原文 caption**:T2A pretraining ablations. (a) Data composition and prediction mode. (b) Flow-matching timestep distribution. (c) T2A training duration.

**这张图讲什么**:三组 T2A 消融支撑"动作学习是解压缩"论点。(a)20% 合成+80% 真实(去图)最佳 71.09%;full-sequence > chunk(+4.94pp);T2A 加图反而 -2.87pp(让 decoder 走视觉捷径,稀释语言-动作先验)。(b)Sig-Norm(T2A)+Beta(SFT)最佳 71.09%;反过来都掉——T2A 无视觉引导时中间 timestep 信噪比最 informative,SFT 有 VLM 条件时 Beta 更 sample-efficient。(c)2000 步最佳,40000 步过拟合掉到 60.42%。这张图决定性证明 T2A 必须无视觉且不能训太久。

## Figure 7 — VL 共训练 + 预训练 DiT 迁移(page 23)

![Figure 7](../../extracted/qwen-vla/pages/p23.png)

**原文 caption**:Vision-language co-training ablations. (a) Impact of VL data on action learning. (b) Transferability of pretrained DiT.

**这张图讲什么**:两组共训练消融。(a)VLA-Only vs VL+VLA:简单 benchmark(Libero/Simpler)持平,需细粒度物体识别和组合指令的(RoboCasa +4.9pp/RoboTwin +4.6pp)VL+VLA 明显赢——证明 VL 共训练无干扰且在难任务有用。(b)预训练 DiT vs from-scratch DiT(都接 Qwen3.5-4B):预训练 DiT 全程更快收敛且更高峰值。证明 T2A 学的先验可迁移到新 VLM,不是绑定特定 backbone。

---

## 用法

- 想看模型长什么样:Figure 1(overview)、Figure 2(四阶段训练)
- 想看结果:Figure 5(OOD 定性)、Table 4-9(数值)
- 想理解核心论点:Figure 6(T2A 消融,最重要)、Figure 7(VL 共训练)
- 想看数据:Figure 3(合成数据)、Figure 4(真实评测任务)
