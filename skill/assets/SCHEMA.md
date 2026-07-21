# Structured Paper Card — Schema

Each paper lives under `out/<slug>/` with these files:

```
out/<slug>/
  card.json       # 机器可读结构化卡片
  card.md         # 人类可读结构化卡片(与 card.json 同源)
  podcast.md      # 口播稿(基于 card 深度改写,面向"听懂论文")
  architecture.md # 架构图描述(文字版结构图,可后续渲染为图)
  sources.txt     # 该卡片引用的论文原文片段(便于核对)
```

## card.json 字段定义

```json
{
  "slug": "dreamzero",
  "title": "World Action Models are Zero-shot Policies",
  "authors": ["Seonghyeon Ye", "..."],
  "affiliation": "NVIDIA",
  "arxiv": "2602.15922",
  "date": "2026-02-19",
  "venue": "arXiv preprint (NVIDIA)",
  "category": "Joint WAM",            // VLA | Cascaded WAM | Joint WAM | Planning/MPC | Human Data | Survey | Eval
  "one_liner": "一句话定位这篇论文的真正贡献(不是抄 abstract)",

  "problem": {
    "what_they_address": "这篇要解决什么具体问题",
    "why_prior_work_falls_short": "之前的方法为什么不够(精确到哪条"
  },

  "inputs_outputs": {
    "inputs": [
      {"name": "visual context", "type": "image/video", "detail": "当前与历史观测,VAE 编码到 latent"},
      ...
    ],
    "outputs": [
      {"name": "future video latents", "type": "latent", "detail": "未来 K 个 latent 视频帧"},
      {"name": "action chunk", "type": "continuous", "detail": "H 步连续动作(相对关节位置)"}
    ],
    "control_frequency": "30Hz (AgiBot) / 15Hz (DROID),chunk 1.6s",
    "concat_protocol": "显式写出输入序列的拼接形式,例如:\n<bos_context>[VAE(o_{t-3}) ... VAE(o_t)]<eos_context><bos_lang>T(text(c))<eos_lang><bos_state>state(q_t)<eos_state><bos_chunk>[noise z_1, noise a_1 | ... | noise z_M, noise a_M]<eos_chunk>\n标注每个 token 的含义、维度、来源模块、是否训练/推理时不同"
  },

  "datasets": [
    {"name": "AgiBot G1 自采", "size": "~500h", "note": "22 个环境,7193 episodes,平均 4.4min/42 subtasks"},
    {"name": "DROID", "size": "公开", "note": "Franka 单臂验证可复现性"},
    {"name": "YAM 跨本体", "size": "20min", "note": "video-only,robot-to-robot transfer"},
    {"name": "人类第一视角", "size": "12min", "note": "video-only,human-to-robot transfer"}
  ],

  "architecture": {
    "backbone": "Wan2.1-I2V-14B-480P (image-to-video diffusion)",
    "params": "14B",
    "type": "autoregressive DiT + flow matching",
    "key_components": [
      "VAE 编码视觉 -> latent",
      "文本编码器(冻结)",
      "state encoder / action encoder / action decoder(新增最小参数)",
      "autoregressive DiT 主干,KV-cache 推理",
      "chunk-wise teacher-forcing 训练"
    ],
    "why_this_design": "为什么这样设计(从论文 reasoning 还原,不是抄)",
    "numerical_sense": {
      "dit": "主干 transformer 规格:层数/heads/hidden_dim/ffn_dim",
      "resolution": "输入分辨率",
      "vae": "VAE 类型 + 空间/时间压缩比 + latent 通道数",
      "per_frame_latent_dim": "每帧 latent 的近似维度,给听众一个 size 的感觉",
      "chunk": "chunk 在 latent 帧/raw 帧/秒/动作步数 上的换算",
      "context": "最大上下文换算到秒",
      "action": "动作空间维度与含义",
      "training": "steps × batch, 更新哪些参数, 是否 LoRA"
    },
    "see": "architecture.md"
  },

  "figures": [
    {
      "id": "Figure 1",
      "page": 1,
      "title": "短标题",
      "caption_original": "论文原文 caption(精简)",
      "what_it_shows": "我们写的:这张图到底在讲什么、为什么重要、对应哪个论断",
      "image_path": "extracted/<slug>/pages/p01.png",
      "importance": "key | supportive | supplementary"
    }
  ],

  "key_techniques": [
    {
      "name": "Joint video-action flow matching",
      "problem_it_solves": "video 和 action 对齐",
      "how": "单一模型端到端联合去噪,共享 timestep,共享 velocity 预测目标",
      "why_it_works": "论文给出的理由 + 我们的理解",
      "detail_level": "deep"
    },
    ...
  ],

  "key_results": [
    {"metric": "seen tasks avg task progress", "value": "62.2%", "baseline_best": "27.4% (pretrained VLA)", "setup": "AgiBot G1"},
    ...
  ],

  "insights": [
    "论文最反常识/最值得记住的判断(附原文位置)"
  ],

  "novelty_vs_peers": [
    "vs DreamZero 之前的 WAM(如 BagelVLA/Cosmos):差在哪",
    ...
  ],

  "limitations": [
    "论文自己承认的 + 我们读出的"
  ],

  "reproducibility": {
    "code": "https://github.com/dreamzero0/dreamzero",
    "weights": "开源",
    "sim_benchmark": "PolaRiS / Genie Sim 3.0"
  },

  "tags": ["WAM", "video diffusion", "autoregressive", "cross-embodiment", "NVIDIA"]
}
```

## 口播稿 (podcast.md) 原则

1. **先讲清楚这篇要解决什么问题**,不是念 abstract。
2. **架构用"输入-处理-输出"讲清**,假设听众没看过图也能在脑子里建出结构。
3. **关键技术讲透机制**,不是罗列名词:flow matching 怎么训、KV-cache 为什么省、Flash 为什么能少步。
4. **数字带 baseline 和 setup**,禁止裸报"2× 提升"。
5. **承认局限和争议**,不替论文背书。
6. 口语化但不口水:每段一个论点,避免"然后然后然后"。

## 渲染优先级

- `card.md` 是真相源(机器+人读)
- `podcast.md` 由 `card` 深度改写而来,不是简单拼接
- `architecture.md` 用 Mermaid + 文字,后续可生成图
