---
name: paper-podcast
description: Generate deep, structured paper-interpretation artifacts (card.json + card.md + architecture.md + figures.md + podcast.md + source.pdf) from a research paper PDF, then render them into a single-page HTML view. Use when the user asks to "read a paper deeply", "make a podcast script from a paper", "structure a paper", "解读论文", or wants paper content turned into audio-ready narration with architecture diagrams and figure walkthroughs. Produces a per-paper directory under the project's out/ folder with copy of source PDF, extracted text/images, machine-readable card, human-readable card, architecture deep-dive, figure guide, oral-style podcast script, and a standalone HTML rendering.
metadata:
  version: '1.0.0'
---

# Paper Podcast Skill

把一篇研究论文 PDF 变成一套"结构化解读资产"，并渲染成单页 HTML。目标是让一个没看过论文的人，听完 podcast、看完 HTML，就能在脑子里建出这篇论文的结构、记住关键数字、理解关键设计。

## 何时使用

- 用户给一个论文 PDF 路径，要"结构化解读 / 做口播稿 / 做论文播客"
- 用户要"把这篇论文讲清楚 / 讲透"
- 用户已有 survey_podcast 项目，要继续加新论文
- 用户问"怎么把 paper 变成 podcast"

不要用于：单纯抽文本（用 pdf skill）、单纯做摘要（直接写 md 即可）、非论文 PDF。

## 输出契约

每篇论文产出一个目录 `out/<slug>/`，包含：

| 文件 | 作用 | 真相源? |
|---|---|---|
| `source.pdf` | 原 PDF 副本，作为解读资产归档 | - |
| `card.json` | 机器可读结构化卡片 | **是** |
| `card.md` | 人类可读卡片（card.json 的视图） | 否 |
| `architecture.md` | 架构深入展开（Mermaid + 文字） | 否 |
| `figures.md` | 论文所有重要图 + 介绍 | 否 |
| `podcast.md` | 口播稿（基于 card 深度改写，非拼接） | 否 |
| `index.html` | 单页 HTML 渲染（含 protocol + 图） | 否 |

`card.json` 是唯一真相源。其它文件都是它的视图或扩展。改字段先改 card.json，再同步其它文件。

## card.json 字段

完整定义见 `assets/SCHEMA.md`。核心字段：

- `slug` / `title` / `authors` / `affiliation` / `arxiv` / `date` / `venue` / `category` / `one_liner` / `tags`
- `problem`: `what_they_address` + `why_prior_work_falls_short`
- `inputs_outputs`: `inputs[]` / `outputs[]` / `control_frequency` / **`concat_protocol`**（显式拼接序列 + 逐 token 解释）
- `datasets[]`
- `architecture`: `backbone` / `params` / `type` / `key_components[]` / `why_this_design` / **`numerical_sense`**（DiT/VAE/分辨率/chunk/上下文/动作/训练 全部具体数字）/ `see`
- `figures[]`: 每张图 `id` / `page` / `title` / `caption_original` / `what_it_shows` / `image_path` / `importance`(key|supportive)
- `key_techniques[]`: `name` / `problem_it_solves` / `how` / `why_it_works` / `detail_level`
- `key_results[]`: `metric` / `value` / `baseline_best` / `setup`
- `insights[]` / `novelty_vs_peers[]` / `limitations[]`
- `reproducibility`

## 工作流

### 步骤 0：定位项目根

skill 假定用户在 `survey_podcast` 项目里工作。项目根有 `scripts/` `extracted/` `out/` `SCHEMA.md`。如果用户给的是别的目录，按用户指定的来。

### 步骤 1：抽取

```bash
python3 scripts/extract_paper.py <pdf-path> <slug>
```

产出 `extracted/<slug>/`：
- `full_text.txt`：按 `===== PAGE N =====` 分页的全文
- `pages/pXX.png`：每页整页渲染（150 dpi，用于 Figure 引用）
- `images/pXX_NNN.png`：每张零散图
- `meta.json`

**升级后的 extract_paper.py 会渲染整页 PNG**，这样"Figure N"能对应到完整页面，不会拿到底部碎图。

### 步骤 2：精读 + 定位 Figure

- 读 `full_text.txt`，分批读（每批 ~300 行）
- 用 grep 找所有 `^Figure \d+` caption
- 用脚本定位每个 Figure 在 PDF 的页码（caption 出现位置 → 反推图所在页）

定位脚本：
```python
import re
from pathlib import Path
text = Path("extracted/<slug>/full_text.txt").read_text(encoding="utf-8")
pages = re.split(r"===== PAGE (\d+) =====", text)
page_of_fig = {}
for i in range(1, len(pages), 2):
    pno = int(pages[i]); content = pages[i+1]
    for m in re.finditer(r"Figure\s+(\d+)\s*:", content):
        fig = int(m.group(1))
        if fig not in page_of_fig: page_of_fig[fig] = pno
```

### 步骤 3：产出 card.json

按 `assets/SCHEMA.md` 写。关键要求：

1. **`concat_protocol` 必须是显式拼接序列**，例如：
   ```
   <bos_context>[VAE(o_{t-3}) ... VAE(o_t)]<eos_context><bos_instruction>TextEnc(c)<eos_instruction><bos_state>StateEnc(q_t)<eos_state><bos_chunk>[noisy z, noisy a]<eos_chunk>
   ```
   然后逐 token 解释：含义、维度、来源模块、训练/推理差异。

2. **`numerical_sense` 必须给具体数字**，不是"14B"就完事：
   - DiT：层数/heads/hidden_dim/ffn_dim
   - 分辨率
   - VAE：类型 + 空间/时间压缩比 + latent 通道数
   - 每帧 latent 维（推算）
   - chunk：latent 帧/raw 帧/秒/动作步数换算
   - 上下文换算到秒
   - 动作空间维度与含义
   - 训练：steps × batch，更新哪些参数

   数字来源标注：论文第 N 页 / backbone model card / 推算。混标会误导。

3. **`figures[]` 必须含全部重要图**，每张写 `what_it_shows`（不是抄 caption，是讲这张图到底在说什么、为什么重要、对应哪个论断）。

4. **数字带 baseline 和 setup**，禁止裸报"2× 提升"。

5. **承认局限**，包括论文自承和我们读出的。

### 步骤 4：产出 card.md

card.json 的人类可读视图。用表格 + 小节，把 protocol、numerical_sense、figures 速查表都放进去。

### 步骤 5：产出 architecture.md

- Mermaid 数据流图（输入→编码→主干→输出）
- Mermaid sequence 图（如闭环 KV-cache 替换）
- 每个组件文字详解
- 数值 sense 表（带出处）

### 步骤 6：产出 figures.md

每张图：
- 相对路径引用 `../../extracted/<slug>/pages/pXX.png`
- 原文 caption（精简）
- 我们写的"这张图到底在讲什么"

### 步骤 7：产出 podcast.md

口播稿原则：
1. 先讲清楚这篇要解决什么问题，不要念 abstract
2. 架构用"输入-处理-输出"讲清，假设听众没看过图
3. **必须有"输入拼接 protocol"段**：把序列讲成能在脑子里建出来的形式，说清三个坑（language 是 cross-attention 还是序列拼接、多视角怎么处理、训练/推理 context 差异）
4. **必须有"数值 sense"段**：把 14B 这种数字落到维度
5. 关键技术讲透机制，不要罗列名词
6. 数字带 baseline 和 setup
7. 承认局限和争议
8. 口语化但不口水：每段一个论点

### 语言禁忌（重要，发布自媒体必查）

**避免"不是 X，而是 Y"这一类口头禅**。这是中文自媒体最容易翻车的句式，听起来做作、说教、套路化。以下变体都要避开：

- "不是 X，而是 Y"
- "不是 X，是 Y"
- "X 不是 Y，而是 Z"（前置否定）
- "它不是 X，它要解决的是 Y"
- "真正的 X 不是 Y，而是 Z"
- "这不仅是 X，更是 Y"（递进式，同样套路）
- "X 不只是 Y，更是 Z"

**改写方法**：
- 直接说"是什么"。把"它不是又一篇跑分论文，而是要解决泛化问题"改成"它要解决泛化问题"。
- 用具体对比替代修辞性否定。把"失败不是 action 错，而是 video 错"改成"失败主要来自 video 预测错，action 提取本身没问题"。
- 必要的对比用"相比""对照""区别在"等，不用"不是…而是…"。

**检测**：跑 `python3 scripts/lint_phrases.py out/<slug>`，它扫所有 md 和 card.json，标出疑似句式。0 命中才能发布。lint 脚本见 `assets/lint_phrases.py`。

**其它要避的口语化套话**：值得一提的是 / 众所周知 / 毋庸置疑 / 不言而喻 / 换句话说（多次用）/ 总而言之 / 综上所述。这些在口播里都显得油腻。

### 步骤 8：复制原 PDF

```bash
cp <pdf-path> out/<slug>/source.pdf
```

作为解读资产归档。HTML 渲染器会在页面顶部提供 PDF 链接。

### 步骤 9：渲染 HTML

```bash
python3 scripts/render_html.py out/<slug>
```

产出 `out/<slug>/index.html`：
- 单页、自包含（CSS 内联，图片用相对路径引用 extracted/）
- 顶部：标题、作者、arxiv 链接、source.pdf 链接
- Tab 切换：Overview / Architecture / Figures / Podcast
- Overview tab：problem / inputs_outputs（含 protocol 代码块）/ datasets / numerical_sense 表 / key_results 表 / insights / limitations
- Architecture tab：architecture.md 渲染（Mermaid 用 CDN）
- Figures tab：每张图大图 + caption + what_it_shows
- Podcast tab：podcast.md 渲染

## 何时不用这个 skill

- 用户只要简单摘要 → 直接写 md
- 用户要做视频/slide → 用别的 skill
- 用户要在 webapp 里嵌入 → 自己写集成

## 文件清单（skill 自带）

- `assets/SCHEMA.md`：card.json 完整字段定义（复制到项目根 SCHEMA.md）
- `assets/extract_paper.py`：PDF 抽取器
- `assets/render_html.py`：HTML 渲染器
- `assets/lint_phrases.py`：自媒体口头禅 lint
- `assets/_template_card.json`：空 card 模板
- `assets/_template_podcast.md`：口播稿模板

## 质量标准

每篇论文做完，自检：

- [ ] card.json 合法 JSON
- [ ] concat_protocol 是显式序列，不是描述
- [ ] numerical_sense 每个子项都有具体数字
- [ ] figures 全部重要图都在，每张有 what_it_shows
- [ ] key_results 每条都有 baseline_best 和 setup
- [ ] limitations 至少 3 条，含论文自承和我们读出的
- [ ] podcast.md 有 protocol 段和数值 sense 段
- [ ] source.pdf 在 out/<slug>/ 里
- [ ] index.html 能打开，图能显示
- [ ] 所有图路径 resolve（`python3 -c "import json,os; c=json.load(open('card.json')); print([f['image_path'] for f in c['figures'] if not os.path.exists(f['image_path'])])"`）
- [ ] **`python3 scripts/lint_phrases.py out/<slug>` 返回 0 banned**（"不是…而是…"等口头禅清零才能发布）
