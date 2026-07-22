# Paper Podcast

把研究论文变成深度解读资产：结构化卡片 + 架构图 + 口播稿 + HTML 视图 + 音频播客。

每篇论文产出一个目录 `papers/<slug>/`，包含：
- `source.pdf` — 原 PDF
- `card.json` — 机器可读结构化卡片（**真相源**）
- `card.md` — 人类可读卡片
- `architecture.md` — 架构深入展开（Mermaid + 文字）
- `figures.md` — 论文所有重要图 + 介绍
- `podcast.md` — 口播稿
- `index.html` — 单页 HTML 渲染（4 tab：Overview / Architecture / Figures / Podcast）
- `figures/` — 页图（HTML 自包含）
- `audio/` — 生成的 MP3（走 Release 附件，不提交）

## 仓库结构

```
paper-podcast/
├── skill/                  # 可被 install 的 skill
│   ├── SKILL.md
│   └── assets/
│       ├── SCHEMA.md
│       ├── extract_paper.py
│       ├── render_html.py
│       ├── lint_phrases.py
│       └── build_blog_index.py
├── scripts/                # 项目用的工作脚本（从 skill/assets 同步）
├── papers/                 # 论文解读产物
│   └── <slug>/
├── docs/                   # GitHub Pages 站点
│   └── index.html          # 博客首页
└── README.md
```

## 安装 skill

```bash
# 复制 skill 到 Verdent skills 目录
cp -r skill/* ~/.verdent/skills/paper-podcast/
```

## 处理一篇新论文

```bash
# 1. 抽取全文 + 页图
python3 scripts/extract_paper.py <pdf> <slug>

# 2. 精读 extracted/<slug>/full_text.txt，产出 card.json / card.md /
#    architecture.md / figures.md / podcast.md

# 3. 复制原 PDF
cp <pdf> papers/<slug>/source.pdf

# 4. 复制页图
cp extracted/<slug>/pages/*.png papers/<slug>/figures/

# 5. 渲染 HTML
python3 scripts/render_html.py papers/<slug>

# 6. lint 口头禅
python3 scripts/lint_phrases.py papers/<slug>

# 7. 更新博客首页
python3 scripts/build_blog_index.py
```

## 质量标准

每篇论文做完，自检：
- card.json 合法 JSON
- concat_protocol 是显式拼接序列
- numerical_sense 每个子项有具体数字
- figures 全部重要图都在，每张有 what_it_shows
- key_results 每条有 baseline_best 和 setup
- limitations 至少 3 条
- podcast.md 有 protocol 段和数值 sense 段
- `lint_phrases.py` 返回 0 banned

## 发布

- **Blog**：GitHub Pages，source 指向 `docs/` 或 main 分支根
- **音频**：走 GitHub Release 附件，podcast.md 里外链播放
- **Bilibili / 小红书**：用 audio + figures 关键图合成视频

## License

MIT
