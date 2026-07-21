# Paper Podcast DSL — Domain-specific 结构化扩展

## 为什么需要 DSL

不同领域的论文,结构化表述的关键字段不同:
- ML/CV 论文:architecture / dataset / benchmark / ablation
- 理论论文:theorem / lemma / proof sketch / complexity
- 系统论文:system design / throughput / latency / fault tolerance
- 生物医学:cohort / endpoint / p-value / mechanism

默认的 `card.json` schema 是为 ML/具身智能论文设计的。处理其它领域时,需要 domain pack 来:
1. 覆盖/扩展 card schema 的字段
2. 定制 podcast.md 的段落模板(不同领域要讲的重点不同)
3. 定制 lint 规则(不同领域的口头禅不同)

## 设计:domain pack

一个 domain pack 是一个目录 `domains/<domain>/`,包含:

```
domains/<domain>/
  schema_extension.json   # 覆盖/新增 card.json 字段定义
  podcast_template.md     # 段落结构模板(带 {{slot}} 占位)
  lint_rules.py           # 额外的口头禅/句式规则
  examples/               # 该领域的范例 card + podcast
  README.md               # 该 domain 的解读约定
```

### schema_extension.json

```json
{
  "extends": "default",
  "add_fields": {
    "theorems": [
      {"name": "主定理名", "statement": "定理陈述", "proof_sketch": "证明思路", "assumptions": ["假设1"]}
    ],
    "complexity": {"time": "O(n log n)", "space": "O(n)", "note": ""}
  },
  "override_fields": {
    "architecture": "用 theorem / lemma / algorithm 结构代替 network architecture"
  }
}
```

### podcast_template.md

```
# {{title}}

{{date}}，{{authors}}。

这篇论文要解决的核心问题是：{{problem.what_they_address}}

## 主要结果

{{theorem_block}}

## 证明思路

{{proof_sketch_block}}

## 和已有工作的区别

{{novelty_vs_peers}}

{{limitations_block}}
```

模板里的 `{{slot}}` 在生成 podcast.md 时从 card.json 填充。不同领域用不同模板。

### lint_rules.py

```python
# 领域特定的口头禅。比如理论论文常滥用"优雅""深刻"
PATTERNS = [
    (r"优雅的?[^，。]{0,20}", "warn", "套话:优雅", "删掉或具体说哪里优雅"),
    (r"深刻地?揭示", "warn", "套话:深刻揭示", "删掉"),
]
```

## 内置 domain packs

初期只维护一个:
- `domains/ml_embodied/` — ML/具身智能(当前默认,覆盖 DreamZero 这类)

后续按需加:
- `domains/theory/` — 理论 CS
- `domains/systems/` — 系统论文
- `domains/bio/` — 生物医学

## 切换 domain

```bash
# 处理一篇新论文时指定 domain
python3 scripts/generate_audio.py papers/<slug> --domain ml_embodied
# 或在 card.json 里写 "domain": "theory",后续渲染/生成自动读
```

## 当前状态

- schema_extension / podcast_template / lint_rules 的接口已设计,实现待需要时补
- 当前所有论文走默认 ml_embodied pack
- 优先级:先把 ml_embodied 这个 pack 的多篇论文做扎实,再抽象
