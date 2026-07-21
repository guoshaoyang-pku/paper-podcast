#!/usr/bin/env python3
"""Render a paper card directory into a single-page HTML view.

Usage:
    python3 render_html.py out/<slug>

Reads:
    out/<slug>/card.json
    out/<slug>/architecture.md (optional)
    out/<slug>/podcast.md (optional)
    out/<slug>/source.pdf (optional, linked at top)

Writes:
    out/<slug>/index.html

The HTML is self-contained except for image references (relative paths to
extracted/<slug>/pages/*.png) and the Mermaid CDN (for architecture diagrams).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path


def slug_dirs(slug_dir: Path) -> dict[str, Path | None]:
    return {
        "card": slug_dir / "card.json",
        "arch": slug_dir / "architecture.md",
        "podcast": slug_dir / "podcast.md",
        "source_pdf": slug_dir / "source.pdf",
    }


def md_to_html(md: str) -> str:
    """Minimal markdown -> HTML. Handles headings, bold, code fences, tables,
    lists, and paragraphs. Good enough for our content; not a full MD parser.
    """
    if md is None:
        return ""
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    in_code = False
    code_lang = ""
    code_buf: list[str] = []
    while i < len(lines):
        line = lines[i]
        # code fence
        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_lang = line[3:].strip()
                code_buf = []
            else:
                in_code = False
                code_buf_str = "\n".join(code_buf)
                out.append(f'<pre class="code"><code>{escape(code_buf_str)}</code></pre>')
                code_buf = []
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        # heading
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            out.append(f"<h{level}>{escape(text)}</h{level}>")
            i += 1
            continue
        # table (very simple: | a | b |)
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|"):
            rows: list[str] = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            # rows[0] header, rows[1] separator, rows[2:] body
            def parse_row(r: str) -> list[str]:
                cells = [c.strip() for c in r.strip().strip("|").split("|")]
                return cells

            if len(rows) >= 2:
                header = parse_row(rows[0])
                body = [parse_row(r) for r in rows[2:]]
                out.append('<table class="md-table">')
                out.append("<thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in header) + "</tr></thead>")
                out.append("<tbody>")
                for r in body:
                    out.append("<tr>" + "".join(f"<td>{inline_md(c)}</td>" for c in r) + "</tr>")
                out.append("</tbody></table>")
            continue
        # bullet
        if line.startswith("- ") or line.startswith("* "):
            text = line[2:].strip()
            out.append(f"<ul><li>{inline_md(text)}</li></ul>")
            i += 1
            # collect continuation bullets
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("* ")):
                out.append(f"<li>{inline_md(lines[i][2:].strip())}</li>")
                i += 1
            out.append("</ul>")
            continue
        # empty
        if not line.strip():
            out.append("")
            i += 1
            continue
        # paragraph
        out.append(f"<p>{inline_md(line)}</p>")
        i += 1
    return "\n".join(out)


def inline_md(text: str) -> str:
    """Inline markdown: bold, code, links, images. Order matters."""
    # images ![alt](path)
    import re
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)",
                 lambda m: f'<img src="{escape(m.group(2))}" alt="{escape(m.group(1))}" />',
                 text)
    # links [text](path)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                 lambda m: f'<a href="{escape(m.group(2))}">{escape(m.group(1))}</a>',
                 text)
    # bold
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    # inline code
    text = re.sub(r"`([^`]+)`", lambda m: f"<code>{escape(m.group(1))}</code>", text)
    return text


def render_io(card: dict) -> str:
    io = card.get("inputs_outputs", {})
    rows = []
    rows.append("<h3>输入</h3><table class='md-table'><thead><tr><th>名称</th><th>类型</th><th>说明</th></tr></thead><tbody>")
    for inp in io.get("inputs", []):
        rows.append(f"<tr><td>{escape(inp.get('name',''))}</td><td>{escape(inp.get('type',''))}</td><td>{inline_md(inp.get('detail',''))}</td></tr>")
    rows.append("</tbody></table>")
    rows.append("<h3>输出</h3><table class='md-table'><thead><tr><th>名称</th><th>类型</th><th>说明</th></tr></thead><tbody>")
    for out in io.get("outputs", []):
        rows.append(f"<tr><td>{escape(out.get('name',''))}</td><td>{escape(out.get('type',''))}</td><td>{inline_md(out.get('detail',''))}</td></tr>")
    rows.append("</tbody></table>")
    if io.get("control_frequency"):
        rows.append(f"<p><strong>控制频率</strong>：{escape(io['control_frequency'])}</p>")
    # protocol
    if io.get("concat_protocol"):
        rows.append("<h3>输入拼接 protocol</h3>")
        # split into code block + explanation
        proto = io["concat_protocol"]
        lines = proto.split("\n")
        # first non-empty line is the sequence; rest is explanation
        seq_lines: list[str] = []
        explain_lines: list[str] = []
        in_explain = False
        for ln in lines:
            if not ln.strip():
                continue
            if ln.startswith("- ") or ln.startswith("注意"):
                in_explain = True
            if in_explain:
                explain_lines.append(ln)
            else:
                seq_lines.append(ln)
        if seq_lines:
            rows.append('<pre class="code protocol"><code>' + escape("\n".join(seq_lines)) + "</code></pre>")
        if explain_lines:
            rows.append("<div class='protocol-explain'>")
            for ln in explain_lines:
                if ln.startswith("- "):
                    rows.append(f"<p class='proto-item'>{inline_md(ln[2:])}</p>")
                else:
                    rows.append(f"<p class='proto-note'>{inline_md(ln)}</p>")
            rows.append("</div>")
    return "\n".join(rows)


def render_arch(card: dict) -> str:
    arch = card.get("architecture", {})
    parts = [f"<h3>主干与结构</h3>"]
    parts.append(f"<p><strong>backbone</strong>：{escape(arch.get('backbone',''))}</p>")
    parts.append(f"<p><strong>参数</strong>：{escape(arch.get('params',''))}</p>")
    parts.append(f"<p><strong>类型</strong>：{escape(arch.get('type',''))}</p>")
    if arch.get("key_components"):
        parts.append("<h4>关键组件</h4><ul>")
        for c in arch["key_components"]:
            parts.append(f"<li>{inline_md(c)}</li>")
        parts.append("</ul>")
    if arch.get("why_this_design"):
        parts.append(f"<h4>为什么这样设计</h4><p>{inline_md(arch['why_this_design'])}</p>")
    # numerical sense
    ns = arch.get("numerical_sense")
    if ns:
        parts.append("<h4>数值 sense</h4>")
        parts.append("<table class='md-table numerical'><thead><tr><th>项</th><th>值</th></tr></thead><tbody>")
        label_map = {
            "dit": "DiT 规格", "resolution": "分辨率", "vae": "VAE",
            "per_frame_latent_dim": "每帧 latent 维", "chunk": "Chunk",
            "context": "上下文", "action": "动作", "training": "训练",
        }
        for k, v in ns.items():
            label = label_map.get(k, k)
            parts.append(f"<tr><td>{escape(label)}</td><td>{inline_md(v)}</td></tr>")
        parts.append("</tbody></table>")
    return "\n".join(parts)


def render_figures(card: dict) -> str:
    figs = card.get("figures", [])
    if not figs:
        return "<p>无图。</p>"
    parts = ["<div class='figures-grid'>"]
    for f in figs:
        importance = f.get("importance", "supportive")
        badge_class = f"badge-{importance}"
        parts.append(f"""
        <div class='figure-card {badge_class}'>
          <div class='figure-head'>
            <span class='fig-id'>{escape(f.get('id',''))}</span>
            <span class='fig-page'>p.{escape(str(f.get('page','')))}</span>
            <span class='badge {badge_class}'>{escape(importance)}</span>
          </div>
          <h4>{escape(f.get('title',''))}</h4>
          <a href="{escape(f.get('image_path',''))}" target="_blank">
            <img src="{escape(f.get('image_path',''))}" alt="{escape(f.get('title',''))}" loading="lazy" />
          </a>
          <p class='caption'><strong>原文 caption</strong>：{escape(f.get('caption_original',''))}</p>
          <p class='whatitshows'>{inline_md(f.get('what_it_shows',''))}</p>
        </div>
        """)
    parts.append("</div>")
    return "\n".join(parts)


def render_results(card: dict) -> str:
    results = card.get("key_results", [])
    if not results:
        return ""
    rows = ["<table class='md-table results'><thead><tr><th>指标</th><th>值</th><th>最强 baseline</th><th>setup</th></tr></thead><tbody>"]
    for r in results:
        rows.append(f"<tr><td>{escape(r.get('metric',''))}</td><td><strong>{escape(r.get('value',''))}</strong></td><td>{escape(r.get('baseline_best',''))}</td><td>{escape(r.get('setup',''))}</td></tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def render_list(items: list[str], cls: str = "") -> str:
    if not items:
        return ""
    return f"<ul class='{cls}'>" + "".join(f"<li>{inline_md(i)}</li>" for i in items) + "</ul>"


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    slug_dir = Path(sys.argv[1]).resolve()
    paths = slug_dirs(slug_dir)
    if not paths["card"].exists():
        sys.exit(f"card.json not found in {slug_dir}")
    card = json.loads(paths["card"].read_text(encoding="utf-8"))

    arch_md = paths["arch"].read_text(encoding="utf-8") if paths["arch"] and paths["arch"].exists() else ""
    podcast_md = paths["podcast"].read_text(encoding="utf-8") if paths["podcast"] and paths["podcast"].exists() else ""

    # resolve image paths to be relative to the html file (which sits in out/<slug>/)
    # card.json image_path is like "extracted/dreamzero/pages/p01.png" — from out/<slug>/index.html
    # we need ../../extracted/...
    def fix_img_path(p: str) -> str:
        if not p:
            return p
        if p.startswith("http") or p.startswith("/"):
            return p
        # strip leading ./ if any
        p = p.lstrip("./")
        # already relative to the slug dir (figures/xxx.png) -> keep as is
        if p.startswith("figures/"):
            return p
        # legacy: extracted/<slug>/pages/xx.png -> resolve relative to slug dir
        # (papers/<slug>/index.html and extracted/ are siblings of out/<slug>/ in
        # the survey_podcast project layout)
        return "../../" + p

    # deep-copy figures with fixed paths
    figs_fixed = []
    for f in card.get("figures", []):
        f2 = dict(f)
        f2["image_path"] = fix_img_path(f2.get("image_path", ""))
        figs_fixed.append(f2)
    card["figures"] = figs_fixed

    # build html
    html_parts = []
    html_parts.append(f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(card.get('title',''))}</title>
<style>
:root {{
  --bg:#fafbfa; --paper:#fff; --ink:#17201c; --muted:#65706b;
  --faint:#8b9690; --line:#dfe6e2; --accent:#0f766e; --accent2:#8a5a00;
  --soft:#eaf5f2; --warn:#fff7e6; --danger:#9f2f2f; --dark:#111614;
  --code-bg:#1e2622; --code-fg:#e8f1ee;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",Arial,sans-serif;
  line-height:1.7; }}
header.top {{ background:var(--dark); color:#e8f1ee; padding:28px 40px; }}
header.top .back-home {{ display:inline-block; color:#5eead4; text-decoration:none;
  font-size:14px; font-weight:600; margin-bottom:14px; }}
header.top .back-home:hover {{ text-decoration:underline; }}
header.top h1 {{ margin:0 0 8px; font-size:26px; line-height:1.3; }}
header.top .meta {{ color:#9fb3ac; font-size:14px; }}
header.top .meta a {{ color:#5eead4; }}
header.top .links {{ margin-top:14px; display:flex; gap:18px; flex-wrap:wrap; }}
header.top .links a {{ color:#5eead4; text-decoration:none; font-size:14px;
  border:1px solid #2a3934; padding:6px 12px; border-radius:6px; }}
header.top .links a:hover {{ background:#1e2622; }}
nav.tabs {{ background:var(--paper); border-bottom:1px solid var(--line);
  padding:0 40px; position:sticky; top:0; z-index:10; display:flex; gap:4px; }}
nav.tabs button {{ background:none; border:none; border-bottom:3px solid transparent;
  padding:16px 18px; font-size:15px; color:var(--muted); cursor:pointer; font-weight:600; }}
nav.tabs button.active {{ color:var(--accent); border-bottom-color:var(--accent); }}
main {{ max-width:1100px; margin:0 auto; padding:32px 40px; }}
.tab-panel {{ display:none; }}
.tab-panel.active {{ display:block; }}
h2 {{ border-bottom:2px solid var(--line); padding-bottom:8px; margin-top:36px; }}
h3 {{ margin-top:28px; }}
h4 {{ margin-top:20px; color:var(--accent); }}
table.md-table {{ border-collapse:collapse; width:100%; margin:14px 0; font-size:14px; }}
table.md-table th, table.md-table td {{ border:1px solid var(--line); padding:10px 12px; text-align:left; vertical-align:top; }}
table.md-table th {{ background:var(--soft); color:var(--accent); font-weight:700; }}
table.md-table.numerical td:first-child {{ font-weight:600; width:140px; color:var(--muted); }}
table.md-table.results td:nth-child(2) {{ color:var(--accent); }}
pre.code {{ background:var(--code-bg); color:var(--code-fg); padding:16px 18px;
  border-radius:8px; overflow-x:auto; font-size:13px; line-height:1.55;
  font-family:"SF Mono",Menlo,Consolas,monospace; }}
pre.code.protocol {{ border-left:4px solid var(--accent); }}
.protocol-explain {{ margin-top:12px; }}
.protocol-explain .proto-item {{ margin:6px 0; padding-left:14px; border-left:2px solid var(--line); }}
.protocol-explain .proto-note {{ color:var(--muted); font-size:14px; }}
.figures-grid {{ display:grid; grid-template-columns:1fr; gap:28px; margin-top:20px; }}
.figure-card {{ background:var(--paper); border:1px solid var(--line); border-radius:10px; padding:20px; }}
.figure-card.badge-key {{ border-left:4px solid var(--accent); }}
.figure-card.badge-supportive {{ border-left:4px solid var(--faint); }}
.figure-head {{ display:flex; align-items:center; gap:12px; margin-bottom:8px; font-size:13px; }}
.fig-id {{ font-weight:700; color:var(--accent); }}
.fig-page {{ color:var(--muted); }}
.badge {{ padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; text-transform:uppercase; }}
.badge-key {{ background:var(--soft); color:var(--accent); }}
.badge-supportive {{ background:#f0f2f1; color:var(--muted); }}
.figure-card img {{ max-width:100%; height:auto; border:1px solid var(--line); border-radius:6px; margin:10px 0; }}
.figure-card .caption {{ color:var(--muted); font-size:13px; }}
.figure-card .whatitshows {{ margin-top:10px; }}
.arch-prose {{ max-width:none; }}
.arch-prose pre.code {{ margin:16px 0; }}
.podcast-prose {{ max-width:780px; }}
.podcast-prose h2 {{ color:var(--accent); }}
.podcast-prose pre.code {{ background:#fff; color:var(--ink); border:1px solid var(--line); }}
.audio-player {{ background:var(--soft); border:1px solid var(--line); border-radius:10px;
  padding:18px 22px; margin-bottom:24px; }}
.audio-player h3 {{ margin:0 0 12px; color:var(--accent); }}
.audio-player audio {{ display:block; }}
.audio-player .dur {{ margin:8px 0 0; color:var(--muted); font-size:13px; }}
ul.insights li {{ margin:8px 0; }}
ul.limitations li {{ color:var(--muted); }}
.footer {{ text-align:center; color:var(--faint); padding:40px 20px; font-size:13px; }}
@media (max-width:780px) {{
  main {{ padding:20px; }}
  header.top {{ padding:20px; }}
  nav.tabs {{ padding:0 12px; overflow-x:auto; }}
}}
</style>
</head>
<body>
<header class="top">
  <a class="back-home" href="../index.html" title="返回首页">← Home</a>
  <h1>{escape(card.get('title',''))}</h1>
  <div class="meta">
    {escape('、'.join(card.get('authors',[])[:3]) + (' et al.' if len(card.get('authors',[]))>3 else ''))} · {escape(card.get('affiliation',''))} · {escape(card.get('date',''))} · arXiv:{escape(card.get('arxiv',''))}
  </div>
  <div class="links">
""")
    if paths["source_pdf"] and paths["source_pdf"].exists():
        html_parts.append(f'    <a href="source.pdf" target="_blank">📄 source.pdf</a>')
    if card.get("reproducibility", {}).get("code"):
        html_parts.append(f'    <a href="{escape(card["reproducibility"]["code"])}" target="_blank">🔗 code</a>')
    html_parts.append(f'    <a href="../../SCHEMA.md" target="_blank">📋 schema</a>')
    html_parts.append("""  </div>
</header>
<nav class="tabs">
  <button class="tab-btn active" data-tab="overview">Overview</button>
  <button class="tab-btn" data-tab="architecture">Architecture</button>
  <button class="tab-btn" data-tab="figures">Figures</button>
  <button class="tab-btn" data-tab="podcast">Podcast</button>
</nav>
<main>
""")

    # Overview panel
    html_parts.append('<section class="tab-panel active" id="overview">')
    html_parts.append(f"<p class='one-liner'>{inline_md(card.get('one_liner',''))}</p>")
    # problem
    prob = card.get("problem", {})
    html_parts.append("<h2>问题</h2>")
    html_parts.append(f"<p><strong>要解决什么</strong>：{inline_md(prob.get('what_they_address',''))}</p>")
    html_parts.append(f"<p><strong>为什么 prior work 不够</strong>：{inline_md(prob.get('why_prior_work_falls_short',''))}</p>")
    # io
    html_parts.append("<h2>输入 / 输出</h2>")
    html_parts.append(render_io(card))
    # datasets
    html_parts.append("<h2>数据集</h2>")
    html_parts.append("<table class='md-table'><thead><tr><th>数据</th><th>规模</th><th>备注</th></tr></thead><tbody>")
    for d in card.get("datasets", []):
        html_parts.append(f"<tr><td>{escape(d.get('name',''))}</td><td>{escape(d.get('size',''))}</td><td>{inline_md(d.get('note',''))}</td></tr>")
    html_parts.append("</tbody></table>")
    # architecture summary
    html_parts.append("<h2>架构（摘要）</h2>")
    html_parts.append(render_arch(card))
    html_parts.append("<p>→ 详见 <strong>Architecture</strong> tab。</p>")
    # key results
    html_parts.append("<h2>关键结果</h2>")
    html_parts.append(render_results(card))
    # insights
    html_parts.append("<h2>Insights</h2>")
    html_parts.append(render_list(card.get("insights", []), "insights"))
    # novelty
    html_parts.append("<h2>vs 同类工作</h2>")
    html_parts.append(render_list(card.get("novelty_vs_peers", [])))
    # limitations
    html_parts.append("<h2>局限</h2>")
    html_parts.append(render_list(card.get("limitations", []), "limitations"))
    # reproducibility
    rep = card.get("reproducibility", {})
    if rep:
        html_parts.append("<h2>可复现性</h2><ul>")
        for k, v in rep.items():
            html_parts.append(f"<li><strong>{escape(k)}</strong>：{inline_md(str(v))}</li>")
        html_parts.append("</ul>")
    html_parts.append('<div class="tags">' + " ".join(f"<span class='tag'>{escape(t)}</span>" for t in card.get("tags", [])) + "</div>")
    html_parts.append("</section>")

    # Architecture panel
    html_parts.append('<section class="tab-panel" id="architecture">')
    html_parts.append('<div class="arch-prose">')
    if arch_md:
        html_parts.append(md_to_html(arch_md))
    else:
        html_parts.append("<p>architecture.md 未找到。</p>")
    html_parts.append("</div>")
    # include mermaid CDN so ```mermaid blocks render (our md_to_html doesn't handle them; we pass them through as code)
    html_parts.append("""
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: true, theme: "default" });
</script>
""")
    html_parts.append("</section>")

    # Figures panel
    html_parts.append('<section class="tab-panel" id="figures">')
    html_parts.append(render_figures(card))
    html_parts.append("</section>")

    # Podcast panel
    html_parts.append('<section class="tab-panel" id="podcast">')
    # audio player if mp3 exists
    audio_path = slug_dir / "audio" / "podcast.mp3"
    if audio_path.exists():
        dur_label = ""
        ff = shutil.which("ffprobe")
        if ff:
            try:
                r = subprocess.run([ff, "-v","error","-show_entries","format=duration",
                                    "-of","default=noprint_wrappers=1:nokey=1",
                                    audio_path.as_posix()], capture_output=True, text=True)
                dur = float(r.stdout.strip())
                dur_label = f"{int(dur//60)}:{int(dur%60):02d}"
            except Exception:
                pass
        html_parts.append(f"""
<div class="audio-player">
  <h3>🎧 音频版</h3>
  <audio controls preload="metadata" style="width:100%;max-width:560px;">
    <source src="audio/podcast.mp3" type="audio/mpeg">
    你的浏览器不支持 audio 元素。<a href="audio/podcast.mp3">下载 MP3</a>
  </audio>
  {f'<p class="dur">时长 {dur_label} · Edge TTS</p>' if dur_label else ''}
</div>
""")
    html_parts.append('<div class="podcast-prose">')
    if podcast_md:
        html_parts.append(md_to_html(podcast_md))
    else:
        html_parts.append("<p>podcast.md 未找到。</p>")
    html_parts.append("</div>")
    html_parts.append("</section>")

    html_parts.append("""
</main>
<div class="footer">generated by paper-podcast skill · card.json 是真相源，其它文件是其视图</div>
<script>
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
      window.scrollTo({top:0, behavior:'smooth'});
    });
  });
</script>
</body>
</html>
""")
    out_path = slug_dir / "index.html"
    out_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"OK -> {out_path}  ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
