#!/usr/bin/env python3
"""Build a mobile-first podcast site: episode list + RSS feed.

Usage:
    python3 scripts/build_podcast_site.py

Produces:
    docs/index.html  — episode list with inline players + sticky mini-player
    docs/feed.xml     — podcast RSS 2.0 feed (Apple Podcasts / 小宇宙 compatible)

Reads each papers/<slug>/card.json + audio/podcast.mp3.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers"
DOCS = ROOT / "docs"

# Base URL of the GitHub Pages site. Change if custom domain.
BASE_URL = "https://guoshaoyang-pku.github.io/paper-podcast"


def probe_duration(path: Path) -> float:
    ff = shutil.which("ffprobe")
    if not ff:
        return 0.0
    try:
        r = subprocess.run([ff, "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", path.as_posix()],
                           capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def fmt_dur(sec: float) -> str:
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"


def iso_date(d: str) -> str:
    """'2026-02-19' -> RFC 822-ish for RSS pubDate."""
    try:
        dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.strftime("%a, %d %b %Y %H:%M:%S +0000")
    except Exception:
        return datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")


def load_episodes() -> list[dict]:
    eps = []
    for card_path in sorted(PAPERS.glob("*/card.json")):
        c = json.loads(card_path.read_text(encoding="utf-8"))
        slug = c.get("slug", card_path.parent.name)
        audio = card_path.parent / "audio" / "podcast.mp3"
        dur = probe_duration(audio) if audio.exists() else 0.0
        # find a key figure as cover
        cover = next((f["image_path"] for f in c.get("figures", [])
                       if f.get("importance") == "key"), "figures/p01.png")
        # resolve cover relative to slug dir
        if not cover.startswith("figures/"):
            cover = "figures/p01.png"
        eps.append({
            "slug": slug,
            "title": c.get("title", slug),
            "authors": ", ".join(c.get("authors", [])[:2]) + (" et al." if len(c.get("authors", [])) > 2 else ""),
            "affiliation": c.get("affiliation", ""),
            "date": c.get("date", ""),
            "arxiv": c.get("arxiv", ""),
            "one_liner": c.get("one_liner", ""),
            "category": c.get("category", ""),
            "tags": c.get("tags", []),
            "has_audio": audio.exists(),
            "duration_sec": dur,
            "duration_str": fmt_dur(dur),
            "cover": cover,
        })
    # newest first
    eps.sort(key=lambda e: e["date"], reverse=True)
    return eps


def build_index_html(eps: list[dict]) -> str:
    cards_html = []
    for e in eps:
        play_block = ""
        if e["has_audio"]:
            play_block = f"""
        <button class="play-btn" data-slug="{e['slug']}"
                data-title="{escape(e['title'])}"
                data-src="{e['slug']}/audio/podcast.mp3"
                data-dur="{e['duration_str']}">
          <span class="play-icon">▶</span>
          <span class="play-label">播放 {e['duration_str']}</span>
        </button>"""
        else:
            play_block = '<div class="no-audio">音频生成中</div>'

        tags = "".join(f"<span class='tag'>{escape(t)}</span>" for t in e["tags"][:3])
        cards_html.append(f"""
      <article class="episode" data-slug="{e['slug']}">
        <a class="ep-cover" href="{e['slug']}/index.html">
          <img src="{e['slug']}/{e['cover']}" alt="cover" loading="lazy"/>
          <span class="cat-badge">{escape(e['category'])}</span>
        </a>
        <div class="ep-body">
          <div class="ep-meta">
            <span class="ep-date">{escape(e['date'])}</span> · <span class="ep-aff">{escape(e['affiliation'])}</span>
          </div>
          <h2><a href="{e['slug']}/index.html">{escape(e['title'])}</a></h2>
          <p class="ep-one">{escape(e['one_liner'])}</p>
          <div class="ep-actions">
            {play_block}
            <a class="read-link" href="{e['slug']}/index.html">读全文 →</a>
          </div>
          <div class="ep-tags">{tags}</div>
        </div>
      </article>
""")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#111614">
<title>Paper Podcast — 论文深度解读</title>
<link rel="alternate" type="application/rss+xml" title="Paper Podcast" href="feed.xml" />
<style>
:root {{
  --bg:#fafbfa; --paper:#fff; --ink:#17201c; --muted:#65706b;
  --line:#dfe6e2; --accent:#0f766e; --soft:#eaf5f2; --dark:#111614;
}}
* {{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }}
html,body {{ margin:0; padding:0; }}
body {{
  background:var(--bg); color:var(--ink); line-height:1.6;
  font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
  padding-bottom:90px;  /* room for sticky player */
  padding-bottom:calc(90px + env(safe-area-inset-bottom,0));
}}
header.hero {{
  background:linear-gradient(135deg,#111614,#0f766e);
  color:#e8f1ee; padding:48px 20px 36px; text-align:center;
}}
header.hero h1 {{ margin:0 0 8px; font-size:26px; letter-spacing:-0.02em; }}
header.hero p {{ color:#9fb3ac; margin:6px auto 16px; font-size:14px; max-width:520px; }}
.subscribe {{
  display:inline-flex; gap:10px; flex-wrap:wrap; justify-content:center;
}}
.subscribe a {{
  color:#5eead4; text-decoration:none; font-size:13px;
  border:1px solid #2a3934; padding:8px 14px; border-radius:20px;
  background:rgba(255,255,255,0.04);
}}
.subscribe a:hover {{ background:rgba(94,234,212,0.1); }}
main {{ max-width:680px; margin:0 auto; padding:24px 16px 60px; }}
.episode {{
  display:flex; gap:14px; background:var(--paper); border:1px solid var(--line);
  border-radius:14px; padding:14px; margin-bottom:16px;
  transition:box-shadow .15s;
}}
.episode:hover {{ box-shadow:0 4px 14px rgba(0,0,0,0.06); }}
.ep-cover {{
  flex-shrink:0; width:84px; height:84px; border-radius:10px; overflow:hidden;
  position:relative; display:block;
}}
.ep-cover img {{ width:100%; height:100%; object-fit:cover; }}
.cat-badge {{
  position:absolute; bottom:4px; left:4px; background:rgba(15,118,110,0.92);
  color:#fff; font-size:9px; font-weight:700; padding:2px 6px; border-radius:3px;
  text-transform:uppercase; letter-spacing:0.04em;
}}
.ep-body {{ flex:1; min-width:0; }}
.ep-meta {{ font-size:12px; color:var(--muted); margin-bottom:4px; }}
.ep-body h2 {{ margin:0 0 6px; font-size:16px; line-height:1.35; }}
.ep-body h2 a {{ color:var(--ink); text-decoration:none; }}
.ep-body h2 a:hover {{ color:var(--accent); }}
.ep-one {{
  margin:0 0 10px; font-size:13px; color:var(--muted); line-height:1.5;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
}}
.ep-actions {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; }}
.play-btn {{
  background:var(--accent); color:#fff; border:none; cursor:pointer;
  padding:8px 14px; border-radius:20px; font-size:13px; font-weight:600;
  display:inline-flex; align-items:center; gap:6px; font-family:inherit;
}}
.play-btn:hover {{ background:#0c5f57; }}
.play-btn.playing {{ background:#9f2f2f; }}
.play-icon {{ font-size:10px; }}
.read-link {{ color:var(--accent); text-decoration:none; font-size:13px; font-weight:500; }}
.read-link:hover {{ text-decoration:underline; }}
.ep-tags {{ margin-top:8px; }}
.tag {{
  display:inline-block; background:#f0f2f1; color:var(--muted);
  font-size:10px; padding:2px 7px; border-radius:3px; margin-right:4px;
}}
.no-audio {{ color:var(--muted); font-size:12px; font-style:italic; }}

/* sticky mini player */
.sticky-player {{
  position:fixed; bottom:0; left:0; right:0; z-index:100;
  background:var(--dark); color:#e8f1ee;
  padding:10px 16px; padding-bottom:calc(10px + env(safe-area-inset-bottom,0));
  box-shadow:0 -4px 16px rgba(0,0,0,0.15);
  display:none; align-items:center; gap:12px;
}}
.sticky-player.show {{ display:flex; }}
.sticky-player .sp-play {{
  background:#5eead4; color:#111614; border:none; width:40px; height:40px;
  border-radius:50%; font-size:14px; cursor:pointer; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
}}
.sticky-player .sp-info {{ flex:1; min-width:0; }}
.sticky-player .sp-title {{
  font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; margin-bottom:4px;
}}
.sticky-player .sp-progress {{
  height:3px; background:rgba(255,255,255,0.15); border-radius:2px; overflow:hidden;
}}
.sticky-player .sp-progress-bar {{
  height:100%; background:#5eead4; width:0%; transition:width .1s linear;
}}
.sticky-player .sp-time {{ font-size:11px; color:#9fb3ac; flex-shrink:0; }}
.sticky-player .sp-close {{
  background:none; border:none; color:#9fb3ac; cursor:pointer;
  font-size:18px; padding:0 4px;
}}
audio.hidden-src {{ display:none; }}

.footer {{ text-align:center; color:var(--muted); padding:20px; font-size:12px; }}
.footer a {{ color:var(--accent); text-decoration:none; }}

@media (max-width:480px) {{
  header.hero {{ padding:36px 16px 28px; }}
  header.hero h1 {{ font-size:22px; }}
  .episode {{ padding:12px; gap:10px; }}
  .ep-cover {{ width:68px; height:68px; }}
  .ep-body h2 {{ font-size:15px; }}
}}
</style>
</head>
<body>
<header class="hero">
  <h1>Paper Podcast</h1>
  <p>研究论文深度解读。每篇 15 分钟口播，听完脑子里能建出论文架构。</p>
  <div class="subscribe">
    <a href="feed.xml">📡 RSS 订阅</a>
    <a href="https://podcasts.apple.com" target="_blank">Apple Podcasts</a>
    <a href="https://github.com/guoshaoyang-pku/paper-podcast" target="_blank">GitHub</a>
  </div>
</header>
<main>
{''.join(cards_html)}
</main>

<div class="sticky-player" id="stickyPlayer">
  <button class="sp-play" id="spPlay">▶</button>
  <div class="sp-info">
    <div class="sp-title" id="spTitle">—</div>
    <div class="sp-progress"><div class="sp-progress-bar" id="spBar"></div></div>
  </div>
  <span class="sp-time" id="spTime">0:00</span>
  <button class="sp-close" id="spClose">✕</button>
</div>
<audio class="hidden-src" id="hiddenAudio" preload="metadata"></audio>

<div class="footer">
  generated by <a href="https://github.com/guoshaoyang-pku/paper-podcast">paper-podcast</a> ·
  <a href="feed.xml">RSS feed</a>
</div>

<script>
(function(){{
  const audio = document.getElementById('hiddenAudio');
  const player = document.getElementById('stickyPlayer');
  const playBtn = document.getElementById('spPlay');
  const spTitle = document.getElementById('spTitle');
  const spBar = document.getElementById('spBar');
  const spTime = document.getElementById('spTime');
  const spClose = document.getElementById('spClose');
  let currentSlug = null;

  function fmt(s){{
    s=Math.floor(s); const m=Math.floor(s/60); const r=s%60;
    return m+':'+(r<10?'0':'')+r;
  }}

  document.querySelectorAll('.play-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const slug = btn.dataset.slug;
      const src = btn.dataset.src;
      const title = btn.dataset.title;
      const dur = btn.dataset.dur;
      if (currentSlug === slug && !audio.paused) {{
        audio.pause();
        return;
      }}
      audio.src = src;
      spTitle.textContent = title;
      player.classList.add('show');
      currentSlug = slug;
      // reset all btn states
      document.querySelectorAll('.play-btn').forEach(b=>b.classList.remove('playing'));
      document.querySelectorAll('.play-btn').forEach(b=>{{
        if(b.dataset.slug===slug) {{ b.classList.add('playing'); b.querySelector('.play-icon').textContent='⏸'; }}
        else {{ b.querySelector('.play-icon').textContent='▶'; }}
      }});
      audio.play();
    }});
  }});

  playBtn.addEventListener('click', () => {{
    if (audio.paused) audio.play(); else audio.pause();
  }});

  spClose.addEventListener('click', () => {{
    audio.pause(); audio.currentTime=0;
    player.classList.remove('show');
    document.querySelectorAll('.play-btn').forEach(b=>{{
      b.classList.remove('playing'); b.querySelector('.play-icon').textContent='▶';
    }});
    currentSlug = null;
  }});

  audio.addEventListener('play', () => {{ playBtn.textContent='⏸'; }});
  audio.addEventListener('pause', () => {{ playBtn.textContent='▶'; }});
  audio.addEventListener('timeupdate', () => {{
    if (audio.duration) {{
      spBar.style.width = (audio.currentTime/audio.duration*100) + '%';
      spTime.textContent = fmt(audio.currentTime) + ' / ' + fmt(audio.duration);
    }}
  }});
}})();
</script>
</body>
</html>
"""


def build_rss(eps: list[dict]) -> str:
    """Podcast RSS 2.0 feed."""
    items = []
    for e in eps:
        if not e["has_audio"]:
            continue
        audio_url = f"{BASE_URL}/{e['slug']}/audio/podcast.mp3"
        page_url = f"{BASE_URL}/{e['slug']}/index.html"
        # get audio size
        audio_path = PAPERS / e["slug"] / "audio" / "podcast.mp3"
        size = audio_path.stat().st_size if audio_path.exists() else 0
        dur_min = int(e["duration_sec"] // 60)
        pub = iso_date(e["date"])
        # description: one-liner + link
        desc = f"{e['one_liner']} 论文解读全文见 {page_url}"
        items.append(f"""    <item>
      <title>{xml_escape(e['title'])}</title>
      <description>{xml_escape(desc)}</description>
      <pubDate>{pub}</pubDate>
      <guid isPermaLink="true">{audio_url}</guid>
      <enclosure url="{audio_url}" length="{size}" type="audio/mpeg"/>
      <itunes:duration>{e['duration_str']}</itunes:duration>
      <itunes:summary>{xml_escape(e['one_liner'])}</itunes:summary>
      <link>{page_url}</link>
    </item>""")

    last_build = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Paper Podcast</title>
    <link>{BASE_URL}/</link>
    <language>zh-cn</language>
    <description>研究论文深度解读。每篇 15 分钟口播，听完脑子里能建出论文架构。</description>
    <itunes:author>Paper Podcast</itunes:author>
    <itunes:summary>研究论文深度解读。每篇 15 分钟口播，听完脑子里能建出论文架构。</itunes:summary>
    <itunes:category text="Technology"/>
    <itunes:category text="Science"/>
    <itunes:explicit>false</itunes:explicit>
    <itunes:image href="{BASE_URL}/dreamzero/figures/p01.png"/>
    <image>
      <url>{BASE_URL}/dreamzero/figures/p01.png</url>
      <title>Paper Podcast</title>
      <link>{BASE_URL}/</link>
    </image>
    <lastBuildDate>{last_build}</lastBuildDate>
{''.join(items)}
  </channel>
</rss>
"""
    return rss


def main() -> None:
    eps = load_episodes()
    print(f"loaded {len(eps)} episodes ({sum(1 for e in eps if e['has_audio'])} with audio)")

    html = build_index_html(eps)
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    print(f"-> {DOCS/'index.html'}")

    rss = build_rss(eps)
    (DOCS / "feed.xml").write_text(rss, encoding="utf-8")
    print(f"-> {DOCS/'feed.xml'}")


if __name__ == "__main__":
    main()
