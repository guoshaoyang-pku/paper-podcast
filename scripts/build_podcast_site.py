#!/usr/bin/env python3
"""Build a mobile-first podcast SPA: inline player + playlist + queue.

Usage:
    python3 scripts/build_podcast_site.py

Produces:
    docs/index.html  — SPA: episode cards expand inline, sticky player,
                       loop toggle, queue management, Media Session API
    docs/feed.xml     — podcast RSS 2.0 feed

Reads each papers/<slug>/card.json + audio/podcast.mp3 + podcast.md.
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
        cover = next((f["image_path"] for f in c.get("figures", [])
                       if f.get("importance") == "key"), "figures/p01.png")
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
    eps.sort(key=lambda e: e["date"], reverse=True)
    return eps


def build_index_html(eps: list[dict]) -> str:
    eps_json = json.dumps(eps, ensure_ascii=False)

    cards = []
    for e in eps:
        tags = "".join(f"<span class='tag'>{escape(t)}</span>" for t in e["tags"][:3])
        play_label = f"播放 {e['duration_str']}" if e["has_audio"] else "音频生成中"
        cards.append(f"""
      <article class="episode" data-slug="{e['slug']}" data-date="{e['date']}">
        <div class="ep-row">
          <a class="ep-cover" href="{e['slug']}/index.html" onclick="event.stopPropagation()">
            <img src="{e['slug']}/{e['cover']}" alt="cover" loading="lazy"/>
            <span class="cat-badge">{escape(e['category'])}</span>
          </a>
          <div class="ep-body">
            <div class="ep-meta"><span>{escape(e['date'])}</span> &middot; <span>{escape(e['affiliation'])}</span></div>
            <h2><a href="{e['slug']}/index.html" onclick="event.stopPropagation()">{escape(e['title'])}</a></h2>
            <p class="ep-one">{escape(e['one_liner'])}</p>
            <div class="ep-actions">
              <button class="play-btn" data-slug="{e['slug']}" data-dur="{e['duration_str']}">
                <span class="play-icon"></span><span class="play-label">{play_label}</span>
              </button>
              <button class="queue-btn" data-slug="{e['slug']}" title="加入歌单">＋歌单</button>
              <a class="read-link" href="{e['slug']}/index.html" onclick="event.stopPropagation()">读全文 &rarr;</a>
            </div>
            <div class="ep-tags">{tags}</div>
          </div>
        </div>
        <div class="ep-text" id="text-{e['slug']}"></div>
      </article>
""")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#111614">
<title>Paper Podcast &mdash; 论文深度解读</title>
<link rel="alternate" type="application/rss+xml" title="Paper Podcast" href="feed.xml" />
<style>
:root {{
  --bg:#fafbfa; --paper:#fff; --ink:#17201c; --muted:#65706b;
  --line:#dfe6e2; --accent:#0f766e; --soft:#eaf5f2; --dark:#111614;
  --danger:#9f2f2f; --warn:#8a5a00;
}}
* {{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }}
html,body {{ margin:0; padding:0; }}
body {{
  background:var(--bg); color:var(--ink); line-height:1.6;
  font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
  padding-bottom:100px;
  padding-bottom:calc(100px + env(safe-area-inset-bottom,0));
}}
header.hero {{
  background:linear-gradient(135deg,#111614,#0f766e);
  color:#e8f1ee; padding:40px 20px 28px; text-align:center;
}}
header.hero h1 {{ margin:0 0 6px; font-size:24px; }}
header.hero p {{ color:#9fb3ac; margin:4px auto 14px; font-size:13px; max-width:480px; }}
.subscribe {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:center; }}
.subscribe a {{
  color:#5eead4; text-decoration:none; font-size:12px;
  border:1px solid #2a3934; padding:6px 12px; border-radius:16px;
  background:rgba(255,255,255,0.04);
}}

nav.tabs {{
  display:flex; background:var(--paper); border-bottom:1px solid var(--line);
  position:sticky; top:0; z-index:20;
}}
nav.tabs button {{
  flex:1; background:none; border:none; border-bottom:3px solid transparent;
  padding:14px; font-size:14px; color:var(--muted); cursor:pointer; font-weight:600;
  font-family:inherit;
}}
nav.tabs button.active {{ color:var(--accent); border-bottom-color:var(--accent); }}
nav.tabs .queue-count {{
  display:inline-block; background:var(--accent); color:#fff;
  font-size:10px; border-radius:8px; padding:1px 6px; margin-left:4px; min-width:16px;
}}

main {{ max-width:680px; margin:0 auto; padding:16px 12px 60px; }}

.episode {{
  background:var(--paper); border:1px solid var(--line); border-radius:12px;
  margin-bottom:12px; overflow:hidden; cursor:pointer;
  transition:box-shadow .15s;
}}
.episode:hover {{ box-shadow:0 4px 12px rgba(0,0,0,0.06); }}
.episode.playing {{ border-left:3px solid var(--accent); }}
.ep-row {{ display:flex; gap:12px; padding:12px; }}
.ep-cover {{
  flex-shrink:0; width:72px; height:72px; border-radius:8px; overflow:hidden;
  position:relative; display:block;
}}
.ep-cover img {{ width:100%; height:100%; object-fit:cover; }}
.cat-badge {{
  position:absolute; bottom:3px; left:3px; background:rgba(15,118,110,0.92);
  color:#fff; font-size:8px; font-weight:700; padding:1px 5px; border-radius:3px;
  text-transform:uppercase;
}}
.ep-body {{ flex:1; min-width:0; }}
.ep-meta {{ font-size:11px; color:var(--muted); margin-bottom:3px; }}
.ep-body h2 {{ margin:0 0 4px; font-size:15px; line-height:1.3; }}
.ep-body h2 a {{ color:var(--ink); text-decoration:none; }}
.ep-body h2 a:hover {{ color:var(--accent); }}
.ep-one {{
  margin:0 0 8px; font-size:12px; color:var(--muted); line-height:1.5;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
}}
.ep-actions {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
.play-btn {{
  background:var(--accent); color:#fff; border:none; cursor:pointer;
  padding:6px 12px; border-radius:16px; font-size:12px; font-weight:600;
  display:inline-flex; align-items:center; gap:4px; font-family:inherit;
}}
.play-btn:hover {{ background:#0c5f57; }}
.play-btn.playing {{ background:var(--danger); }}
.play-btn.playing .play-icon::before {{ content:'\\23F8'; }}
.play-btn:not(.playing) .play-icon::before {{ content:'\\25B6'; }}
.play-icon {{ font-size:9px; }}
.queue-btn {{
  background:none; border:1px solid var(--line); color:var(--muted);
  cursor:pointer; padding:5px 10px; border-radius:16px; font-size:11px;
  font-family:inherit; white-space:nowrap;
}}
.queue-btn:hover {{ border-color:var(--accent); color:var(--accent); }}
.queue-btn.in-queue {{ background:var(--soft); border-color:var(--accent); color:var(--accent); }}
.read-link {{ color:var(--accent); text-decoration:none; font-size:12px; font-weight:500; }}
.read-link:hover {{ text-decoration:underline; }}
.ep-tags {{ margin-top:6px; }}
.tag {{
  display:inline-block; background:#f0f2f1; color:var(--muted);
  font-size:9px; padding:1px 6px; border-radius:3px; margin-right:3px;
}}
.ep-text {{
  display:none; padding:0 16px 14px; border-top:1px solid var(--line);
  margin-top:0; font-size:14px; max-height:400px; overflow-y:auto; line-height:1.8;
}}
.ep-text.open {{ display:block; }}
.ep-text h1,.ep-text h2,.ep-text h3 {{ font-size:15px; margin:12px 0 6px; color:var(--accent); }}
.ep-text p {{ margin:6px 0; }}
.ep-text .loading {{ color:var(--muted); font-style:italic; }}

.queue-panel {{ display:none; }}
.queue-panel.active {{ display:block; }}
.queue-empty {{ text-align:center; color:var(--muted); padding:60px 20px; font-size:14px; }}
.queue-item {{
  display:flex; align-items:center; gap:10px; background:var(--paper);
  border:1px solid var(--line); border-radius:10px; padding:10px 12px; margin-bottom:8px;
}}
.queue-item img {{ width:40px; height:40px; border-radius:6px; object-fit:cover; flex-shrink:0; }}
.queue-item .qi-body {{ flex:1; min-width:0; }}
.queue-item .qi-title {{ font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.queue-item .qi-meta {{ font-size:11px; color:var(--muted); }}
.queue-item .qi-controls {{ display:flex; gap:4px; }}
.queue-item .qi-btn {{
  background:none; border:1px solid var(--line); cursor:pointer;
  width:28px; height:28px; border-radius:6px; font-size:12px; color:var(--muted);
  display:flex; align-items:center; justify-content:center;
}}
.queue-item .qi-btn:hover {{ border-color:var(--accent); color:var(--accent); }}
.queue-item .qi-btn:disabled {{ opacity:0.3; cursor:default; }}
.queue-item .qi-btn.danger:hover {{ border-color:var(--danger); color:var(--danger); }}

.sticky-player {{
  position:fixed; bottom:0; left:0; right:0; z-index:100;
  background:var(--dark); color:#e8f1ee;
  padding:10px 14px; padding-bottom:calc(10px + env(safe-area-inset-bottom,0));
  box-shadow:0 -4px 16px rgba(0,0,0,0.2);
  display:none; align-items:center; gap:10px;
}}
.sticky-player.show {{ display:flex; }}
.sp-controls {{ display:flex; align-items:center; gap:6px; flex-shrink:0; }}
.sp-btn {{
  background:rgba(255,255,255,0.1); color:#e8f1ee; border:none; cursor:pointer;
  width:36px; height:36px; border-radius:50%; font-size:13px;
  display:flex; align-items:center; justify-content:center;
}}
.sp-btn:hover {{ background:rgba(255,255,255,0.2); }}
.sp-btn.main {{ background:#5eead4; color:#111614; width:42px; height:42px; font-size:15px; }}
.sp-btn.main:hover {{ background:#7cf5e1; }}
.sp-btn.active {{ background:var(--accent); color:#fff; }}
.sp-info {{ flex:1; min-width:0; }}
.sp-title {{
  font-size:12px; font-weight:600; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; margin-bottom:4px;
}}
.sp-progress {{ height:3px; background:rgba(255,255,255,0.15); border-radius:2px; overflow:hidden; cursor:pointer; }}
.sp-progress-bar {{ height:100%; background:#5eead4; width:0%; transition:width .2s linear; }}
.sp-time {{ font-size:10px; color:#9fb3ac; flex-shrink:0; min-width:60px; text-align:right; }}
.sp-loop {{
  background:none; border:1px solid rgba(255,255,255,0.2); color:#9fb3ac;
  cursor:pointer; font-size:11px; padding:3px 8px; border-radius:4px; flex-shrink:0;
}}
.sp-loop.on {{ background:var(--accent); border-color:var(--accent); color:#fff; }}

.footer {{ text-align:center; color:var(--muted); padding:20px; font-size:11px; }}
.footer a {{ color:var(--accent); text-decoration:none; }}

@media (max-width:480px) {{
  header.hero {{ padding:30px 16px 22px; }}
  header.hero h1 {{ font-size:20px; }}
  .ep-cover {{ width:60px; height:60px; }}
  .ep-body h2 {{ font-size:14px; }}
  .sp-btn {{ width:32px; height:32px; font-size:11px; }}
  .sp-btn.main {{ width:38px; height:38px; font-size:13px; }}
}}
</style>
</head>
<body>
<header class="hero">
  <h1>Paper Podcast</h1>
  <p>研究论文深度解读。每篇 15 分钟口播，听完脑子里能建出论文架构。</p>
  <div class="subscribe">
    <a href="feed.xml">RSS 订阅</a>
    <a href="https://github.com/guoshaoyang-pku/paper-podcast">GitHub</a>
  </div>
</header>

<nav class="tabs">
  <button class="tab-btn active" data-tab="episodes">全部论文</button>
  <button class="tab-btn" data-tab="queue">我的歌单 <span class="queue-count" id="qc">0</span></button>
</nav>

<main>
  <div id="episodes-panel">
    {''.join(cards)}
  </div>
  <div id="queue-panel" class="queue-panel">
    <div id="queue-list"></div>
  </div>
</main>

<div class="sticky-player" id="stickyPlayer">
  <div class="sp-controls">
    <button class="sp-btn" id="spPrev" title="上一首">&#9198;</button>
    <button class="sp-btn main" id="spPlay">&#9654;</button>
    <button class="sp-btn" id="spNext" title="下一首">&#9197;</button>
  </div>
  <div class="sp-info">
    <div class="sp-title" id="spTitle">&mdash;</div>
    <div class="sp-progress" id="spProgress"><div class="sp-progress-bar" id="spBar"></div></div>
  </div>
  <span class="sp-time" id="spTime">0:00</span>
  <button class="sp-loop" id="spLoop" title="单曲循环">&#128257;</button>
  <button class="sp-btn" id="spClose" title="关闭">&times;</button>
</div>
<audio id="audio" preload="metadata"></audio>

<div class="footer">
  <a href="https://github.com/guoshaoyang-pku/paper-podcast">paper-podcast</a> &middot;
  <a href="feed.xml">RSS</a> &middot;
  锁屏后台播放请用<a href="feed.xml">播客 app 订阅</a>
</div>

<script>
(function(){{
  const EPS = {eps_json};
  const BASE_URL = '{BASE_URL}';
  const audio = document.getElementById('audio');
  const player = document.getElementById('stickyPlayer');
  const spPlay = document.getElementById('spPlay');
  const spTitle = document.getElementById('spTitle');
  const spBar = document.getElementById('spBar');
  const spTime = document.getElementById('spTime');
  const spProgress = document.getElementById('spProgress');
  const spLoop = document.getElementById('spLoop');
  const spClose = document.getElementById('spClose');
  const spPrev = document.getElementById('spPrev');
  const spNext = document.getElementById('spNext');
  const qc = document.getElementById('qc');

  let currentSlug = null;
  let loop = false;
  let queue = JSON.parse(localStorage.getItem('pp_queue') || '[]');

  function fmt(s){{
    s=Math.floor(s); const m=Math.floor(s/60); const r=s%60;
    return m+':'+(r<10?'0':'')+r;
  }}
  function findEp(slug){{ return EPS.find(e=>e.slug===slug); }}

  function updateQueueUI(){{
    qc.textContent = queue.length;
    document.querySelectorAll('.queue-btn').forEach(b=>{{
      b.classList.toggle('in-queue', queue.includes(b.dataset.slug));
      b.textContent = queue.includes(b.dataset.slug) ? '\\u2713已加入' : '\\uff0b歌单';
    }});
    const ql = document.getElementById('queue-list');
    if(queue.length === 0){{
      ql.innerHTML = '<div class="queue-empty">歌单为空。点击论文卡片上的"\\uff0b歌单"添加。</div>';
      return;
    }}
    ql.innerHTML = queue.map((slug, i) => {{
      const e = findEp(slug); if(!e) return '';
      return '<div class="queue-item" data-slug="'+slug+'">' +
        '<img src="'+e.slug+'/'+e.cover+'" alt="">' +
        '<div class="qi-body"><div class="qi-title">'+e.title+'</div>' +
        '<div class="qi-meta">'+e.duration_str+' &middot; '+e.affiliation+'</div></div>' +
        '<div class="qi-controls">' +
        '<button class="qi-btn" onclick="moveQueue('+i+',-1)" '+(i===0?'disabled':'')+'>\\u2191</button>' +
        '<button class="qi-btn" onclick="moveQueue('+i+',1)" '+(i===queue.length-1?'disabled':'')+'>\\u2193</button>' +
        '<button class="qi-btn" onclick="playFromQueue('+i+')">\\u25B6</button>' +
        '<button class="qi-btn danger" onclick="removeFromQueue('+i+')">\\u00D7</button>' +
        '</div></div>';
    }}).join('');
  }}

  function toggleQueue(slug){{
    const idx = queue.indexOf(slug);
    if(idx >= 0) queue.splice(idx, 1);
    else queue.push(slug);
    localStorage.setItem('pp_queue', JSON.stringify(queue));
    updateQueueUI();
  }}
  window.toggleQueue = toggleQueue;

  window.moveQueue = function(idx, dir){{
    const ni = idx + dir;
    if(ni < 0 || ni >= queue.length) return;
    const tmp = queue[idx]; queue[idx] = queue[ni]; queue[ni] = tmp;
    localStorage.setItem('pp_queue', JSON.stringify(queue));
    updateQueueUI();
  }};
  window.removeFromQueue = function(idx){{
    queue.splice(idx, 1);
    localStorage.setItem('pp_queue', JSON.stringify(queue));
    updateQueueUI();
  }};
  window.playFromQueue = function(idx){{
    const slug = queue[idx];
    if(slug) play(slug);
  }};

  function play(slug){{
    const ep = findEp(slug);
    if(!ep || !ep.has_audio) return;
    if(currentSlug === slug && !audio.paused){{
      audio.pause(); return;
    }}
    audio.src = slug + '/audio/podcast.mp3';
    spTitle.textContent = ep.title;
    player.classList.add('show');
    currentSlug = slug;
    document.querySelectorAll('.episode').forEach(el=>{{
      el.classList.toggle('playing', el.dataset.slug === slug);
      const btn = el.querySelector('.play-btn');
      if(btn) btn.classList.toggle('playing', el.dataset.slug === slug);
    }});
    audio.play();
    updateMediaSession(ep);
  }}
  window.play = play;

  function updateMediaSession(ep){{
    if(!('mediaSession' in navigator)) return;
    try {{
      navigator.mediaSession.metadata = new MediaMetadata({{
        title: ep.title,
        artist: ep.affiliation,
        album: 'Paper Podcast',
        artwork: [{{src: BASE_URL + '/' + ep.slug + '/' + ep.cover, sizes: '512x512', type: 'image/png'}}]
      }});
      navigator.mediaSession.setActionHandler('play', () => audio.play());
      navigator.mediaSession.setActionHandler('pause', () => audio.pause());
      navigator.mediaSession.setActionHandler('previoustrack', () => playPrev());
      navigator.mediaSession.setActionHandler('nexttrack', () => playNext());
    }} catch(e) {{}}
  }}

  function playNext(){{
    if(loop && currentSlug){{
      audio.currentTime = 0; audio.play(); return;
    }}
    const qi = queue.indexOf(currentSlug);
    if(qi >= 0 && qi < queue.length - 1){{
      play(queue[qi + 1]); return;
    }}
    const ei = EPS.findIndex(e=>e.slug===currentSlug);
    if(ei >= 0 && ei < EPS.length - 1){{
      play(EPS[ei + 1].slug);
    }} else if(loop && EPS.length > 0) {{
      play(EPS[0].slug);
    }}
  }}
  function playPrev(){{
    const ei = EPS.findIndex(e=>e.slug===currentSlug);
    if(ei > 0) play(EPS[ei - 1].slug);
  }}

  document.querySelectorAll('.play-btn').forEach(btn => {{
    btn.addEventListener('click', (e) => {{
      e.stopPropagation();
      play(btn.dataset.slug);
    }});
  }});
  document.querySelectorAll('.queue-btn').forEach(btn => {{
    btn.addEventListener('click', (e) => {{
      e.stopPropagation();
      toggleQueue(btn.dataset.slug);
    }});
  }});

  document.querySelectorAll('.episode').forEach(el => {{
    el.addEventListener('click', () => {{
      const slug = el.dataset.slug;
      const textDiv = document.getElementById('text-' + slug);
      if(textDiv.classList.contains('open')){{
        textDiv.classList.remove('open');
        return;
      }}
      document.querySelectorAll('.ep-text.open').forEach(t => t.classList.remove('open'));
      textDiv.classList.add('open');
      if(!textDiv.dataset.loaded){{
        textDiv.innerHTML = '<p class="loading">加载文稿...</p>';
        fetch(slug + '/podcast.md').then(r=>r.text()).then(md=>{{
          textDiv.dataset.loaded = '1';
          textDiv.innerHTML = mdToHtml(md);
        }}).catch(()=>{{ textDiv.innerHTML = '<p class="loading">加载失败，请点"读全文"查看</p>'; }});
      }}
    }});
  }});

  function mdToHtml(md){{
    return md.split('\\n').map(line => {{
      if(line.startsWith('### ')) return '<h3>'+line.slice(4)+'</h3>';
      if(line.startsWith('## ')) return '<h2>'+line.slice(3)+'</h2>';
      if(line.startsWith('# ')) return '<h1>'+line.slice(2)+'</h1>';
      if(line.startsWith('> ')) return '<p style="color:var(--muted)">'+line.slice(2)+'</p>';
      if(line.trim()==='') return '';
      return '<p>'+line+'</p>';
    }}).join('\\n');
  }}

  spPlay.addEventListener('click', () => {{
    if(audio.paused) audio.play(); else audio.pause();
  }});
  spPrev.addEventListener('click', playPrev);
  spNext.addEventListener('click', playNext);
  spClose.addEventListener('click', () => {{
    audio.pause(); audio.currentTime=0; player.classList.remove('show');
    document.querySelectorAll('.episode').forEach(el=>el.classList.remove('playing'));
    document.querySelectorAll('.play-btn').forEach(b=>b.classList.remove('playing'));
    currentSlug = null;
  }});
  spLoop.addEventListener('click', () => {{
    loop = !loop;
    spLoop.classList.toggle('on', loop);
    spLoop.title = loop ? '单曲循环:开' : '单曲循环:关';
  }});

  audio.addEventListener('play', () => {{ spPlay.innerHTML = '\\u23F8'; }});
  audio.addEventListener('pause', () => {{ spPlay.innerHTML = '\\u25B6'; }});
  audio.addEventListener('ended', () => {{ playNext(); }});
  audio.addEventListener('timeupdate', () => {{
    if(audio.duration){{
      spBar.style.width = (audio.currentTime/audio.duration*100) + '%';
      spTime.textContent = fmt(audio.currentTime) + ' / ' + fmt(audio.duration);
    }}
  }});
  spProgress.addEventListener('click', (e) => {{
    if(audio.duration){{
      const rect = spProgress.getBoundingClientRect();
      audio.currentTime = (e.clientX - rect.left) / rect.width * audio.duration;
    }}
  }});

  document.querySelectorAll('.tab-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.dataset.tab;
      document.getElementById('episodes-panel').style.display = tab==='episodes' ? '' : 'none';
      document.getElementById('queue-panel').classList.toggle('active', tab==='queue');
    }});
  }});

  updateQueueUI();
}})();
</script>
</body>
</html>
"""


def build_rss(eps: list[dict]) -> str:
    items = []
    for e in eps:
        if not e["has_audio"]:
            continue
        audio_url = f"{BASE_URL}/{e['slug']}/audio/podcast.mp3"
        page_url = f"{BASE_URL}/{e['slug']}/index.html"
        audio_path = PAPERS / e["slug"] / "audio" / "podcast.mp3"
        size = audio_path.stat().st_size if audio_path.exists() else 0
        pub = iso_date(e["date"])
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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Paper Podcast</title>
    <link>{BASE_URL}/</link>
    <language>zh-cn</language>
    <description>研究论文深度解读。每篇 15 分钟口播，听完脑子里能建出论文架构。</description>
    <itunes:author>Paper Podcast</itunes:author>
    <itunes:summary>研究论文深度解读。每篇 15 分钟口播，听完脑子里能建出论文架构。</itunes:summary>
    <itunes:category text="Technology"/>
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
