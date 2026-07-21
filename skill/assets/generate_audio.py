#!/usr/bin/env python3
"""Generate a single-host podcast MP3 from podcast.md using Edge TTS.

Usage:
    python3 generate_audio.py papers/<slug>
    python3 generate_audio.py papers/<slug> --voice zh-CN-YunyangNeural
    python3 generate_audio.py papers/<slug> --rate +15% --gap 0.4

Reads:  papers/<slug>/podcast.md
Writes: papers/<slug>/audio/podcast.mp3
        papers/<slug>/audio/segments/  (per-segment mp3, for debugging)

Strategy:
- Strip markdown (headings, code fences become spoken text).
- Split into segments by paragraph / heading. Each segment ~5-25s when spoken.
- Edge TTS each segment in parallel (asyncio).
- ffmpeg concat with configurable inter-segment silence.
- Skip embedded LaTeX/math-y symbols that TTS will mispronounce.

Notes:
- Edge TTS single request has a ~5-10 min soft limit; segmenting also avoids this.
- Code blocks: we read them but mark them so they're spoken literally.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
from pathlib import Path


# ---------- markdown -> speakable segments ----------

def md_to_segments(md: str) -> list[str]:
    """Turn podcast.md into a list of speakable text segments.

    - Headings -> spoken as a section title (drop the #)
    - Code fences (```) -> spoken literally, but stripped of pure syntax noise
    - Bullet/numbered list prefixes -> dropped
    - Bold/italic markers -> dropped
    - Inline code `xxx` -> spoken as-is
    - URLs -> dropped (already absent in our podcasts)
    - Blank lines -> segment boundaries
    """
    lines = md.split("\n")
    segments: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            text = "\n".join(buf).strip()
            if text:
                segments.append(text)
            buf.clear()

    in_code = False
    code_buf: list[str] = []

    for line in lines:
        # code fence toggle
        if line.startswith("```"):
            if not in_code:
                flush()
                in_code = True
                code_buf = []
            else:
                in_code = False
                code_text = "\n".join(code_buf).strip()
                if code_text:
                    # spoken literally but cleaned
                    cleaned = re.sub(r"\s+", " ", code_text)
                    segments.append(cleaned)
            continue
        if in_code:
            code_buf.append(line)
            continue

        # skip horizontal rules / frontmatter separators
        if re.match(r"^\s*-{3,}\s*$", line) or re.match(r"^\s*\*{3,}\s*$", line):
            flush()
            continue
        # skip blockquote markers but keep text
        if line.startswith(">"):
            line = line.lstrip(">").strip()
            if not line:
                continue

        # heading
        if line.startswith("#"):
            flush()
            text = line.lstrip("#").strip()
            # strip markdown formatting from heading
            text = _strip_inline(text)
            if text:
                segments.append(text)
            continue

        # blank line -> segment boundary
        if not line.strip():
            flush()
            continue

        # bullet / numbered list item
        line = re.sub(r"^\s*[-*]\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)

        # strip inline markdown
        line = _strip_inline(line)
        buf.append(line)

    flush()
    # filter out segments that are pure symbols / too short to speak
    return [s for s in segments if len(re.sub(r"[\s\W]+", "", s)) >= 2]


def _strip_inline(text: str) -> str:
    # inline code: keep content
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # images ![alt](path) -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    return text


# ---------- TTS ----------

async def synth_segment(text: str, out_path: Path, voice: str, rate: str) -> None:
    """Synthesize one segment to MP3 via edge-tts."""
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(out_path.as_posix())


async def synth_all(segments: list[str], seg_dir: Path, voice: str, rate: str) -> list[Path]:
    """Synthesize all segments concurrently (bounded). Returns list of mp3 paths in order."""
    import os
    seg_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(6)  # avoid hammering the service
    paths: list[Path | None] = [None] * len(segments)

    async def one(i: int, text: str) -> None:
        async with sem:
            p = seg_dir / f"seg_{i:04d}.mp3"
            try:
                await synth_segment(text, p, voice, rate)
                paths[i] = p
            except Exception as e:
                # log and skip; usually empty/symbol-only segment
                print(f"  [skip] seg {i}: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
                print(f"          text: {text[:100]!r}", file=sys.stderr)

    tasks = [one(i, t) for i, t in enumerate(segments)]
    await asyncio.gather(*tasks)
    return [p for p in paths if p is not None]


# ---------- ffmpeg concat ----------

def concat_segments(seg_paths: list[Path], out_mp3: Path, gap_seconds: float) -> None:
    """Concat mp3 segments with `gap_seconds` of silence between each."""
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    if not seg_paths:
        raise SystemExit("no segments to concat")

    # build silence clip via ffmpeg lavfi
    # then build concat list: seg1, silence, seg2, silence, seg3, ...
    list_path = out_mp3.parent / "concat.txt"
    lines: list[str] = []
    for idx, p in enumerate(seg_paths):
        lines.append(f"file '{p.as_posix()}'")
        if idx < len(seg_paths) - 1 and gap_seconds > 0:
            # silence as a generated input is awkward with concat demuxer;
            # we generate a silence mp3 file once and reuse it
            pass
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if gap_seconds > 0 and len(seg_paths) > 1:
        # generate silence clip
        silence_path = out_mp3.parent / f"silence_{gap_seconds:.2f}.mp3"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate=24000",
            "-t", f"{gap_seconds}", "-c:a", "libmp3lame", "-b:a", "96k",
            silence_path.as_posix()
        ], check=True, capture_output=True)
        # rebuild list with silence interleaved
        lines2: list[str] = []
        for idx, p in enumerate(seg_paths):
            lines2.append(f"file '{p.as_posix()}'")
            if idx < len(seg_paths) - 1:
                lines2.append(f"file '{silence_path.as_posix()}'")
        list_path.write_text("\n".join(lines2) + "\n", encoding="utf-8")

    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_path.as_posix(),
        "-c:a", "libmp3lame", "-b:a", "128k",
        out_mp3.as_posix()
    ], check=True)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug_dir")
    ap.add_argument("--voice", default="zh-CN-YunyangNeural",
                    help="edge-tts voice (zh-CN-YunyangNeural / XiaoxiaoNeural / ...)")
    ap.add_argument("--rate", default="+8%",
                    help="speech rate, e.g. +10% / -5%")
    ap.add_argument("--gap", type=float, default=0.35,
                    help="silence seconds between segments")
    args = ap.parse_args()

    slug_dir = Path(args.slug_dir).resolve()
    md_path = slug_dir / "podcast.md"
    if not md_path.exists():
        sys.exit(f"podcast.md not found: {md_path}")

    md = md_path.read_text(encoding="utf-8")
    segments = md_to_segments(md)
    total_chars = sum(len(s) for s in segments)
    print(f"parsed {len(segments)} segments, {total_chars} chars total")
    for i, s in enumerate(segments[:3]):
        preview = s.replace("\n", " ")[:80]
        print(f"  [{i}] {preview}...")

    seg_dir = slug_dir / "audio" / "segments"
    print(f"synthesizing with {args.voice} rate={args.rate} ...")
    seg_paths = asyncio.run(synth_all(segments, seg_dir, args.voice, args.rate))
    print(f"got {len(seg_paths)} segment mp3s")

    out_mp3 = slug_dir / "audio" / "podcast.mp3"
    print(f"concatenating with {args.gap}s gap -> {out_mp3}")
    concat_segments(seg_paths, out_mp3, args.gap)

    # probe duration
    import shutil
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        d = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", out_mp3.as_posix()],
                           capture_output=True, text=True)
        dur = float(d.stdout.strip())
        size_mb = out_mp3.stat().st_size / 1024 / 1024
        print(f"\nOK: {out_mp3}")
        print(f"    duration: {dur:.1f}s ({dur/60:.1f} min)")
        print(f"    size: {size_mb:.2f} MB")
        print(f"    voice: {args.voice}, rate: {args.rate}")


if __name__ == "__main__":
    import edge_tts  # noqa: F401  (import here so argparse --help works without it)
    main()
