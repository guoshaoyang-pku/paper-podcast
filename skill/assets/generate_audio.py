#!/usr/bin/env python3
"""Generate a single-host podcast MP3 from podcast.md using a pluggable TTS provider.

Usage:
    python3 generate_audio.py papers/<slug>
    python3 generate_audio.py papers/<slug> --provider edge
    python3 generate_audio.py papers/<slug> --provider openai --voice alloy
    python3 generate_audio.py papers/<slug> --provider elevenlabs --voice <voice_id>
    python3 generate_audio.py papers/<slug> --provider f5 --ref-audio ref.wav
    python3 generate_audio.py papers/<slug> --config tts_config.yaml

Reads:  papers/<slug>/podcast.md
Writes: papers/<slug>/audio/podcast.mp3
        papers/<slug>/audio/segments/  (per-segment mp3, for debugging)

Providers:
    edge        — Microsoft Edge TTS (free, no API key, default)
    openai      — OpenAI gpt-4o-mini-tts (needs OPENAI_API_KEY)
    elevenlabs  — ElevenLabs (needs ELEVENLABS_API_KEY + voice_id)
    f5          — F5-TTS local (needs GPU + ref audio; stub, implement when you have compute)

Config: see tts_config.yaml for voice/rate/gap defaults per provider.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


# ---------- markdown -> speakable segments ----------

def md_to_segments(md: str) -> list[str]:
    """Turn podcast.md into a list of speakable text segments."""
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
        if line.startswith("```"):
            if not in_code:
                flush()
                in_code = True
                code_buf = []
            else:
                in_code = False
                code_text = "\n".join(code_buf).strip()
                if code_text:
                    cleaned = re.sub(r"\s+", " ", code_text)
                    segments.append(cleaned)
            continue
        if in_code:
            code_buf.append(line)
            continue
        if re.match(r"^\s*-{3,}\s*$", line) or re.match(r"^\s*\*{3,}\s*$", line):
            flush()
            continue
        if line.startswith(">"):
            line = line.lstrip(">").strip()
            if not line:
                continue
        if line.startswith("#"):
            flush()
            text = _strip_inline(line.lstrip("#").strip())
            if text:
                segments.append(text)
            continue
        if not line.strip():
            flush()
            continue
        line = re.sub(r"^\s*[-*]\s+", "", line)
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        line = _strip_inline(line)
        buf.append(line)

    flush()
    return [s for s in segments if len(re.sub(r"[\s\W]+", "", s)) >= 2]


def _strip_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    return text


# ---------- TTS provider abstraction ----------

class TTSProvider:
    """Base class. Each provider implements synth_segment."""
    name = "base"

    def __init__(self, voice: str, rate: str, **kw):
        self.voice = voice
        self.rate = rate

    async def synth_segment(self, text: str, out_path: Path) -> None:
        raise NotImplementedError

    def probe_duration(self, path: Path) -> float:
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


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS. Free, no API key. Good for testing."""
    name = "edge"

    async def synth_segment(self, text: str, out_path: Path) -> None:
        import edge_tts
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        await communicate.save(out_path.as_posix())


class OpenAITTSProvider(TTSProvider):
    """OpenAI gpt-4o-mini-tts. Needs OPENAI_API_KEY. High quality, ~¥4/episode."""
    name = "openai"

    def __init__(self, voice: str, rate: str, instructions: str = "", **kw):
        super().__init__(voice, rate, **kw)
        self.instructions = instructions or "用自然中文技术口播风格朗读，语速中等偏快，避免棒读。"

    async def synth_segment(self, text: str, out_path: Path) -> None:
        from openai import OpenAI
        client = OpenAI()
        # rate -> speed: OpenAI doesn't take rate str, map +8% to instructions
        response = client.audio.speech.create(
            model=os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=self.voice,
            input=text,
            response_format="mp3",
            instructions=self.instructions,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        response.write_to_file(out_path.as_posix())


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs. Needs ELEVENLABS_API_KEY + voice_id. Best for voice cloning."""
    name = "elevenlabs"

    async def synth_segment(self, text: str, out_path: Path) -> None:
        from elevenlabs import ElevenLabs
        client = ElevenLabs()
        audio = client.text_to_speech.convert(
            voice_id=self.voice,
            model_id=os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
            text=text,
            output_format=os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_192"),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)


class F5TTSProvider(TTSProvider):
    """F5-TTS local. Needs GPU + reference audio for voice cloning.
    Stub: implement when you have compute. The interface is stable; just fill
    in the model loading and inference below.
    """
    name = "f5"

    def __init__(self, voice: str, rate: str, ref_audio: str = "", ref_text: str = "", **kw):
        super().__init__(voice, rate, **kw)
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        # TODO: implement when you have GPU
        # from f5_tts.api import F5TTS
        # self._model = F5TTS(ckpt_file=..., vocab_file=...)
        raise NotImplementedError(
            "F5-TTS provider not yet implemented. Implement _load_model + synth_segment "
            "when you have GPU compute. Interface is stable; see TODO comments."
        )

    async def synth_segment(self, text: str, out_path: Path) -> None:
        model = self._load_model()
        # TODO: model.infer(ref_audio, ref_text, text, ...) -> wav
        # then ffmpeg wav -> mp3
        raise NotImplementedError


PROVIDERS = {
    "edge": EdgeTTSProvider,
    "openai": OpenAITTSProvider,
    "elevenlabs": ElevenLabsProvider,
    "f5": F5TTSProvider,
}

# default config per provider
DEFAULT_CONFIG = {
    "edge":       {"voice": "zh-CN-YunyangNeural", "rate": "+8%", "gap": 0.35, "concurrency": 6},
    "openai":     {"voice": "alloy", "rate": "+8%", "gap": 0.35, "concurrency": 4,
                   "instructions": "用自然中文技术口播风格朗读，语速中等偏快，避免棒读。"},
    "elevenlabs": {"voice": "", "rate": "+0%", "gap": 0.4, "concurrency": 3},
    "f5":         {"voice": "", "rate": "+0%", "gap": 0.4, "concurrency": 1, "ref_audio": "", "ref_text": ""},
}


def load_config(config_path: Optional[Path], provider: str) -> dict:
    """Load tts_config.yaml if present, else use DEFAULT_CONFIG[provider]."""
    cfg = dict(DEFAULT_CONFIG.get(provider, {}))
    if config_path and config_path.exists():
        try:
            import yaml
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            cfg.update(data.get(provider, {}))
        except ImportError:
            print("warning: pyyaml not installed, using defaults", file=sys.stderr)
    return cfg


# ---------- synth + concat ----------

async def synth_all(segments: list[str], seg_dir: Path, provider: TTSProvider,
                    concurrency: int = 6) -> list[Path]:
    seg_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    paths: list[Path | None] = [None] * len(segments)

    async def one(i: int, text: str) -> None:
        async with sem:
            p = seg_dir / f"seg_{i:04d}.mp3"
            try:
                await provider.synth_segment(text, p)
                paths[i] = p
            except Exception as e:
                print(f"  [skip] seg {i}: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
                print(f"          text: {text[:100]!r}", file=sys.stderr)

    await asyncio.gather(*[one(i, t) for i, t in enumerate(segments)])
    return [p for p in paths if p is not None]


def concat_segments(seg_paths: list[Path], out_mp3: Path, gap_seconds: float,
                    bitrate: str = "64k") -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    if not seg_paths:
        raise SystemExit("no segments to concat")

    list_path = out_mp3.parent / "concat.txt"
    lines: list[str] = []

    if gap_seconds > 0 and len(seg_paths) > 1:
        silence_path = out_mp3.parent / f"silence_{gap_seconds:.2f}.mp3"
        if not silence_path.exists():
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"anullsrc=channel_layout=stereo:sample_rate=24000",
                "-t", f"{gap_seconds}", "-c:a", "libmp3lame", "-b:a", "96k",
                silence_path.as_posix()
            ], check=True, capture_output=True)
        for idx, p in enumerate(seg_paths):
            lines.append(f"file '{p.as_posix()}'")
            if idx < len(seg_paths) - 1:
                lines.append(f"file '{silence_path.as_posix()}'")
    else:
        lines = [f"file '{p.as_posix()}'" for p in seg_paths]

    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_path.as_posix(),
        "-c:a", "libmp3lame", "-b:a", bitrate,
        out_mp3.as_posix()
    ], check=True)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug_dir")
    ap.add_argument("--provider", default="edge", choices=list(PROVIDERS.keys()),
                    help="TTS provider (edge/openai/elevenlabs/f5)")
    ap.add_argument("--voice", default=None, help="override voice from config")
    ap.add_argument("--rate", default=None, help="override rate, e.g. +8%")
    ap.add_argument("--gap", type=float, default=None, help="silence seconds between segments")
    ap.add_argument("--config", default=None, help="path to tts_config.yaml")
    ap.add_argument("--bitrate", default="64k", help="output mp3 bitrate")
    args = ap.parse_args()

    slug_dir = Path(args.slug_dir).resolve()
    md_path = slug_dir / "podcast.md"
    if not md_path.exists():
        sys.exit(f"podcast.md not found: {md_path}")

    config_path = Path(args.config) if args.config else (ROOT / "tts_config.yaml")
    cfg = load_config(config_path, args.provider)
    voice = args.voice or cfg["voice"]
    rate = args.rate or cfg["rate"]
    gap = args.gap if args.gap is not None else cfg.get("gap", 0.35)
    concurrency = cfg.get("concurrency", 6)

    # check API keys
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY required for --provider openai")
    if args.provider == "elevenlabs" and not os.environ.get("ELEVENLABS_API_KEY"):
        sys.exit("ELEVENLABS_API_KEY required for --provider elevenlabs")
    if args.provider == "elevenlabs" and not voice:
        sys.exit("ELEVENLABS_VOICE_ID (or --voice) required for --provider elevenlabs")

    ProviderCls = PROVIDERS[args.provider]
    provider_kwargs = {k: v for k, v in cfg.items()
                       if k in {"instructions", "ref_audio", "ref_text"}}
    provider = ProviderCls(voice=voice, rate=rate, **provider_kwargs)

    md = md_path.read_text(encoding="utf-8")
    segments = md_to_segments(md)
    total_chars = sum(len(s) for s in segments)
    print(f"provider: {args.provider} | voice: {voice} | rate: {rate}")
    print(f"parsed {len(segments)} segments, {total_chars} chars")

    seg_dir = slug_dir / "audio" / "segments"
    print(f"synthesizing {len(segments)} segments (concurrency {concurrency})...")
    seg_paths = asyncio.run(synth_all(segments, seg_dir, provider, concurrency))
    print(f"got {len(seg_paths)} segment mp3s")

    out_mp3 = slug_dir / "audio" / "podcast.mp3"
    print(f"concatenating with {gap}s gap -> {out_mp3}")
    concat_segments(seg_paths, out_mp3, gap, args.bitrate)

    dur = provider.probe_duration(out_mp3)
    size_mb = out_mp3.stat().st_size / 1024 / 1024
    print(f"\nOK: {out_mp3}")
    print(f"    duration: {dur:.1f}s ({dur/60:.1f} min)")
    print(f"    size: {size_mb:.2f} MB")
    print(f"    provider: {args.provider}, voice: {voice}, rate: {rate}")


if __name__ == "__main__":
    main()
