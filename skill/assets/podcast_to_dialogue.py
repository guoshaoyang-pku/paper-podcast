#!/usr/bin/env python3
"""Convert a single-host podcast.md into a two-host dialogue JSON script.

This borrows the two-host conversation format from research-radio
(fabiogiglietto/research-radio): Speaker1 (host) + Speaker2 (researcher),
natural back-and-forth, JSON output.

Usage:
    python3 scripts/podcast_to_dialogue.py papers/<slug>
    python3 scripts/podcast_to_dialogue.py papers/<slug> --llm openai
    python3 scripts/podcast_to_dialogue.py papers/<slug> --llm gemini

Reads:  papers/<slug>/podcast.md
Writes: papers/<slug>/dialogue.json
        [
          {"speaker": "1", "text": "..."},  # host
          {"speaker": "2", "text": "..."},  # researcher
          ...
        ]

The dialogue.json can then be passed to generate_audio.py --dialogue to produce
a two-voice podcast (when a multi-speaker TTS provider is configured).

If no LLM is available (no API key), falls back to a deterministic heuristic
split that assigns alternating host/researcher turns by paragraph. The
heuristic is a placeholder; real quality needs the LLM rewrite.

LLM setup:
    openai  — needs OPENAI_API_KEY; uses gpt-4o-mini by default
    gemini  — needs GEMINI_API_KEY; uses gemini-2.5-flash
    none    — heuristic split only (for testing the pipeline)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Import segment parser from generate_audio
sys.path.insert(0, str(ROOT / "scripts"))
from generate_audio import md_to_segments  # noqa: E402


PROMPT_TEMPLATE = """你是一个技术播客编剧。把下面这篇单人技术口播稿改写成双人对谈脚本。

要求：
1. 两位说话人：Speaker1 是主持人(好奇、提问、总结)，Speaker2 是研究员(讲解技术细节、有判断)。
2. 保持原文的全部技术信息和数字，不要编造，不要简化掉关键细节。
3. 对谈要自然，像两个同事在聊一篇论文。Speaker1 适时追问、复述、给直觉；Speaker2 负责把方法讲透、给数字、给局限。
4. 每段发言 1-4 句话。总共 30-50 个 turn。
5. 不要用"不是X而是Y"这种口头禅。
6. 输出纯 JSON 数组，每个元素是 {{"speaker":"1","text":"..."}} 或 {{"speaker":"2","text":"..."}}。不要任何额外文字。

单人原稿：
---
{content}
---
"""


def heuristic_split(segments: list[str]) -> list[dict]:
    """Fallback: deterministically assign host/researcher turns.
    Not as good as LLM rewrite, but lets you test the two-voice pipeline
    without an API key.
    """
    dialogue = []
    for i, seg in enumerate(segments):
        speaker = "1" if i % 2 == 0 else "2"
        # host rephrases slightly, researcher keeps content
        if speaker == "1":
            # heuristic: if segment looks like a heading/topic, frame as question
            if len(seg) < 40 and not seg.endswith("。"):
                text = f"我们来看{seg}。"
            else:
                text = seg
        else:
            text = seg
        dialogue.append({"speaker": speaker, "text": text})
    return dialogue


def rewrite_with_openai(segments: list[str]) -> list[dict]:
    from openai import OpenAI
    client = OpenAI()
    content = "\n\n".join(segments)
    prompt = PROMPT_TEMPLATE.format(content=content)
    resp = client.chat.completions.create(
        model=os.environ.get("DIALOGUE_LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content
    # the model may wrap in a key
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # find the list value
            for v in data.values():
                if isinstance(v, list):
                    return v
        return data
    except json.JSONDecodeError:
        # try to extract JSON array
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def rewrite_with_gemini(segments: list[str]) -> list[dict]:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(os.environ.get("DIALOGUE_LLM_MODEL", "gemini-2.5-flash"))
    content = "\n\n".join(segments)
    prompt = PROMPT_TEMPLATE.format(content=content)
    resp = model.generate_content(prompt)
    text = resp.text
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise RuntimeError("could not parse JSON from gemini response")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug_dir")
    ap.add_argument("--llm", default="none", choices=["none", "openai", "gemini"],
                    help="LLM for rewrite (none=openai=gemini). none=heuristic split")
    args = ap.parse_args()

    slug_dir = Path(args.slug_dir).resolve()
    md_path = slug_dir / "podcast.md"
    if not md_path.exists():
        sys.exit(f"podcast.md not found: {md_path}")

    md = md_path.read_text(encoding="utf-8")
    segments = md_to_segments(md)
    print(f"parsed {len(segments)} segments from podcast.md")

    if args.llm == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY required for --llm openai")
        print("rewriting with OpenAI...")
        dialogue = rewrite_with_openai(segments)
    elif args.llm == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            sys.exit("GEMINI_API_KEY required for --llm gemini")
        print("rewriting with Gemini...")
        dialogue = rewrite_with_gemini(segments)
    else:
        print("heuristic split (no LLM) — for pipeline testing only")
        dialogue = heuristic_split(segments)

    out_path = slug_dir / "dialogue.json"
    out_path.write_text(json.dumps(dialogue, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    s1 = sum(1 for d in dialogue if d["speaker"] == "1")
    s2 = sum(1 for d in dialogue if d["speaker"] == "2")
    print(f"\nOK: {out_path}")
    print(f"    {len(dialogue)} turns (speaker1={s1}, speaker2={s2})")
    print(f"    llm: {args.llm}")


if __name__ == "__main__":
    main()
