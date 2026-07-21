#!/usr/bin/env python3
"""Lint paper-podcast outputs for banned self-media phrases.

Usage:
    python3 lint_phrases.py out/<slug>
    python3 lint_phrases.py out/<slug> --strict   # also fail on warning-level

Scans all .md files and card.json under the slug dir. Reports every match
with file:line and the matched text. Exit code 0 = clean, 1 = banned hits.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Banned patterns. Each entry: (regex, severity, label, suggestion)
# severity: "banned" = must fix before publish; "warn" = stylistic, review.
PATTERNS: list[tuple[re.Pattern, str, str, str]] = [
    # "不是 X，而是 Y" family — the main target.
    (re.compile(r"不是[^，。\n;；]{1,40}(，|,|。|;|；)\s*(而是|是)\s"), "banned", "不是X而是Y", "直接说'是什么'，或用'相比/区别在'"),
    (re.compile(r"不是[^，。\n;；]{1,40}(，|,)\s*而是"), "banned", "不是X而是Y", "直接说'是什么'"),
    # Leading "X 不是 Y，而是 Z"
    (re.compile(r"^[^，。\n]{1,30}不是[^，。\n]{1,30}，而是"), "banned", "X不是Y而是Z", "改成肯定句"),
    # "它不是 X，它要..."
    (re.compile(r"它不是[^，。\n]{1,40}，它"), "banned", "它不是X它要", "删掉前半句"),
    # "真正的 X 不是 Y"
    (re.compile(r"真正的[^，。\n]{1,30}不是"), "banned", "真正的X不是Y", "改肯定句"),
    # "不仅是 X，更是 Y" / "不只是 X，更是 Y"
    (re.compile(r"(不仅|不只是?)[^，。\n]{1,40}(，|,)\s*更是"), "banned", "不仅是X更是Y", "拆成两句或改肯定"),
    # Filler / cliché
    (re.compile(r"值得一提(的是)?"), "warn", "套话:值得一提", "删掉"),
    (re.compile(r"众所周知"), "warn", "套话:众所周知", "删掉或具体化"),
    (re.compile(r"毋庸置疑"), "warn", "套话:毋庸置疑", "删掉"),
    (re.compile(r"不言而喻"), "warn", "套话:不言而喻", "删掉或具体说"),
    (re.compile(r"总而言之|综上所述"), "warn", "套话:结尾词", "删掉或换具体结论"),
]


def lint_text(text: str, source: str) -> list[tuple[int, str, str, str, str]]:
    """Return list of (line_no, severity, label, suggestion, matched_line)."""
    hits: list[tuple[int, str, str, str, str]] = []
    for line_no, line in enumerate(text.split("\n"), 1):
        for pat, sev, label, sug in PATTERNS:
            if pat.search(line):
                hits.append((line_no, sev, label, sug, line.strip()))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug_dir")
    ap.add_argument("--strict", action="store_true", help="also fail on warn-level")
    args = ap.parse_args()

    root = Path(args.slug_dir).resolve()
    if not root.exists():
        sys.exit(f"not found: {root}")

    targets: list[tuple[str, str]] = []  # (rel_path, text)
    for p in sorted(root.glob("*")):
        if p.suffix == ".md":
            targets.append((p.name, p.read_text(encoding="utf-8")))
        elif p.name == "card.json":
            # lint as raw text so we catch strings inside JSON
            targets.append((p.name, p.read_text(encoding="utf-8")))

    total_banned = 0
    total_warn = 0
    for name, text in targets:
        hits = lint_text(text, name)
        if not hits:
            continue
        print(f"\n=== {name} ===")
        for line_no, sev, label, sug, line in hits:
            marker = "BANNED" if sev == "banned" else "WARN"
            print(f"  [{marker}] line {line_no} {label}")
            print(f"    -> {sug}")
            print(f"    | {line[:200]}")
            if sev == "banned":
                total_banned += 1
            else:
                total_warn += 1

    print(f"\n--- summary ---")
    print(f"banned: {total_banned}  warn: {total_warn}")
    if total_banned > 0 or (args.strict and total_warn > 0):
        print("FAIL — fix banned phrases before publishing.")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
