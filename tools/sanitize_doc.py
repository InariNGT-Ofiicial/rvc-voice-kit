# -*- coding: utf-8 -*-
"""把工作区技术报告脱敏后同步进仓库 docs/。

为什么要它：交付文档写在工作区（带本机绝对路径），仓库里不能带。
每次改完报告都要重跑一次，规则散在一次性的 shell 里很容易漏。

用法：
  python tools/sanitize_doc.py <源文件...> [--dst docs/]

规则（顺序敏感：长路径先替换）：
  盘符路径 → 占位符；占位符在文件头有一份说明。
"""
import argparse
import re
import sys
from pathlib import Path

RULES: list[tuple[str, str]] = [
    (r"[A-Za-z]:[\\/]AI-RVC[\\/]venv[\\/]Scripts[\\/]python\.exe", "<PY>"),
    (r"[A-Za-z]:[\\/]AI-RVC[\\/]repo", "<REPO>"),
    (r"[A-Za-z]:[\\/]AI-RVC", "<ROOT>"),
    (r"[A-Za-z]:[\\/]AI-Song[\\/]03_音色模型_SenrenBanka", "<MODELS>"),
    (r"[A-Za-z]:[\\/]AI-Song[\\/]01_人声分离_MSST-GUI[\\/]env", "<FFMPEG_DIR>"),
    (r"[A-Za-z]:[\\/]AI-Song[\\/]04_训练工作区", "<DATASET_SRC>"),
    (r"[A-Za-z]:[\\/]AI-Song", "<AI_SONG>"),
    (r"[A-Za-z]:[\\/]AI-Voice[\\/]01_情感配音_CosyVoice[\\/]CosyVoice_源码", "<COSY_SRC>"),
    (r"[A-Za-z]:[\\/]AI-Voice[\\/]01_情感配音_CosyVoice[\\/]env", "<COSY_ENV>"),
    (r"[A-Za-z]:[\\/]AI-Voice[\\/]01_情感配音_CosyVoice", "<COSY>"),
    (r"[A-Za-z]:[\\/]AI-Voice[\\/]音色预设", "<PRESETS>"),
    (r"[A-Za-z]:[\\/]AI-Voice[\\/]02_GPT-SoVITS[\\/]\.venv", "<SOVITS_ENV>"),
    (r"[A-Za-z]:[\\/]AI-Voice[\\/]02_GPT-SoVITS", "<SOVITS>"),
    (r"[A-Za-z]:[\\/]AI-Voice", "<AI_VOICE>"),
    (r"[A-Za-z]:[\\/]Codex[\\/]\d{4}-\d{2}-\d{2}[\\/][^\\s\"'`]*", "<CODEX>"),
    (r"[A-Za-z]:[\\/]WorkData[\\/][^\\s\"'`]*?rvc-research", "<REPORT>"),
    (r"[A-Za-z]:[\\/]WorkData[\\/][^\\s\"'`]*?qqbot-workspace", "<AGENT_CWD>"),
    (r"[A-Za-z]:[\\/]WorkData[\\/][^\\s\"'`]*", "<WORKSPACE>"),
    (r"[A-Za-z]:[\\/]Users[\\/][^\\s\"'`]+[\\/]\.dsh", "<DSH_HOME>"),
    (r"[A-Za-z]:[\\/]Users[\\/][^\\s\"'`]+", "<USER_HOME>"),
]

HEADER = (
    "<!-- 本文件由工作区技术报告迁移而来，原始绝对路径已替换为占位符：\n"
    "     <ROOT> 工作根 / <REPO> 上游 clone / <PY> venv 解释器 / <MODELS> 音色库 /\n"
    "     <FFMPEG_DIR> ffmpeg 目录 / <COSY> CosyVoice2 / <COSY_SRC> 其源码目录 /\n"
    "     <COSY_ENV> 其 venv / <PRESETS> 音色预设 / <DATASET_SRC> 原始训练集 /\n"
    "     <REPORT> 交付目录 / <WORKSPACE> 工作区 / <AGENT_CWD> agent 工作目录 /\n"
    "     <DSH_HOME> DSH 配置目录 / <CODEX> Codex 工作区。\n"
    "     本仓库实际以 rvcx/paths.py + rvcx.local.toml 解析这些路径。 -->\n\n"
)

# 匹配后仍应视为「残留」的模式（URL 里的盘符不算）
LEFTOVER = re.compile(r"[A-Za-z]:[\\/][^\s\)\|`，。；、\"'`]{2,60}")


def sanitize(text: str) -> tuple[str, int]:
    n = 0
    for pat, rep in RULES:
        text, k = re.subn(pat, rep, text)
        n += k
    return text, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+", help="源文件（工作区里的报告）")
    ap.add_argument("--dst", default="docs", help="目标目录")
    a = ap.parse_args()

    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    bad = 0
    for s in a.src:
        p = Path(s)
        raw = p.read_text(encoding="utf-8")
        clean, n = sanitize(raw)
        out = dst / p.name
        out.write_text(HEADER + clean, encoding="utf-8")
        left = sorted(set(LEFTOVER.findall(clean)))
        # URL 里的 https:// 与 wss:// 会误判，过滤掉明显是协议的
        left = [x for x in left if not x.startswith(("s://", "p://"))]
        print(f"{p.name}: 替换 {n} 处 -> {out}")
        if left:
            bad += 1
            print(f"  ⚠ 残留疑似绝对路径: {left[:6]}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
