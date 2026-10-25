# -*- coding: utf-8 -*-
"""DS娘验收 —— ASR 回读三层 CER（汉字 / 拼音带调 / 声调调号）。

★ 为什么必须在 **CosyVoice env** 跑：pypinyin 只在那个 env 里
  （`D:/AI-Voice/01_情感配音_CosyVoice/env/Scripts/python.exe`），
  RVC venv 没有。faster-whisper 同理。
★ 三层 CER 的理由：中文同音字会被汉字层误判成「掉字」，
  拼音层（同音字不罚）判「音素是否走形」，声调层判「声调有没有被抹平」。
★ 参考有两组：
  - ref = 目标文本（判「能不能听懂」）
  - ref = 源音频的 ASR（判「换声有没有伤到」）—— 这是硬标准：
    换声后 CER 应当与**源**持平；源本身不好的不算换声的锅。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

WORK = Path(r"E:/AI-RVC/work/dsx_accept")
XVAL = Path(r"E:/AI-RVC/dataset/dsxiaoyi_xval")


def norm(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", s)


def edit_distance(a, b):
    if not a:
        return 0.0
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[len(b)] / len(a)


def py_seq(s: str):
    from pypinyin import lazy_pinyin, Style
    t = norm(s)
    parts = lazy_pinyin(t, style=Style.TONE3, neutral_tone_with_five=True,
                        errors=lambda x: list(x))
    out = []
    for p in parts:
        p = re.sub(r"[^a-z0-9]", "", p.lower())
        if p:
            out.append(p)
    return out


def tone_seq(pys):
    return [(re.search(r"\d$", p).group(0) if re.search(r"\d$", p) else "0")
            for p in pys]


def cer_layers(ref_text, hyp_text):
    r, h = norm(ref_text), norm(hyp_text)
    c_han = edit_distance(r, h) if r else 0.0
    rp, hp = py_seq(ref_text), py_seq(hyp_text)
    return (round(c_han, 4), round(edit_distance(rp, hp), 4),
            round(edit_distance(tone_seq(rp), tone_seq(hp)), 4),
            " ".join(rp), " ".join(hp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="e20,e40,e100,e150")
    ap.add_argument("--out", default=str(WORK / "cer.json"))
    a = ap.parse_args()
    ckpts = [c.strip() for c in a.ckpts.split(",") if c.strip()]

    # 目标文本（交叉集有 ground truth；真人干声是「唱歌/说话片段」无文本）
    man = json.loads((XVAL / "_synth_manifest.json").read_text(encoding="utf-8"))
    TEXT = {r["id"]: r["text"] for r in man["rows"] if r.get("state") == "ok"}

    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    def asr(p):
        segs, _ = model.transcribe(str(p), language="zh", beam_size=5,
                                   vad_filter=True,
                                   initial_prompt="以下是普通话的句子。")
        return "".join(s.text for s in segs).strip()

    rows = []
    # 源（交叉集）
    for lid in sorted(TEXT):
        p = XVAL / f"{lid}.wav"
        if not p.is_file():
            continue
        hyp = asr(p)
        h, py, tn, rp, hp = cer_layers(TEXT[lid], hyp)
        rows.append(dict(group="xval", ckpt="src", src=lid,
                         cer_han=h, cer_py=py, cer_tone=tn,
                         hyp=hyp, ref_text=TEXT[lid], ref_py=rp, hyp_py=hp))
        print(f"src {lid} CER={h:.3f} 拼音={py:.3f} 声调={tn:.3f} | {hyp}", flush=True)

    # 各检查点输出
    for ck in ckpts:
        for lid in sorted(TEXT):
            p = WORK / "out" / ck / "xval" / f"{lid}.wav"
            if not p.is_file():
                continue
            hyp = asr(p)
            h, py, tn, rp, hp = cer_layers(TEXT[lid], hyp)
            rows.append(dict(group="xval", ckpt=ck, src=lid,
                             cer_han=h, cer_py=py, cer_tone=tn,
                             hyp=hyp, ref_text=TEXT[lid], ref_py=rp, hyp_py=hp))
            print(f"{ck:5s} {lid} CER={h:.3f} 拼音={py:.3f} 声调={tn:.3f} | {hyp}",
                  flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print("\n=== 均值（汉字 / 拼音 / 声调）===")
    import statistics as st
    for ck in ["src"] + ckpts:
        v = [r for r in rows if r["ckpt"] == ck]
        if not v:
            continue
        print(f"{ck:5s} n={len(v):3d} 汉字={st.mean([x['cer_han'] for x in v]):.4f} "
              f"拼音={st.mean([x['cer_py'] for x in v]):.4f} "
              f"声调={st.mean([x['cer_tone'] for x in v]):.4f}")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
