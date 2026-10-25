# -*- coding: utf-8 -*-
"""整曲 ASR 回读：换声前 vs 换声后的可懂度相对比较。

★ 为什么单独写这个（不直接用 dsx_accept_cer.py）：
  那个脚本针对**台词片段**，用 large-v3 + 逐句对齐；
  整首 187 s 的歌用 large-v3 在 CPU 上既慢又吃内存（3 GB 级），
  而本轮训练正占着 6.4 GB ⇒ 这里用 **base + int8 + VAD**，峰值约 0.8 GB。

★ 怎么读这个数：
  唱歌的 ASR 绝对 CER 一定很难看（不是换声的锅，是唱歌本来难识别）。
  ★ 所以**只看相对差**：换声后 CER 与**源音频自己的 CER** 相比。
    换声后 ≤ 源 + 2 个百分点 ⇒ 判定「没伤可懂度」。
    换声后明显更高 ⇒ 说明换声确实掉了字，要修。

★ 三层 CER 的理由：中文同音字会被汉字层误判成「掉字」，
  拼音层（同音字不罚）判「音素是否走形」，声调层判「声调有没有被抹平」。

用法（必须在 CosyVoice env 跑：pypinyin 与 faster-whisper 只在那个 env）：
  python dsx_song_cer.py --src <源.wav> --conv <换声.wav> --json out.json
"""
import argparse
import json
import sys
from pathlib import Path


def norm(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum())


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def py_seq(s: str):
    from pypinyin import Style, lazy_pinyin
    return lazy_pinyin(s, style=Style.TONE)


def tone_seq(pys):
    m = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
         "a": "1", "e": "2", "i": "3", "o": "4", "u": "5", "v": "4", "ü": "1"}
    out = []
    for p in pys:
        if not p:
            continue
        c = p[-1]
        out.append(m.get(c, "5"))
    return "".join(out)


def cer_layers(ref_text, hyp_text):
    r, h = norm(ref_text), norm(hyp_text)
    d = edit_distance(r, h)
    han = d / max(len(r), 1)
    rp, hp = "".join(py_seq(r)), "".join(py_seq(h))
    d2 = edit_distance(rp, hp)
    py = d2 / max(len(rp), 1)
    rt, ht = tone_seq(py_seq(r)), tone_seq(py_seq(h))
    d3 = edit_distance(rt, ht)
    tone = d3 / max(len(rt), 1)
    return dict(han_cer=round(han, 4), pinyin_cer=round(py, 4),
                tone_cer=round(tone, 4),
                ref_chars=len(r), hyp_chars=len(h))


def transcribe(model, path):
    segs, info = model.transcribe(str(path), language="zh", beam_size=1,
                                   vad_filter=True, condition_on_previous_text=False)
    parts = [s.text for s in segs]
    return "".join(parts).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="换声前的源干声")
    ap.add_argument("--conv", required=True, help="换声结果")
    ap.add_argument("--json", default=None)
    ap.add_argument("--model-size", default="base")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from faster_whisper import WhisperModel
    model = WhisperModel(a.model_size, device=a.device, compute_type="int8")

    out = {"model_size": a.model_size, "device": a.device}
    texts = {}
    for tag, p in (("src", a.src), ("conv", a.conv)):
        t = transcribe(model, p)
        texts[tag] = t
        out[tag + "_text"] = t
        print(f"[{tag}] {t[:200]}")

    # 源 vs 换声：直接比两段 ASR 文本（判「换声有没有改变可懂度」）
    out["src_vs_conv"] = cer_layers(texts["src"], texts["conv"])
    # 以源自己的 ASR 为参照，看换声后掉了多少字
    out["conv_vs_src_asr"] = cer_layers(texts["src"], texts["conv"])
    print(json.dumps(out["src_vs_conv"], ensure_ascii=False, indent=2))

    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print("报告 ->", a.json)


if __name__ == "__main__":
    sys.exit(main())
