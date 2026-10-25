# -*- coding: utf-8 -*-
"""中文台词可懂度：ASR 回读算 CER（whisper large-v3，language=zh）。

★ 为什么中文比日语更需要这条：中文换声最大的风险不是「不像」，而是
  ① 声母韵母糊掉（齿音 sh/zh/ch、送气 p/t/k）；
  ② 声调被抹平（RVC 用源 f0，理论上声调该保住，但索引检索会拉向日语特征）。
  CER 是这两件事的综合体现。声调单独的量化在 dialogue_metrics_zh.py。

读法同日语批：回读 CER 含 whisper 自身误差 ⇒ 只看**相对变化**。
用 cosy env 跑（该 env 有 faster-whisper + torch，一次加载两用）。
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import json
import os
import re
import statistics as st
from pathlib import Path

DLG = _paths.WORK / "dialogue_zh"

TEXT = {
    "zh01": "请只回答我的问题。",
    "zh02": "别看我的脸，就这样听我说。",
    "zh03": "啊，失礼了。我叫日立真子。",
    "zh04": "早上我首先会打扫浴室，因为我挺喜欢泡澡的。",
    "zh05": "真是的，都说了三遍了，你怎么还不明白啊？",
    "zh06": "那个人明明什么都知道，却什么都不肯说，就一直笑着。",
}

# 系统标签：sid -> 输出文件名后缀
SYSTEMS = ["e120", "e200", "e120_ir0", "e120_ir050"]

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def norm(s: str) -> str:
    """只留汉字/字母/数字：ASR 标点不可靠，比标点没意义。"""
    s = s.replace("，", "").replace("。", "").replace("？", "").replace("！", "")
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", s)


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return dp[len(h)] / len(r)


def edit(a, b) -> float:
    """通用序列编辑距离（归一化）。"""
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
    """汉字 → 带声调拼音音节序列。'请只回答' -> ['qing3','zhi1','hui2','da2']

    ★ 为什么要拼音层：汉字层 CER 把同音字判成错（真子/贞子 同音），
      会把「ASR 认错字」误读成「RVC 掉字」。拼音层顺带把**声调**也算进去，
      所以它同时是声调保真的指标。
    """
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
    """只留声调数字：'qing3' -> '3'；轻声记 5；无调记 0。"""
    return [(re.search(r"\d$", p).group(0) if re.search(r"\d$", p) else "0")
            for p in pys]


def collect():
    files = [("src", p) for p in sorted((DLG / "src_tts").glob("*.wav"))]
    for sid in SYSTEMS:
        files += [(sid, p) for p in sorted((DLG / "out").glob(f"*_{sid}.wav"))]
    return files


def main():
    from faster_whisper import WhisperModel

    files = collect()
    print(f"待转写 {len(files)} 条", flush=True)
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    rows = []
    for sid, p in files:
        lid = p.stem.split("_")[0]
        segs, _ = model.transcribe(str(p), language="zh", beam_size=5,
                                   vad_filter=True,
                                   initial_prompt="以下是普通话的句子。")
        hyp = "".join(s.text for s in segs).strip()
        c = cer(TEXT[lid], hyp)
        rp, hp = py_seq(TEXT[lid]), py_seq(hyp)
        cpy = edit(rp, hp)
        ctone = edit(tone_seq(rp), tone_seq(hp))
        rows.append(dict(lid=lid, sid=sid, cer=round(c, 4), cer_py=round(cpy, 4),
                         cer_tone=round(ctone, 4), hyp=hyp,
                         hyp_py=" ".join(hp)))
        flag = "  <<< 掉字" if cpy > 0.12 else ""
        print(f"{lid} {sid:9s} CER={c:.3f} 拼音={cpy:.3f} 声调={ctone:.3f}  "
              f"{hyp}{flag}", flush=True)

    (DLG / "cer.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print("\n=== 均值（汉字 / 拼音 / 声调）===")
    for m in ["src"] + SYSTEMS:
        v = [r for r in rows if r["sid"] == m]
        if v:
            print(f"{m:9s} n={len(v)} 汉字CER={st.mean([x['cer'] for x in v]):.4f} "
                  f"拼音CER={st.mean([x['cer_py'] for x in v]):.4f} "
                  f"声调CER={st.mean([x['cer_tone'] for x in v]):.4f}")


if __name__ == "__main__":
    main()
