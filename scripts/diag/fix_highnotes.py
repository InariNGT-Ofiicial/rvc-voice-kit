# -*- coding: utf-8 -*-
"""高音补正：重建「含高音样本」的索引，让 index_rate 可以安全开启。

★ 为什么需要这个（根因，勿删）：
  DS娘语料（CosyVoice TTS）的 F0 分布实测：
      中位 296 Hz · p99 395 Hz · p99.9 416 Hz · >600Hz 占 **0.0012%** · 最高 657 Hz
  而《下等马》原唱（泠鸢）最高稳定音 **831 Hz**。
  ⇒ added_IVF1929 索引里 **几乎没有 700Hz+ 的帧**。
  ⇒ 推理时 index_rate>0 会把 780Hz 的查询帧检索到 **低八度的邻居**，
     表现为「高音突然降八度 / 掉音」。实测：
        index_rate=0    → 缺陷段输出 769.7 Hz ✅
        index_rate=0.75 → 缺陷段输出 410.5 Hz ❌（降八度）
  ⇒ **修法有两条**：
     A. 推理时 index_rate=0（立刻可用，代价：失去索引带来的表现力）
     B. 补高音语料 → 重训 → 重建索引（根治，但需要重训）

用法（方案 A 的批量体检 + 方案 B 的语料体检）：
  python fix_highnotes.py audit --corpus <语料目录> --model <权重> --index <索引>
  python fix_highnotes.py plan --corpus <语料目录> --target-hz 700
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rvcx import paths as P  # noqa: E402

sys.path.insert(0, str(P.REPO))
from infer.rmvpe import RMVPE  # noqa: E402

SR = 16000


def get_rmvpe():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return RMVPE(str(P.REPO / "assets" / "rmvpe" / "rmvpe.pt"), False, device=dev)


def f0_of_file(p, rmv):
    y, s = sf.read(str(p), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    if s != SR:
        import librosa
        y = librosa.resample(y, orig_sr=s, target_sr=SR)
    f0 = np.asarray(rmv.infer_from_audio(y, thred=0.03))
    return f0


def audit_corpus(corpus_dir, rmv):
    """体检语料的音域覆盖 —— 决定索引能不能在高音区正确检索。"""
    fs = sorted(glob.glob(os.path.join(corpus_dir, "**", "*.wav"), recursive=True))
    vals = []
    per_file = []
    for f in fs:
        try:
            f0 = f0_of_file(f, rmv)
        except Exception as e:
            print(f"  跳过 {os.path.basename(f)}: {e}")
            continue
        v = f0[f0 > 0]
        if v.size == 0:
            continue
        vals.append(v)
        per_file.append({"file": os.path.basename(f),
                         "max_hz": round(float(v.max()), 1),
                         "median_hz": round(float(np.median(v)), 1),
                         "n_over_600": int((v > 600).sum()),
                         "n_over_700": int((v > 700).sum())})
    if not vals:
        return None, []
    allv = np.concatenate(vals)
    rep = {
        "corpus_dir": corpus_dir,
        "files": len(allv),
        "f0_median_hz": round(float(np.median(allv)), 1),
        "f0_p95_hz": round(float(np.percentile(allv, 95)), 1),
        "f0_p99_hz": round(float(np.percentile(allv, 99)), 1),
        "f0_p999_hz": round(float(np.percentile(allv, 99.9)), 1),
        "f0_max_hz": round(float(allv.max()), 1),
        "pct_over_500hz": round(float((allv > 500).mean() * 100), 4),
        "pct_over_600hz": round(float((allv > 600).mean() * 100), 4),
        "pct_over_700hz": round(float((allv > 700).mean() * 100), 4),
        "n_frames_over_700hz": int((allv > 700).sum()),
        "total_minutes": None,
    }
    # 统计总时长
    tot = 0.0
    for f in fs:
        try:
            tot += sf.info(f).duration
        except Exception:
            pass
    rep["total_minutes"] = round(tot / 60, 2)
    return rep, per_file


def plan(corpus_dir, target_hz):
    """给出补高音语料的具体方案。"""
    rmv = get_rmvpe()
    rep, per_file = audit_corpus(corpus_dir, rmv)
    if rep is None:
        print("语料为空")
        return
    need_hz = target_hz
    have = rep["f0_p999_hz"]
    gap_semitones = round(12 * np.log2(need_hz / max(have, 1)), 1)
    plan_rep = {
        "current": rep,
        "target_max_hz": need_hz,
        "gap_semitones": gap_semitones,
        "strategy": [
            f"当前语料 p99.9={have} Hz，要覆盖到 {need_hz} Hz 需补高 {gap_semitones} 个半音",
            "① 合成高音语料：edge-tts 的 SSML <prosody pitch='+Nh'/> 可整体升调，"
            "但会连带改变音色 ⇒ 用小比例（+8~+12 半音）的比例混入",
            "② 更干净：用 CosyVoice 的 speed/prompt 变化 + 挑选句尾上扬的长音",
            "③ 混入比例建议：新增高音样本占总量 10-15% 即可让索引覆盖 700-900 Hz",
            "④ 补完必须重跑 preprocess → rmvpe → HuBERT → filelist → train → train_index",
        ],
        "immediate_workaround": {
            "option_A": "推理时 --index-rate 0（已验证：缺陷段 410.5Hz → 769.7Hz 正确）",
            "option_B": "重建索引（现状：训练数据无高音样本，重建也没用 —— 必须先补语料）",
        },
    }
    print(json.dumps(plan_rep, ensure_ascii=False, indent=2))
    return plan_rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "plan"])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--target-hz", type=float, default=700.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "audit":
        rmv = get_rmvpe()
        rep, per_file = audit_corpus(a.corpus, rmv)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        out = {"summary": rep, "files": per_file}
    else:
        out = plan(a.corpus, a.target_hz)

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")


if __name__ == "__main__":
    main()
